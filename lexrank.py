import pandas as pd
import networkx as nx
import numpy as np
import re, string, random

from stopwords import STOPWORDS

from math import log10
from collections import defaultdict

from scipy.sparse import find
from sklearn.preprocessing import binarize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def clean_string(string):
    """  Remove everything but whitespace, the alphabet. Separate apostrophes for stopwords  """
    string = re.sub(r"[^a-z\s']", '', string.lower())
    string = re.sub(r"[']+", ' ', string)
    return string


class LexRank:
    def __init__(self, method, threshold=None):
        """ Extract keyphrase from an input document using LexRank (pagerank on cosine similarity graph)
        Cosine similarity method must be chosen from ['naive', 'idf-mod', 'tfidf']
        
        Input: list (document) containing strings / utterances / sentences
        Output: reranked list according to salience
        
        Usage:
            >> corpus = load_kathy_corpus('../data/')
            >> keyphrases = LexRank('idf-mod')
            >> keyphrases(speakers = ['k', 'K', 'kathy', 'Kathy', 'cathy', 'Cathy'], use_main = False)
        """
        if method == 'naive':
            self.method = self._naive_sim
            self.threshold = 6.5
        elif method == 'tfidf':
            self.method = self._tfidf_sim
            self.threshold = 1e-2
        elif method == 'idf-mod':
            self.method = self._idf_mod_sim
            self.threshold = 5e-5
        else:
            print("Invalid method from ['naive', 'tfidf', 'idf-mod']. Defaulting to naive.")
            self.method = self._naive_sim
        
        # If user specified a threshold, use it (not recommended)
        if threshold:
            self.threshold = threshold
        
    def __call__(self, *args, **kwargs):
        """ Return keyphrases """
        return self.get_keyphrases(*args, **kwargs)
        
    def get_keyphrases(self, document, 
                       include_scores=False,             
                       maxlen=None):
        """ Get keyphrases from a document using LexRank
        
        Speakers, use_main use case similar to in keynet.py """
        
        # Incoporate documnet being considered
        self.document = document
        
        # Initialize document counts, tfidif scores        
        self.init_counts()
        
        # Build graph of sentences, edges are cossim
        network_graph = self._build_graph()
        
        # Run PageRank on the graph
        ranked = nx.pagerank_scipy(network_graph)
        ranked = [(val, text) for val, text in ranked.items()]
                
        # Sort results by score
        sort_ranked = sorted(ranked, key=lambda t: t[1], reverse=True)

        # Keep only results up to some maximum length in tokens
        if maxlen:
            sort_ranked = [t for t in sort_ranked if len(t[0].split()) < maxlen]

        # For outputting without scores
        if not include_scores:
            sort_ranked = [s[0] for s in sort_ranked]
        
        return sort_ranked
    
    def init_counts(self, pattern=r"(?u)\b\w+\b", stop_words=STOPWORDS):
        """ Initialize matrix counts, stats. Pattern keeps 1 character tokens """
        self.cv = CountVectorizer(token_pattern=pattern, stop_words=stop_words)
        self.counts = self.cv.fit_transform(self.document)
        self.b_counts = binarize(self.counts)
        
        self.tfidf = TfidfTransformer()
        self.matrix = self.tfidf.fit_transform(self.counts)
        
    def _build_graph(self):
        """ Build graph of sentences where edges are cosine similarities """
        network_graph = nx.Graph()
        network_graph.clear()

        # add all the nodes
        network_graph.add_nodes_from(self.document)
        
        # build edges using the similarities data
        indices, sims = self.method()
        edges = self._build_edges(indices, sims, self.threshold)

        # add in edges
        network_graph.add_edges_from(edges)

        return network_graph
    
    def _build_edges(self, indices, sims, threshold):
        """ Build graph edges using a similarity threshold for inclusion """
        edges = [[self.document[i1], # text sentence 1
                  self.document[i2], # text sentence 2
                  {'similarity': sim}] # similarity
                for i1, i2, sim in zip(*indices, sims) if sim > threshold]
        
        return edges
    
    def _naive_sim(self):
        """ Naive cossim defined as the number of unique words shared
        by two sentences. 
        
        Recommended threshold:  """
        # Get cooccurrence statistics
        cooccurs = np.dot(self.b_counts, self.b_counts.T)

        # Uppter triangular indices, values
        indices = np.triu_indices_from(cooccurs, k=1)
        raw_sims = np.asarray(cooccurs[indices]).flatten()

        # Naive similarity edge values
        sims = [self._safe_naive_edge(sim) for sim in raw_sims]
                
        return indices, sims
    
    def _tfidf_sim(self):
        """ Fast tfidf. 
        
        Recommended threshold: 1e-2 """
        # Get tf-idf similarities
        tf_idf = np.dot(self.matrix, self.matrix.T)
        
        # Upper triangular indices, values d
        indices = np.triu_indices_from(tf_idf, k=1)
        sims = np.asarray(tf_idf[indices]).flatten()
                
        return indices, sims
    
    def _idf_mod_sim(self, eps=1e-6):
        """ Fast idf-modified-cosine. 
        
        Recommend threshold:  """
        # Term frequencies * their inverse doc freqs (not tf-idf)
        tf_xidf = np.dot(self.counts, self.tfidf._idf_diag) 

        # Numerator: tf_{w,x} * tf_{w,y} * (idf_{w}**2)
        numerator = np.dot(tf_xidf, tf_xidf.T) 

        # Denominator: np.sqrt((tf_{w}*idf_{w})^2), eps for non-zero dividing
        denominator = np.sqrt(np.dot(self.counts, tf_xidf.T) ** 2).diagonal() + eps 

        # Upper triangular indices, values
        indices = np.triu_indices_from(numerator, k=1)
        flat_denom = [denominator[i1]*denominator[i2] for i1, i2 in zip(*indices)]
        sims = np.asarray(numerator[indices]).flatten() / flat_denom
        
        return indices, sims

    def _safe_naive_edge(self, sim):
        """ Safely compute the similarity score for naive, 
            _naive_sim / log length of each sentence added together. """
        if sim > 1:
            return sim / log10(sim)
        return 0

    def _prune_nodes(self, network_graph):
        """ Remove all nonzero nodes if threshold == 0 """
        nodes_to_remove = list(nx.isolates(network_graph))
        network_graph.remove_nodes_from(nodes_to_remove)
        return network_graph


class Search:
    def __init__(self, corpus, ngram=2):
        """ Efficiently search the corpus for occurrences of a unigram or bigram token.
        Input: token (unigram, bigram)
        Output: list of strings from the corpus where the cleaned token was found
        
        Usage:
            >> search = Search(corpus)
            >> search('Donald Trump') # case insensitive, allows up to ngram string
        """
        self.corpus = corpus
        self.cv, self.vectors = self._vectorize_corpus(ngram)
        
    def __call__(self, *args):
        return self.find_occurrences(*args)        

    def find_occurrences(self, token):
        """ Find all occurrences of a case-insensitive token """
        
        # Clean the token up, convert to ID, find occurrences in sparse matrix
        token = self._clean_token(token)
        try: 
            word_idx = self.cv.vocabulary_[token]
        except:
            print('ERROR: Word not found in vocabulary.')
        
        line_idx, _, _ = find(self.vectors[:, word_idx])
        
        # Return the occurrence sentences in the corpus where the token occurs
        occurrences = [self.corpus[idx] for idx in line_idx]
        
        return occurrences
    
    def _vectorize_corpus(self, ngram):
        """ Vectorize the corpus into a sparse matrix based on binary occurrence """
        cv = CountVectorizer(ngram_range=(1, ngram),  
                             stop_words=STOPWORDS, 
                             token_pattern=r"(?u)\b\w+\b", 
                             binary = True)

        vectors = cv.fit_transform(self.corpus)
        return cv, vectors
    
    def _clean_token(self, token):
        """  Remove everything but whitespace, the alphabet; separate apostrophes for stopwords """
        token = re.sub(r"[^a-z0-9\s]", '', token.lower())
        token = re.sub(r"[']+", ' ', token)
        return token


class Rank:
    """ Search a corpus for unigram/bigram occurrences, output LexRank rankings from documents 
    containing that phrase. Random sampling is used to maintain computational tractability if
    the found results are too large. Entropy is enforce in ranked outputs.
    
    Initialization: corpus
    Input: token (e.g. 'Donald Trump', 'hillary', 'asia', ...)
    Output: related radio segments containing the keyword, organized by salience
    
    Usage:
        >> radio = pd.read_csv('../data.csv', usecols=['sentences'])
        >> radio = [clean_string(i) for i in set(radio.sentences)]
        >> radrank = RadRank(radio, 'idf-mod')
        >> radrank('Donald Trump')
    """
    def __init__(self, radio, method, context=50, resrate=1000, subrate=500):
        
        # Resrate: number of samples from found segments in entire corpus
        self.resrate = resrate 
        
        # Subrate: number of samples from reorganized resrate samples
        self.subrate = subrate
        
        # Context: Contextual window of tokens around the word to use as output
        self.cntx = context
        
        # Initialization of search can take a while depending on ngram
        self.search = Search(radio)
        
        # LexRank (PageRank-based) reordering of output segments
        self.lexrank = LexRank(method)
    
    def __call__(self, *args, **kwargs):
        return self.rank(*args, **kwargs)
    
    def rank(self, string):
        """ Rank Public Talk Radio """
        subsamples = self._get_occurrences(string)
        ratings = self.lexrank(subsamples, include_scores=True)
        return ratings
    
    def _get_occurrences(self, string):
        """ (1) Search corpus for occurrences of a string
            (2) Sample from those occurrences
            (3) Extract contextual window around the string
            (4) Return subsamples of those context windows
        """
        self.string = string
        
        results = self.search(self.string)
        subradio = random.sample(results, min(len(results), self.resrate))
        regrouped = self._regroup_subradio(subradio)
        
        subsamples = []
        for sub in regrouped:
            ids = [idx for idx, j in enumerate(sub) if j == self.string]
            window = [sub[max(0, ix-self.cntx):ix+self.cntx] for ix in ids]
            subsamples.extend(window)
        
        subsamples = random.sample([' '.join(i) for i in subsamples], min(len(subsamples), self.subrate))
        
        return subsamples
            
    def _regroup_subradio(self, subradio):
        """ Redefine the context window around the input string by grouping if it's an ngram """
        subradio = [sub.split() for sub in subradio]
        output = []
        for sub in subradio: 
            idxs, j = [], 0
            for i in range(len(sub)-1):
                if sub[i] + ' '+ sub[i+1] == self.string:
                    idxs.append(j)
                    j+=2
                else:
                    idxs.append(j)
                    j+=1

            output.append([' '.join(sub[idxs[ix]:idxs[ix+1]]) for ix in range(len(idxs)-1)])
        return output
