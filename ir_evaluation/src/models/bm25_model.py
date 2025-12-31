from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import numpy as np
from typing import List, Union, Literal
from .base import RetrievalModel

class BM25Retriever(RetrievalModel):
    def __init__(self, k1=1.5, b=0.75, variant: Literal['okapi', 'l', 'plus'] = 'okapi', preprocessor=None):
        super().__init__()
        self.k1 = k1
        self.b = b
        self.variant = variant
        self.preprocessor = preprocessor
        self.bm25_class = {
            'okapi': BM25Okapi, 
            'l': BM25L, 
            'plus': BM25Plus
        }.get(variant, BM25Okapi)
        self.bm25 = None
        
    def fit(self, corpus: Union[List[str], List[List[str]]], doc_ids: List[str] = None) -> 'BM25Retriever':
        """
        Fit the BM25 model.
        Args:
            corpus: List of raw document strings OR List of list of tokens
            doc_ids: Optional list of document IDs
        """
        self.doc_ids = doc_ids or [str(i) for i in range(len(corpus))]
        
        # specific handling if we received strings but need tokens
        if len(corpus) > 0 and isinstance(corpus[0], str):
            if self.preprocessor:
                tokenized_corpus = [self.preprocessor(doc) for doc in corpus]
            else:
                tokenized_corpus = [doc.split() for doc in corpus]
        else:
            tokenized_corpus = corpus
            
        self.bm25 = self.bm25_class(tokenized_corpus, k1=self.k1, b=self.b)
        return self
    
    def score(self, query: Union[str, List[str]]) -> np.ndarray:
        """
        Calculate scores for a query.
        Args:
            query: Raw query string OR List of query tokens
        """
        if self.bm25 is None:
            raise ValueError("Model must be fit before scoring")
            
        if isinstance(query, str):
            if self.preprocessor:
                tokenized_query = self.preprocessor(query)
            else:
                tokenized_query = query.split()
        else:
            tokenized_query = query
            
        scores = self.bm25.get_scores(tokenized_query)
        return np.array(scores)

