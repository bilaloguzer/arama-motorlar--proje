from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from typing import List, Union
from .base import RetrievalModel

class TFIDFRetriever(RetrievalModel):
    def __init__(self, ngram_range=(1, 2), min_df=2, max_df=0.9, 
                 sublinear_tf=True, max_features=50000):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,    # Capture phrases with bigrams
            min_df=min_df,               # Remove rare noise terms
            max_df=max_df,               # Remove corpus-wide stop words
            sublinear_tf=sublinear_tf,   # Apply log(1+tf) dampening
            max_features=max_features,   # Limit vocabulary for efficiency
            norm='l2'                    # Enable cosine via dot product
        )
        self.tfidf_matrix = None
    
    def fit(self, documents: List[str], doc_ids: List[str] = None) -> 'TFIDFRetriever':
        """
        Fit the TF-IDF model.
        Args:
            documents: List of raw document strings
            doc_ids: Optional list of document IDs
        """
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        return self
    
    def score(self, query: str) -> np.ndarray:
        """
        Calculate scores for a query.
        Args:
            query: Raw query string
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model must be fit before scoring")
            
        query_vec = self.vectorizer.transform([query])
        # linear_kernel is faster than cosine_similarity for L2-normalized vectors
        # Returns shape (1, n_docs), flatten to (n_docs,)
        similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        return similarities


