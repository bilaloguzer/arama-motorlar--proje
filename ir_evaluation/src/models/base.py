from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np

class RetrievalModel(ABC):
    def __init__(self):
        self.doc_ids = []
        
    @abstractmethod
    def fit(self, corpus: Any, doc_ids: List[str] = None) -> 'RetrievalModel':
        """
        Fit the model to the document corpus.
        
        Args:
            corpus: List of documents (raw strings or preprocessed tokens depending on model)
            doc_ids: Optional list of document identifiers
        """
        pass
    
    @abstractmethod
    def score(self, query: Any) -> np.ndarray:
        """
        Calculate similarity scores for a query against all documents.
        
        Args:
            query: The query (raw string or tokens)
            
        Returns:
            np.ndarray: Array of scores corresponding to doc_ids
        """
        pass
    
    def retrieve(self, query: Any, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: The query (raw string or tokens)
            top_k: Number of documents to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        scores = self.score(query)
        if len(scores) == 0:
            return []
            
        # Get indices of top_k scores
        # Note: argpartition is faster than argsort for top-k
        k = min(top_k, len(scores))
        if k == 0:
            return []
            
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort the top k indices by score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]


