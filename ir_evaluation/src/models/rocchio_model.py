import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
from .base import RetrievalModel

class RocchioRetriever(RetrievalModel):
    def __init__(self, alpha=1.0, beta=0.75, gamma=0.15):
        super().__init__()
        self.alpha = alpha  # Original query weight
        self.beta = beta    # Relevant documents weight
        self.gamma = gamma  # Non-relevant documents weight
        # Using standard TF-IDF for the underlying representation
        self.vectorizer = TfidfVectorizer(norm='l2', stop_words='english')
        self.doc_vectors = None
        self.query_vector = None
        self.modified_query = None
        
    def fit(self, documents: List[str], doc_ids: List[str] = None) -> 'RocchioRetriever':
        """
        Fit the Rocchio model (vectorizes documents).
        Args:
            documents: List of raw document strings
            doc_ids: Optional list of document IDs
        """
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        return self
    
    def score(self, query: str) -> np.ndarray:
        """
        Calculate scores for a query. 
        Uses modified query if available, otherwise original query.
        Args:
            query: Raw query string
        """
        if self.doc_vectors is None:
            raise ValueError("Model must be fit before scoring")

        # If we have a modified query from feedback, use it.
        # Otherwise, vectorize the query.
        # Note: This stateful behavior (modified_query) implies this model instance 
        # is tied to a specific query session.
        if self.modified_query is not None:
             target_query = self.modified_query
        else:
            self.query_vector = self.vectorizer.transform([query]).toarray().flatten()
            target_query = self.query_vector
            
        similarities = cosine_similarity([target_query], self.doc_vectors).flatten()
        return similarities
    
    def apply_feedback(self, relevant_ids: List[str], non_relevant_ids: List[str] = None) -> 'RocchioRetriever':
        """
        Apply Rocchio formula: q_new = α*q + β*centroid(rel) - γ*centroid(nonrel)
        Updates self.modified_query.
        """
        if self.query_vector is None:
            raise ValueError("Must perform an initial search/score before applying feedback")

        relevant_indices = [self.doc_ids.index(did) for did in relevant_ids if did in self.doc_ids]
        
        rel_centroid = np.zeros(self.doc_vectors.shape[1])
        if relevant_indices:
             rel_centroid = np.asarray(self.doc_vectors[relevant_indices].mean(axis=0)).flatten()
        
        nonrel_centroid = np.zeros(self.doc_vectors.shape[1])
        if non_relevant_ids:
            nonrel_indices = [self.doc_ids.index(did) for did in non_relevant_ids if did in self.doc_ids]
            if nonrel_indices:
                nonrel_centroid = np.asarray(self.doc_vectors[nonrel_indices].mean(axis=0)).flatten()
        
        # Apply Rocchio formula
        self.modified_query = (
            self.alpha * self.query_vector +
            self.beta * rel_centroid -
            self.gamma * nonrel_centroid
        )
        
        # Zero out negative weights (standard practice)
        self.modified_query = np.maximum(self.modified_query, 0)
        return self
    
    def pseudo_relevance_feedback(self, query: str, num_feedback: int = 10, top_k: int = 100):
        """
        Blind feedback: assume top-k initial results are relevant.
        """
        # Reset any previous feedback
        self.modified_query = None
        
        # Initial search to get top results
        scores = self.score(query)
        top_indices = np.argsort(scores)[::-1][:num_feedback]
        pseudo_relevant_ids = [self.doc_ids[i] for i in top_indices]
        
        # Apply feedback
        self.apply_feedback(pseudo_relevant_ids, non_relevant_ids=[])
        
        # Rerank is implicitly done by calling score() again or retrieve()
        return self


