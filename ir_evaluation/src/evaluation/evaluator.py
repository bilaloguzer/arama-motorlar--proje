from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from ..models.base import RetrievalModel
from .metrics import precision_at_k, recall_at_k, ndcg_at_k, average_precision

class Evaluator:
    def __init__(self, model: RetrievalModel):
        self.model = model
    
    def evaluate(self, queries: List[str], qrels: Dict[str, Dict[str, int]], 
                 query_ids: List[str], k_values: List[int] = [1, 5, 10]):
        """
        Evaluate the model on a set of queries.
        
        Args:
            queries: List of query strings (raw)
            qrels: Dictionary mapping query_id -> {doc_id: relevance_score}
            query_ids: List of query IDs corresponding to queries list
            k_values: List of k values for metrics
            
        Returns:
            Dictionary of metrics averaged over all queries
        """
        metrics = {
            'map': [],
        }
        for k in k_values:
            metrics[f'p@{k}'] = []
            metrics[f'r@{k}'] = []
            metrics[f'ndcg@{k}'] = []
            
        for query_text, qid in tqdm(zip(queries, query_ids), total=len(queries), desc="Evaluating"):
            if qid not in qrels:
                continue
                
            relevant_docs = qrels[qid]
            
            # Get scores for all documents
            # Note: This assumes model.doc_ids aligns with the indices of scores
            scores = self.model.score(query_text)
            
            # Create y_true and y_scores
            # We need to map scores to the ground truth relevance
            y_true = []
            y_scores = []
            
            # Optimization: only consider documents with non-zero scores or known relevance
            # But for standard metrics we often need to rank all candidates.
            # Here we assume the model has scores for all docs in its index.
            
            for idx, doc_id in enumerate(self.model.doc_ids):
                score = scores[idx]
                rel = relevant_docs.get(doc_id, 0)
                y_true.append(rel)
                y_scores.append(score)
            
            # Calculate metrics for this query
            metrics['map'].append(average_precision(y_true, y_scores))
            
            for k in k_values:
                metrics[f'p@{k}'].append(precision_at_k(y_true, y_scores, k))
                metrics[f'r@{k}'].append(recall_at_k(y_true, y_scores, k))
                metrics[f'ndcg@{k}'].append(ndcg_at_k(y_true, y_scores, k))
                
        # Average metrics
        averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return averaged_metrics


