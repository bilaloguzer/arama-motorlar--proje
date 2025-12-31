import numpy as np
from typing import List, Union

def precision_at_k(y_true: List[bool], y_scores: List[float], k: int) -> float:
    """Proportion of relevant items in top-k results"""
    if len(y_scores) == 0:
        return 0.0
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    
    order = np.argsort(y_scores)[::-1][:k]
    return np.sum(y_true[order]) / k

def recall_at_k(y_true: List[bool], y_scores: List[float], k: int) -> float:
    """Proportion of total relevant items found in top-k"""
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    
    total_relevant = np.sum(y_true)
    if total_relevant == 0:
        return 0.0
    
    order = np.argsort(y_scores)[::-1][:k]
    return np.sum(y_true[order]) / total_relevant

def average_precision(y_true: List[bool], y_scores: List[float]) -> float:
    """Average of precision values at each relevant document position"""
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    
    num_relevant = np.sum(y_true)
    if num_relevant == 0:
        return 0.0
    
    precisions = []
    relevant_count = 0
    for i, is_rel in enumerate(y_true_sorted):
        if is_rel:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
            
    return np.mean(precisions) if precisions else 0.0

def mean_average_precision(y_true_list: List[List[bool]], y_scores_list: List[List[float]]) -> float:
    """MAP: Average of AP across all queries"""
    if not y_true_list:
        return 0.0
    return np.mean([average_precision(yt, ys) for yt, ys in zip(y_true_list, y_scores_list)])

def ndcg_at_k(y_true: List[Union[bool, int]], y_scores: List[float], k: int, method='exponential') -> float:
    """Normalized Discounted Cumulative Gain"""
    if len(y_scores) == 0:
        return 0.0
        
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    
    order = np.argsort(y_scores)[::-1][:k]
    y_true_sorted = y_true[order]
    
    discounts = np.log2(np.arange(2, len(y_true_sorted) + 2))
    
    if method == 'exponential':
        gains = (2 ** y_true_sorted - 1) / discounts
    else:
        gains = y_true_sorted / discounts
        
    dcg = np.sum(gains)
    
    # Ideal DCG
    ideal_order = np.sort(y_true)[::-1][:k]
    ideal_discounts = np.log2(np.arange(2, len(ideal_order) + 2))
    
    if method == 'exponential':
        ideal_gains = (2 ** ideal_order - 1) / ideal_discounts
    else:
        ideal_gains = ideal_order / ideal_discounts
        
    idcg = np.sum(ideal_gains)
    
    return dcg / idcg if idcg > 0 else 0.0


