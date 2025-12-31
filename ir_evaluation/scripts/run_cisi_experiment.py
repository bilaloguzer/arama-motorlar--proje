import sys
import os
import json
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import DataLoader
from src.preprocessing.preprocessor import TextPreprocessor
from src.models.tfidf_model import TFIDFRetriever
from src.models.bm25_model import BM25Retriever
from src.models.rocchio_model import RocchioRetriever
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import precision_at_k, ndcg_at_k, average_precision

def main():
    print("Initializing CISI Experiment...")
    
    # 1. Load Data
    # ------------
    try:
        loader = DataLoader('cisi')
        documents, doc_ids, queries, q_ids, qrels = loader.load()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'ir_datasets' is installed: pip install ir_datasets")
        return

    # 2. Preprocessing
    # ----------------
    print("\nPreprocessing data (this may take a while)...")
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    
    # Preprocess documents
    # For BM25, we might want tokens. For TF-IDF/Rocchio, we often use strings (vectorizer tokenizes internally).
    # But for consistency and control, let's preprocess to clean strings.
    
    cleaned_docs = []
    print("Cleaning documents...")
    for doc in documents:
        tokens = preprocessor.preprocess(doc)
        cleaned_docs.append(" ".join(tokens))
        
    cleaned_queries = []
    print("Cleaning queries...")
    for q in queries:
        tokens = preprocessor.preprocess(q)
        cleaned_queries.append(" ".join(tokens))
    
    # 3. Evaluate Models
    # ------------------
    results = {}
    
    # A. TF-IDF
    print("\nEvaluating TF-IDF...")
    tfidf = TFIDFRetriever()
    tfidf.fit(cleaned_docs, doc_ids)
    
    evaluator_tfidf = Evaluator(tfidf)
    results['tfidf'] = evaluator_tfidf.evaluate(cleaned_queries, qrels, q_ids)
    print(f"TF-IDF MAP: {results['tfidf']['map']:.4f}")
    
    # B. BM25
    print("\nEvaluating BM25...")
    # Pass split to ensure it gets tokens from our already cleaned strings
    bm25 = BM25Retriever(preprocessor=lambda x: x.split())
    bm25.fit(cleaned_docs, doc_ids)
    
    evaluator_bm25 = Evaluator(bm25)
    results['bm25'] = evaluator_bm25.evaluate(cleaned_queries, qrels, q_ids)
    print(f"BM25 MAP: {results['bm25']['map']:.4f}")
    
    # C. Rocchio
    print("\nEvaluating Rocchio (with Pseudo-Relevance Feedback)...")
    rocchio = RocchioRetriever()
    rocchio.fit(cleaned_docs, doc_ids)
    
    rocchio_metrics = {
        'map': [], 'p@5': [], 'ndcg@5': []
    }
    
    print("Running Rocchio PRF loop...")
    for i, (q_text, q_id) in enumerate(zip(cleaned_queries, q_ids)):
        if i % 10 == 0:
            print(f"Processing query {i}/{len(queries)}...", end='\r')
            
        if q_id not in qrels:
            continue
            
        # 1. Initial search & PRF
        rocchio.modified_query = None 
        rocchio.pseudo_relevance_feedback(q_text, num_feedback=3, top_k=100)
        
        # 2. Score again
        scores = rocchio.score(q_text)
        
        # 3. Calculate metrics
        relevant_docs = qrels[q_id]
        y_true = [relevant_docs.get(did, 0) for did in rocchio.doc_ids]
        y_scores = scores
        
        rocchio_metrics['map'].append(average_precision(y_true, y_scores))
        rocchio_metrics['p@5'].append(precision_at_k(y_true, y_scores, 5))
        rocchio_metrics['ndcg@5'].append(ndcg_at_k(y_true, y_scores, 5))
            
    results['rocchio_prf'] = {k: np.mean(v) for k, v in rocchio_metrics.items()}
    print(f"Rocchio MAP: {results['rocchio_prf']['map']:.4f}")
    
    # 4. Save Results
    # ------------------
    os.makedirs('ir_evaluation/results/metrics', exist_ok=True)
    with open('ir_evaluation/results/metrics/cisi_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\nExperiment completed. Results saved to ir_evaluation/results/metrics/cisi_results.json")

if __name__ == "__main__":
    main()

