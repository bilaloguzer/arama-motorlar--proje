import sys
import os
import json
import numpy as np
from pprint import pprint

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.preprocessor import TextPreprocessor
from src.models.tfidf_model import TFIDFRetriever
from src.models.bm25_model import BM25Retriever
from src.models.rocchio_model import RocchioRetriever
from src.evaluation.evaluator import Evaluator

def main():
    print("Initializing Experiment...")
    
    # 1. Synthetic Data Setup
    # -----------------------
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models require large datasets for training.",
        "Natural language processing enables computers to understand text.",
        "Information retrieval systems search for relevant documents.",
        "Search engines use ranking algorithms like BM25 and PageRank.",
        "Python is a popular programming language for data science.",
        "Scikit-learn provides tools for machine learning in Python.",
        "Neural networks are inspired by the biological brain.",
        "Reinforcement learning learns through trial and error.",
        "Computer vision deals with how computers see images."
    ]
    doc_ids = [str(i) for i in range(len(documents))]
    
    queries = [
        "machine learning ai",
        "search engine algorithms",
        "python data science",
        "deep neural networks"
    ]
    q_ids = ["q1", "q2", "q3", "q4"]
    
    # Ground truth (q_id -> {doc_id: relevance})
    # Simple binary relevance for this synthetic test
    qrels = {
        "q1": {"0": 1, "1": 1, "6": 1, "8": 1},
        "q2": {"3": 1, "4": 1},
        "q3": {"5": 1, "6": 1},
        "q4": {"1": 1, "7": 1}
    }
    
    # 2. Preprocessing
    # ----------------
    print("\nPreprocessing data...")
    preprocessor = TextPreprocessor(use_stemming=True)
    
    # Note: TF-IDF often prefers raw text (it does its own tokenization usually), 
    # but we can pass preprocessed strings.
    # BM25 needs tokens.
    
    # Let's clean the documents into strings for TF-IDF/Rocchio
    # and tokens for BM25
    
    # Helper to get string back from tokens
    def clean_text(text):
        tokens = preprocessor.preprocess(text)
        return " ".join(tokens)
        
    cleaned_docs = [clean_text(d) for d in documents]
    cleaned_queries = [clean_text(q) for q in queries]
    
    # 3. Evaluate Models
    # ------------------
    results = {}
    
    # A. TF-IDF
    print("\nEvaluating TF-IDF...")
    tfidf = TFIDFRetriever()
    tfidf.fit(cleaned_docs, doc_ids)
    
    evaluator_tfidf = Evaluator(tfidf)
    results['tfidf'] = evaluator_tfidf.evaluate(cleaned_queries, qrels, q_ids)
    
    # B. BM25
    print("\nEvaluating BM25...")
    # We pass the preprocessor's preprocess method to BM25 so it can tokenize internally if needed,
    # or we can pass pre-tokenized data.
    # Since we already cleaned the docs into strings, let's just use simple split for BM25
    # or pass the token list.
    
    # Let's create a wrapper for BM25 that takes the cleaned strings and splits them
    bm25 = BM25Retriever(preprocessor=lambda x: x.split())
    bm25.fit(cleaned_docs, doc_ids)
    
    evaluator_bm25 = Evaluator(bm25)
    results['bm25'] = evaluator_bm25.evaluate(cleaned_queries, qrels, q_ids)
    
    # C. Rocchio
    print("\nEvaluating Rocchio (with Pseudo-Relevance Feedback)...")
    rocchio = RocchioRetriever()
    rocchio.fit(cleaned_docs, doc_ids)
    
    # For Rocchio, we need to perform the feedback loop.
    # The Evaluator class calls score(), which uses the current state.
    # We need a custom evaluation loop for Rocchio to apply PRF per query.
    
    rocchio_metrics = {
        'map': [], 'p@5': [], 'ndcg@5': []
    }
    
    from src.evaluation.metrics import precision_at_k, ndcg_at_k, average_precision
    
    print("Running Rocchio PRF loop...")
    for q_text, q_id in zip(cleaned_queries, q_ids):
        # 1. Initial search
        rocchio.modified_query = None # Reset
        
        # 2. Apply PRF (assume top 2 are relevant for this small dataset)
        rocchio.pseudo_relevance_feedback(q_text, num_feedback=2)
        
        # 3. Score again
        scores = rocchio.score(q_text)
        
        # 4. Calculate metrics
        if q_id in qrels:
            relevant_docs = qrels[q_id]
            y_true = [relevant_docs.get(did, 0) for did in rocchio.doc_ids]
            y_scores = scores
            
            rocchio_metrics['map'].append(average_precision(y_true, y_scores))
            rocchio_metrics['p@5'].append(precision_at_k(y_true, y_scores, 5))
            rocchio_metrics['ndcg@5'].append(ndcg_at_k(y_true, y_scores, 5))
            
    results['rocchio_prf'] = {k: np.mean(v) for k, v in rocchio_metrics.items()}
    
    # 4. Display Results
    # ------------------
    print("\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2))
    
    # Save results
    os.makedirs('ir_evaluation/results/metrics', exist_ok=True)
    with open('ir_evaluation/results/metrics/synthetic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\nExperiment completed successfully.")

if __name__ == "__main__":
    main()


