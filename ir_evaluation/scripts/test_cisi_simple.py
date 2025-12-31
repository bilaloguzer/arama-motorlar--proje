import sys
import os
import json
import numpy as np
import re

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import DataLoader
from src.models.tfidf_model import TFIDFRetriever
from src.models.bm25_model import BM25Retriever
from src.models.rocchio_model import RocchioRetriever
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import precision_at_k, ndcg_at_k, average_precision

# Preprocessor with stemming (as per project proposal)
class PreprocessorWithStemming:
    def __init__(self):
        # Import here to avoid issues
        try:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
            self.use_stemming = True
        except:
            print("  Warning: NLTK not available, stemming disabled")
            self.stemmer = None
            self.use_stemming = False
            
        # Standard English stopwords
        self.stop_words = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'or', 'not', 'been',
            'have', 'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should'
        ])
    
    def clean(self, text):
        """
        Complete preprocessing as per project proposal:
        1. Tokenization
        2. Stopword removal
        3. Stemming (Porter Stemmer)
        4. Normalization
        """
        if not isinstance(text, str):
            return ""
        
        # Normalize
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        tokens = [t for t in text.split() if t not in self.stop_words and len(t) > 2]
        
        # Apply stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return " ".join(tokens)

def main():
    print("=" * 60)
    print("CISI Dataset IR Evaluation")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/4] Loading CISI Dataset...")
    try:
        loader = DataLoader()
        documents, doc_ids, queries, q_ids, qrels = loader.load()
        print(f"✓ Loaded {len(documents)} documents and {len(queries)} queries")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # 2. Preprocessing (with stemming as per project proposal)
    print("\n[2/4] Preprocessing data (with stemming)...")
    preprocessor = PreprocessorWithStemming()
    if preprocessor.use_stemming:
        print("  ✓ Stemming enabled (Porter Stemmer)")
    else:
        print("  ⚠ Stemming disabled (NLTK not available)")
    
    cleaned_docs = [preprocessor.clean(doc) for doc in documents]
    cleaned_queries = [preprocessor.clean(q) for q in queries]
    print(f"✓ Preprocessed all documents and queries")
    
    # 3. Evaluate Models
    results = {}
    
    # A. TF-IDF
    print("\n[3/4] Evaluating Models...")
    print("\n  [A] TF-IDF Model")
    tfidf = TFIDFRetriever(min_df=1, max_df=0.95)  # Adjust for small corpus
    tfidf.fit(cleaned_docs, doc_ids)
    
    evaluator_tfidf = Evaluator(tfidf)
    results['tfidf'] = evaluator_tfidf.evaluate(cleaned_queries, qrels, q_ids, k_values=[5, 10])
    print(f"      MAP: {results['tfidf']['map']:.4f}")
    print(f"      P@5: {results['tfidf']['p@5']:.4f}")
    print(f"      NDCG@10: {results['tfidf']['ndcg@10']:.4f}")
    
    # B. BM25
    print("\n  [B] BM25 Model")
    bm25 = BM25Retriever(k1=1.5, b=0.75, preprocessor=lambda x: x.split())
    bm25.fit(cleaned_docs, doc_ids)
    
    evaluator_bm25 = Evaluator(bm25)
    results['bm25'] = evaluator_bm25.evaluate(cleaned_queries, qrels, q_ids, k_values=[5, 10])
    print(f"      MAP: {results['bm25']['map']:.4f}")
    print(f"      P@5: {results['bm25']['p@5']:.4f}")
    print(f"      NDCG@10: {results['bm25']['ndcg@10']:.4f}")
    
    # C. Rocchio (Simplified - without PRF for this test)
    print("\n  [C] Rocchio Model")
    rocchio = RocchioRetriever(alpha=1.0, beta=0.75, gamma=0.15)
    rocchio.fit(cleaned_docs, doc_ids)
    
    evaluator_rocchio = Evaluator(rocchio)
    results['rocchio'] = evaluator_rocchio.evaluate(cleaned_queries, qrels, q_ids, k_values=[5, 10])
    print(f"      MAP: {results['rocchio']['map']:.4f}")
    print(f"      P@5: {results['rocchio']['p@5']:.4f}")
    print(f"      NDCG@10: {results['rocchio']['ndcg@10']:.4f}")
    
    # 4. Save Results
    print("\n[4/4] Saving Results...")
    os.makedirs('ir_evaluation/results/metrics', exist_ok=True)
    output_file = 'ir_evaluation/results/metrics/cisi_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    
    # Summary Table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'MAP':<10} {'P@5':<10} {'NDCG@10':<10}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name.upper():<15} {metrics['map']:<10.4f} {metrics['p@5']:<10.4f} {metrics['ndcg@10']:<10.4f}")
    print("=" * 60)
    
    print("\n✓ Experiment completed successfully!")

if __name__ == "__main__":
    main()

