import sys
import os
import json
import numpy as np
import re
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.msmarco_loader import MSMARCOLoader
from src.models.tfidf_model import TFIDFRetriever
from src.models.bm25_model import BM25Retriever
from src.models.rocchio_model import RocchioRetriever
from src.evaluation.evaluator import Evaluator

# Preprocessor with stemming (as per project proposal)
class PreprocessorWithStemming:
    def __init__(self):
        # Import here to avoid NLTK download issues at module level
        try:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
            self.use_stemming = True
        except:
            print("  Warning: NLTK not available, stemming disabled")
            self.stemmer = None
            self.use_stemming = False
            
        self.stop_words = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'or', 'not', 'been',
            'have', 'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should'
        ])
    
    def clean(self, text):
        """
        Preprocessing pipeline as per project proposal:
        1. Tokenization
        2. Stopword removal
        3. Stemming
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
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Apply stemming (Porter Stemmer as per proposal)
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return " ".join(tokens)

def main():
    print("=" * 70)
    print("MS MARCO Dataset IR Evaluation")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[1/4] Loading MS MARCO Dataset...")
    print("Choose dataset size:")
    print("  1. Small (10K docs) - Quick test")
    print("  2. Medium (50K docs) - Balanced")
    print("  3. Large (100K docs) - Full evaluation")
    
    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    max_docs_map = {"1": 10000, "2": 50000, "3": 100000}
    max_docs = max_docs_map.get(choice, 10000)
    
    try:
        loader = MSMARCOLoader('msmarco-passage/dev')
        documents, doc_ids, queries, q_ids, qrels = loader.load(max_docs=max_docs)
        
        # Filter queries that have relevance judgments
        valid_queries = []
        valid_q_ids = []
        for q, qid in zip(queries, q_ids):
            if qid in qrels:
                valid_queries.append(q)
                valid_q_ids.append(qid)
        
        queries = valid_queries[:100]  # Limit to 100 queries for speed
        q_ids = valid_q_ids[:100]
        
        print(f"✓ Using {len(documents)} documents and {len(queries)} queries")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nPlease ensure 'ir_datasets' is installed:")
        print("  ./venv/bin/pip install ir_datasets")
        return

    # 2. Preprocessing (with stemming as per project proposal)
    print("\n[2/4] Preprocessing data (with stemming)...")
    preprocessor = PreprocessorWithStemming()
    if preprocessor.use_stemming:
        print("  ✓ Stemming enabled (Porter Stemmer)")
    else:
        print("  ⚠ Stemming disabled (NLTK not available)")
    
    print("  Cleaning documents...")
    cleaned_docs = [preprocessor.clean(doc) for doc in documents]
    print("  Cleaning queries...")
    cleaned_queries = [preprocessor.clean(q) for q in queries]
    print(f"✓ Preprocessed all data")
    
    # 3. Evaluate Models
    results = {}
    start_time = datetime.now()
    
    print("\n[3/4] Evaluating Models...")
    
    # A. TF-IDF
    print("\n  [A] TF-IDF Model")
    tfidf_start = datetime.now()
    tfidf = TFIDFRetriever(min_df=2, max_df=0.95)
    tfidf.fit(cleaned_docs, doc_ids)
    
    evaluator_tfidf = Evaluator(tfidf)
    results['tfidf'] = evaluator_tfidf.evaluate(cleaned_queries, qrels, q_ids, k_values=[5, 10])
    tfidf_time = (datetime.now() - tfidf_start).total_seconds()
    results['tfidf']['time_seconds'] = tfidf_time
    
    print(f"      MAP: {results['tfidf']['map']:.4f}")
    print(f"      P@5: {results['tfidf']['p@5']:.4f}")
    print(f"      Time: {tfidf_time:.2f}s")
    
    # B. BM25
    print("\n  [B] BM25 Model")
    bm25_start = datetime.now()
    bm25 = BM25Retriever(k1=1.5, b=0.75, preprocessor=lambda x: x.split())
    bm25.fit(cleaned_docs, doc_ids)
    
    evaluator_bm25 = Evaluator(bm25)
    results['bm25'] = evaluator_bm25.evaluate(cleaned_queries, qrels, q_ids, k_values=[5, 10])
    bm25_time = (datetime.now() - bm25_start).total_seconds()
    results['bm25']['time_seconds'] = bm25_time
    
    print(f"      MAP: {results['bm25']['map']:.4f}")
    print(f"      P@5: {results['bm25']['p@5']:.4f}")
    print(f"      Time: {bm25_time:.2f}s")
    
    # C. Rocchio
    print("\n  [C] Rocchio Model")
    rocchio_start = datetime.now()
    rocchio = RocchioRetriever(alpha=1.0, beta=0.75, gamma=0.15)
    rocchio.fit(cleaned_docs, doc_ids)
    
    evaluator_rocchio = Evaluator(rocchio)
    results['rocchio'] = evaluator_rocchio.evaluate(cleaned_queries, qrels, q_ids, k_values=[5, 10])
    rocchio_time = (datetime.now() - rocchio_start).total_seconds()
    results['rocchio']['time_seconds'] = rocchio_time
    
    print(f"      MAP: {results['rocchio']['map']:.4f}")
    print(f"      P@5: {results['rocchio']['p@5']:.4f}")
    print(f"      Time: {rocchio_time:.2f}s")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # 4. Save Results
    print("\n[4/4] Saving Results...")
    os.makedirs('ir_evaluation/results/metrics', exist_ok=True)
    output_file = f'ir_evaluation/results/metrics/msmarco_{max_docs}_results.json'
    
    results['metadata'] = {
        'dataset': 'MS MARCO',
        'num_documents': len(documents),
        'num_queries': len(queries),
        'total_time_seconds': total_time
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    
    # Summary Table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - MS MARCO Dataset")
    print("=" * 70)
    print(f"Documents: {len(documents):,} | Queries: {len(queries)}")
    print("-" * 70)
    print(f"{'Model':<15} {'MAP':<10} {'P@5':<10} {'NDCG@10':<12} {'Time (s)':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        if model_name == 'metadata':
            continue
        print(f"{model_name.upper():<15} {metrics['map']:<10.4f} {metrics['p@5']:<10.4f} {metrics['ndcg@10']:<12.4f} {metrics['time_seconds']:<10.2f}")
    print("=" * 70)
    print(f"\nTotal evaluation time: {total_time:.2f} seconds")
    print("\n✓ Experiment completed successfully!")

if __name__ == "__main__":
    main()

