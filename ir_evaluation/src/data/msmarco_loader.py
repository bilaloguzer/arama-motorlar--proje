import ir_datasets
from typing import List, Dict, Tuple
import os
import pickle

class MSMARCOLoader:
    def __init__(self, dataset_name: str = 'msmarco-passage/dev/small', cache_dir: str = 'ir_evaluation/data/processed'):
        """
        Load MS MARCO dataset.
        
        Options:
        - 'msmarco-passage/dev/small': ~7K docs (quick testing)
        - 'msmarco-passage/train': 8.8M passages (full dataset)
        - 'msmarco-passage/dev': 6.9K queries with relevance judgments
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load(self, max_docs: int = None):
        """
        Load MS MARCO dataset with SMART filtering - only loads docs relevant to queries.
        
        Args:
            max_docs: Target number of documents (None = load all)
            
        Returns:
            documents, doc_ids, queries, q_ids, qrels
        """
        cache_file = os.path.join(self.cache_dir, f'msmarco_cache_{max_docs or "all"}.pkl')
        
        # Check cache
        if os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Loading MS MARCO dataset: {self.dataset_name}")
        print("This may take a few minutes on first run...")
        
        try:
            dataset = ir_datasets.load(self.dataset_name)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nTrying alternative: msmarco-passage/dev")
            dataset = ir_datasets.load('msmarco-passage/dev')
        
        # STEP 1: Load qrels first to know which docs we need
        print("Processing relevance judgments...")
        qrels = {}
        relevant_doc_ids = set()
        
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            relevant_doc_ids.add(qrel.doc_id)
        
        print(f"✓ Found {len(relevant_doc_ids)} unique relevant documents")
        
        # STEP 2: Load documents (prioritize relevant ones)
        print("Processing documents...")
        documents = []
        doc_ids = []
        doc_id_set = set()
        
        # First pass: Load all relevant docs
        for doc in dataset.docs_iter():
            if doc.doc_id in relevant_doc_ids:
                documents.append(doc.text)
                doc_ids.append(doc.doc_id)
                doc_id_set.add(doc.doc_id)
                
                if len(documents) % 1000 == 0:
                    print(f"  Loaded {len(documents)} relevant documents...", end='\r')
                
                # Stop if we have enough
                if max_docs and len(documents) >= max_docs:
                    break
        
        # Second pass: Fill up to max_docs with non-relevant docs if needed
        if max_docs and len(documents) < max_docs:
            print(f"\n  Adding non-relevant docs to reach {max_docs}...")
            for doc in dataset.docs_iter():
                if doc.doc_id not in doc_id_set:
                    documents.append(doc.text)
                    doc_ids.append(doc.doc_id)
                    doc_id_set.add(doc.doc_id)
                    
                    if len(documents) >= max_docs:
                        break
        
        print(f"\n✓ Loaded {len(documents)} documents")
        
        # STEP 3: Load Queries and filter to those with relevant docs in our set
        print("Processing queries...")
        queries = []
        q_ids = []
        filtered_qrels = {}
        
        for query in dataset.queries_iter():
            q_id = query.query_id
            
            # Only include queries that have at least one relevant doc in our doc set
            if q_id in qrels:
                has_relevant_in_set = any(doc_id in doc_id_set for doc_id in qrels[q_id].keys())
                
                if has_relevant_in_set:
                    queries.append(query.text)
                    q_ids.append(q_id)
                    # Filter qrels to only docs in our set
                    filtered_qrels[q_id] = {
                        doc_id: rel for doc_id, rel in qrels[q_id].items() 
                        if doc_id in doc_id_set
                    }
        
        print(f"✓ Loaded {len(queries)} queries with relevant documents in corpus")
        print(f"✓ Filtered to {sum(len(v) for v in filtered_qrels.values())} relevant judgments")
        
        result = (documents, doc_ids, queries, q_ids, filtered_qrels)
        
        # Cache for next time
        print(f"Caching data to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result


class KaggleDatasetLoader:
    """
    Template for loading Kaggle datasets.
    User needs to download CSV/JSON from Kaggle first.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_amazon_reviews(self):
        """
        Load Amazon Product Reviews dataset from Kaggle.
        Expected format: CSV with columns [product_id, title, description, reviews]
        """
        import pandas as pd
        
        print(f"Loading Amazon Reviews from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Use product descriptions as documents
        documents = (df['title'] + " " + df['description']).tolist()
        doc_ids = df['product_id'].astype(str).tolist()
        
        # Create synthetic queries from reviews (first sentence of each review)
        queries = df['reviews'].str.split('.').str[0].tolist()
        q_ids = [f"q_{i}" for i in range(len(queries))]
        
        # Create synthetic qrels (each query is relevant to its product)
        qrels = {q_id: {doc_id: 1} for q_id, doc_id in zip(q_ids, doc_ids)}
        
        print(f"✓ Loaded {len(documents)} products")        
        return documents, doc_ids, queries, q_ids, qrels

def load_msmarco_subset(num_docs=10000, num_queries=100, dataset_name='msmarco-passage/dev/small'):
    """
    Convenience function to load MS MARCO subset.
    
    Args:
        num_docs: Number of documents to load
        num_queries: Number of queries to load
        dataset_name: MS MARCO dataset variant
        
    Returns:
        Tuple of (docs_dict, queries_dict, qrels_dict)
    """
    loader = MSMARCOLoader(dataset_name=dataset_name)
    docs_list, doc_ids, queries_list, q_ids, qrels = loader.load(max_docs=num_docs)
    
    # Convert to dictionaries
    docs = {doc_id: doc for doc_id, doc in zip(doc_ids, docs_list)}
    queries_dict = {q_id: query for q_id, query in zip(q_ids[:num_queries], queries_list[:num_queries])}
    
    # Filter qrels to only include the queries we're using
    filtered_qrels = {q_id: qrels[q_id] for q_id in queries_dict.keys() if q_id in qrels}
    
    return docs, queries_dict, filtered_qrels


