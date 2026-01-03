import os
import requests
import tarfile
from typing import List, Dict, Tuple

class DataLoader:
    def __init__(self, data_dir: str = 'ir_evaluation/data/raw'):
        """
        Initialize DataLoader with a specific dataset name.
        """
        self.data_dir = data_dir
        self.cisi_url = "http://ir.dcs.gla.ac.uk/resources/test_collections/cisi/cisi.tar.gz"

    def download_cisi(self):
        """Downloads and extracts CISI dataset if not present."""
        os.makedirs(self.data_dir, exist_ok=True)
        tar_path = os.path.join(self.data_dir, "cisi.tar.gz")
        
        # Download
        if not os.path.exists(tar_path):
            print(f"Downloading CISI dataset from {self.cisi_url}...")
            response = requests.get(self.cisi_url, stream=True)
            if response.status_code == 200:
                with open(tar_path, 'wb') as f:
                    f.write(response.raw.read())
            else:
                raise ValueError("Failed to download CISI dataset.")
        
        # Extract
        if not os.path.exists(os.path.join(self.data_dir, "CISI.ALL")):
            print("Extracting CISI dataset...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.data_dir)

    def _parse_cisi_docs(self, filepath):
        docs = []
        doc_ids = []
        with open(filepath, 'r') as f:
            content = f.read()
        
        # CISI format: .I <id> \n .T <title> ...
        entries = content.split('.I ')
        for entry in entries[1:]: # Skip first empty split
            lines = entry.split('\n')
            doc_id = lines[0].strip()
            
            # Simple parsing (grab everything as text for now)
            text = " ".join(lines[1:])
            # Remove control characters like .T, .A, .W
            clean_text = text.replace('.T', '').replace('.A', '').replace('.W', '').replace('.X', '')
            
            docs.append(clean_text)
            doc_ids.append(doc_id)
        return docs, doc_ids

    def _parse_cisi_queries(self, filepath):
        queries = []
        q_ids = []
        with open(filepath, 'r') as f:
            content = f.read()
            
        entries = content.split('.I ')
        for entry in entries[1:]:
            lines = entry.split('\n')
            q_id = lines[0].strip()
            text = " ".join(lines[1:])
            clean_text = text.replace('.W', '').replace('.T', '').replace('.B', '').replace('.A', '')
            queries.append(clean_text)
            q_ids.append(q_id)
        return queries, q_ids

    def _parse_cisi_qrels(self, filepath):
        qrels = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    qid = parts[0]
                    did = parts[1]
                    # CISI qrels don't have scores, just binary relevance implied
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = 1
        return qrels

    def load(self):
        """
        Loads the CISI dataset manually.
        """
        self.download_cisi()
        
        print("Parsing documents...")
        docs, doc_ids = self._parse_cisi_docs(os.path.join(self.data_dir, "CISI.ALL"))
        
        print("Parsing queries...")
        queries, q_ids = self._parse_cisi_queries(os.path.join(self.data_dir, "CISI.QRY"))
        
        print("Parsing qrels...")
        qrels = self._parse_cisi_qrels(os.path.join(self.data_dir, "CISI.REL"))
        
        print(f"Loaded {len(docs)} documents, {len(queries)} queries.")
        return docs, doc_ids, queries, q_ids, qrels

def load_cisi_dataset(data_dir='ir_evaluation/data/raw'):
    """Convenience function to load CISI dataset"""
    loader = DataLoader(data_dir=data_dir)
    docs_list, doc_ids, queries_list, q_ids, qrels = loader.load()
    
    # Convert to dictionaries for easier lookup
    docs = {doc_id: doc for doc_id, doc in zip(doc_ids, docs_list)}
    queries = {q_id: query for q_id, query in zip(q_ids, queries_list)}
    
    return docs, queries, qrels

