#!/usr/bin/env python3
"""
Quick test script to verify the demo works before launching
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("🧪 Testing Demo Components...")
print("-" * 50)

# Test 1: Import dependencies
print("\n1️⃣ Testing imports...")
try:
    import streamlit
    print("   ✅ Streamlit imported")
except ImportError as e:
    print(f"   ❌ Streamlit import failed: {e}")
    sys.exit(1)

try:
    from src.models.tfidf_model import TFIDFRetriever
    from src.models.bm25_model import BM25Retriever
    from src.models.rocchio_model import RocchioRetriever
    print("   ✅ Models imported")
except ImportError as e:
    print(f"   ❌ Model import failed: {e}")
    sys.exit(1)

try:
    from src.preprocessing.preprocessor import TextPreprocessor
    print("   ✅ Preprocessor imported")
except ImportError as e:
    print(f"   ❌ Preprocessor import failed: {e}")
    sys.exit(1)

try:
    from src.data.loader import load_cisi_dataset
    print("   ✅ Data loader imported")
except ImportError as e:
    print(f"   ❌ Data loader import failed: {e}")
    sys.exit(1)

# Test 2: Load dataset
print("\n2️⃣ Testing dataset loading...")
try:
    docs, queries, qrels = load_cisi_dataset()
    print(f"   ✅ Loaded {len(docs)} documents")
    print(f"   ✅ Loaded {len(queries)} queries")
    print(f"   ✅ Loaded {len(qrels)} qrels")
except Exception as e:
    print(f"   ❌ Dataset loading failed: {e}")
    print("\n   💡 Tip: Make sure CISI dataset has been downloaded")
    print("   Run: ./venv/bin/python3 ir_evaluation/scripts/test_cisi_simple.py")
    sys.exit(1)

# Test 3: Initialize models
print("\n3️⃣ Testing model initialization...")
try:
    doc_ids = list(docs.keys())[:10]  # Test with first 10 docs
    doc_texts = [docs[doc_id] for doc_id in doc_ids]
    
    tfidf = TFIDFRetriever()
    tfidf.fit(doc_texts, doc_ids)
    print("   ✅ TF-IDF model built")
    
    bm25 = BM25Retriever()
    bm25.fit(doc_texts, doc_ids)
    print("   ✅ BM25 model built")
    
    rocchio = RocchioRetriever()
    rocchio.fit(doc_texts, doc_ids)
    print("   ✅ Rocchio model built")
    
except Exception as e:
    print(f"   ❌ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test search
print("\n4️⃣ Testing search functionality...")
try:
    test_query = "information retrieval"
    scores = bm25.score(test_query)
    top_idx = scores.argmax()
    print(f"   ✅ Search completed successfully")
    print(f"   ✅ Top result: Doc {doc_ids[top_idx]}, Score: {scores[top_idx]:.4f}")
    
except Exception as e:
    print(f"   ❌ Search failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("✅ All tests passed! Demo is ready to launch.")
print("=" * 50)
print("\n🚀 To start the demo, run:")
print("   ./ir_evaluation/demo/launch_demo.sh")
print("\nOr manually:")
print("   streamlit run ir_evaluation/demo/app.py")
print()

