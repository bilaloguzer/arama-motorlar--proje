# Instructions to Fix NLTK Data (if needed)

If you encounter NLTK-related errors, run these commands:

```bash
# Remove corrupted NLTK data
rm -rf ~/nltk_data

# Re-download using Python
./venv/bin/python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

# Quick Test (Bypasses NLTK)

Run the simplified test script that uses basic preprocessing:

```bash
./venv/bin/python3 ir_evaluation/scripts/test_cisi_simple.py
```

This will test all three models (TF-IDF, BM25, Rocchio) on the CISI dataset.

