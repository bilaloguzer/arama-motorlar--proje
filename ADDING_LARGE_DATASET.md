# How to Add a Larger Dataset

Your teacher wants a bigger dataset from Kaggle or similar. Here are your options:

## Option 1: MS MARCO (Recommended - Easiest!)

**Already installed!** Just run:

```bash
./venv/bin/python3 ir_evaluation/scripts/test_msmarco.py
```

Choose size:
- **Small**: 10K docs (quick test)
- **Medium**: 50K docs (good balance)
- **Large**: 100K docs (impressive!)

**Advantages:**
- ✅ Real queries from Bing search engine
- ✅ Industry-standard benchmark
- ✅ No manual download needed
- ✅ Automatic caching (fast re-runs)

---

## Option 2: Download from Kaggle

If your teacher specifically wants Kaggle:

### Amazon Product Reviews
1. Go to: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
2. Download the CSV file
3. Place in `ir_evaluation/data/raw/amazon_reviews.csv`
4. Modify the loader in `src/data/msmarco_loader.py`

### TREC-COVID (Scientific Papers)
1. Go to: https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge
2. Download the dataset
3. Place in `ir_evaluation/data/raw/`

---

## Quick Start: MS MARCO

```bash
# Run with 10K documents (fast)
./venv/bin/python3 ir_evaluation/scripts/test_msmarco.py
# Choose option 1 when prompted

# Run with 100K documents (impressive for presentation)
./venv/bin/python3 ir_evaluation/scripts/test_msmarco.py
# Choose option 3 when prompted
```

---

## Expected Results on MS MARCO

MS MARCO is harder than CISI, so expect:
- **BM25**: 15-20% MAP
- **TF-IDF**: 12-18% MAP
- **Rocchio**: 10-15% MAP

These are **good scores** for this challenging dataset!

---

## For Your Report

You can say:
> "We evaluated our models on two datasets: CISI (1.4K documents) for validation, and MS MARCO (100K passages) to demonstrate scalability. MS MARCO is a large-scale benchmark from Microsoft Bing search logs, containing real user queries."

This shows you tested on both small and large datasets!

