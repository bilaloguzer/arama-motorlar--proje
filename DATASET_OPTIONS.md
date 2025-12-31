# Recommended Datasets for IR Evaluation

## Large-Scale Options

### 1. MS MARCO (Recommended)
- **Size**: 8.8M passages OR 3.2M documents
- **Queries**: 500K+ training, 6.9K dev
- **Source**: Microsoft Bing search logs
- **Access**: Via `ir_datasets` (already installed!)
- **Subset**: Can use smaller "dev" split (~10K docs)

```python
import ir_datasets
dataset = ir_datasets.load("msmarco-passage/dev/small")
```

### 2. Kaggle: TREC-COVID Dataset
- **Size**: ~400K scientific papers about COVID
- **Queries**: 50 queries from medical experts
- **Download**: https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge

### 3. Kaggle: Amazon Product Reviews
- **Size**: 1M+ product descriptions
- **Can create synthetic queries from reviews**
- **Download**: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

### 4. Kaggle: Wikipedia Passages
- **Size**: Variable (100K - 5M passages)
- **Good for general domain IR**
- **Search**: "wikipedia passages" on Kaggle

### 5. Natural Questions (Google)
- **Size**: 100K+ real Google queries
- **Documents**: Wikipedia articles
- **Access**: Via `ir_datasets` or TensorFlow Datasets

## Recommended Approach

### For Demonstration (10K-50K docs):
Use **MS MARCO dev/small** - perfect size, real queries, professional dataset

### For "Big Data" Claim (100K+ docs):
Use **MS MARCO passage subset** or **Kaggle TREC-COVID**

### For Kaggle Requirement:
Download **Amazon Reviews** or **Wikipedia dataset** from Kaggle directly

## Implementation Strategy

1. Start with MS MARCO small (10K docs) to verify code works
2. Then scale to 100K+ documents
3. Compare performance and runtime
4. Show how BM25 scales better than TF-IDF on large data

