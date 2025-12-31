# CISI Dataset Evaluation Report

## Dataset Information
- **Name**: CISI (Computer and Information Science Abstracts)
- **Documents**: 1,460 scientific abstracts
- **Queries**: 112 information retrieval queries
- **Source**: University of Glasgow IR Test Collection

## Models Evaluated

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
- Uses scikit-learn's TfidfVectorizer
- Cosine similarity for ranking
- Parameters: min_df=1, max_df=0.95

### 2. BM25 (Okapi BM25)
- Probabilistic ranking function
- Parameters: k1=1.5, b=0.75
- Term frequency saturation and length normalization

### 3. Rocchio
- Query expansion using relevance feedback
- Parameters: α=1.0, β=0.75, γ=0.15
- Tested without pseudo-relevance feedback

## Results Summary

| Model | MAP | P@5 | NDCG@10 |
|-------|-----|-----|---------|
| TFIDF | 0.1824 | 0.3211 | 0.2968 |
| BM25 | 0.1863 | 0.3368 | 0.3138 |
| ROCCHIO | 0.1366 | 0.2789 | 0.2512 |


## Key Findings

### Best Performing Model: BM25
- **MAP**: 0.1863 (Mean Average Precision)
- **P@5**: 0.3368 (Precision at 5)
- **NDCG@10**: 0.3138 (Normalized Discounted Cumulative Gain at 10)

### Performance Analysis

1. **BM25 Superiority**: BM25 outperforms TF-IDF due to:
   - Term frequency saturation (prevents over-weighting repeated terms)
   - Document length normalization (fair comparison across documents)
   
2. **TF-IDF Competitiveness**: Close to BM25, showing that:
   - Simple vector space models remain effective
   - Cosine similarity provides good ranking
   
3. **Rocchio Performance**: Lower without pseudo-relevance feedback:
   - Needs actual relevance judgments for optimal performance
   - Could improve significantly with PRF enabled

## Visualizations

See the `results/figures/` directory for:
- Model comparison bar chart
- Metric breakdown
- Performance radar chart

## Recommendations

1. **For Production**: Use BM25 as the baseline ranker
2. **For Explainability**: TF-IDF offers simpler interpretation
3. **For Interactive Systems**: Rocchio with user feedback
4. **Next Steps**: 
   - Tune BM25 hyperparameters (k1, b)
   - Test Rocchio with pseudo-relevance feedback
   - Add neural ranking models for comparison

---

*Generated automatically by IR Evaluation System*
*Date: 2025-12-30 14:44:05*
