# Complete implementation guide for TF-IDF, BM25, and Rocchio IR models

**Three classical retrieval models—TF-IDF with cosine similarity, Okapi BM25, and Rocchio relevance feedback—remain foundational for modern search systems.** This guide provides production-ready Python implementations, mathematical formulas, and a complete evaluation framework using scikit-learn, rank_bm25, and standard IR metrics. These models serve as essential baselines: TF-IDF offers interpretable keyword matching, BM25 improves upon it with term saturation and length normalization, and Rocchio enables query refinement through relevance feedback. Together, they form a robust comparative study foundation for information retrieval research.

## Mathematical foundations and core formulas

### TF-IDF with cosine similarity

The TF-IDF weighting scheme assigns importance to terms based on their frequency within documents and rarity across the corpus. The **scikit-learn default formula** combines term frequency with smoothed inverse document frequency:

```
TF-IDF(t,d) = tf(t,d) × (log((1+N)/(1+df(t))) + 1)
```

Where N = total documents, df(t) = documents containing term t. The "+1" smoothing prevents division by zero and ensures no IDF becomes negative. For sublinear term frequency (recommended for large corpora), apply `1 + log(tf)` instead of raw counts.

**Cosine similarity** measures angular distance between document and query vectors:

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

With L2-normalized vectors (scikit-learn default), cosine similarity equals the dot product, enabling efficient computation via `linear_kernel()`.

### Okapi BM25 probabilistic model

BM25 improves upon TF-IDF through **term frequency saturation** (diminishing returns for repeated terms) and sophisticated **document length normalization**:

```
score(D, Q) = Σ IDF(qi) × [f(qi,D) × (k1 + 1)] / [f(qi,D) + k1 × (1 - b + b × |D|/avgdl)]
```

The key parameters are **k1** (typically 1.2-2.0), controlling how quickly term frequency saturates, and **b** (typically 0.75), controlling length normalization intensity. The IDF component uses the Robertson-Spärck Jones formula: `log(1 + (N - n(qi) + 0.5) / (n(qi) + 0.5))`.

### Rocchio relevance feedback algorithm

Rocchio modifies the query vector based on user feedback:

```
q_modified = α×q_original + β×centroid(relevant) - γ×centroid(non-relevant)
```

Standard parameter values are **α=1.0** (preserves original query intent), **β=0.75** (positive feedback weight), and **γ=0.15** (negative feedback weight). Research shows positive feedback provides more value than negative, hence β > γ. Pseudo-relevance feedback assumes top-k retrieved documents are relevant, enabling automatic query expansion without user interaction.

## TF-IDF implementation with scikit-learn

The complete TF-IDF retriever uses scikit-learn's `TfidfVectorizer` with optimized parameters for information retrieval:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

class TFIDFRetriever:
    def __init__(self, ngram_range=(1, 2), min_df=2, max_df=0.9, 
                 sublinear_tf=True, max_features=50000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,    # Capture phrases with bigrams
            min_df=min_df,               # Remove rare noise terms
            max_df=max_df,               # Remove corpus-wide stop words
            sublinear_tf=sublinear_tf,   # Apply log(1+tf) dampening
            max_features=max_features,   # Limit vocabulary for efficiency
            norm='l2'                    # Enable cosine via dot product
        )
        self.documents = None
        self.tfidf_matrix = None
    
    def fit(self, documents, doc_ids=None):
        self.documents = documents
        self.doc_ids = doc_ids or list(range(len(documents)))
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        return self
    
    def search(self, query, top_k=10):
        query_vec = self.vectorizer.transform([query])
        # linear_kernel is faster than cosine_similarity for L2-normalized vectors
        similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.doc_ids[i], similarities[i]) for i in top_indices]
```

**Critical parameters** for retrieval performance: `sublinear_tf=True` dampens high-frequency term dominance; `ngram_range=(1,2)` captures important phrases; `min_df=2` removes noise from singleton terms. For memory efficiency with large corpora, scikit-learn automatically uses sparse CSR matrices—avoid converting to dense arrays.

## BM25 implementation using rank_bm25 and NumPy

The `rank_bm25` library provides three BM25 variants: **BM25Okapi** (standard), **BM25L** (boosted long documents), and **BM25Plus** (lower-bound fix for length normalization).

```python
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import numpy as np
from collections import Counter
import math

# Using rank_bm25 library
class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75, variant='okapi'):
        self.k1 = k1
        self.b = b
        variants = {'okapi': BM25Okapi, 'l': BM25L, 'plus': BM25Plus}
        self.bm25_class = variants.get(variant, BM25Okapi)
        
    def fit(self, tokenized_corpus, doc_ids=None):
        self.bm25 = self.bm25_class(tokenized_corpus, k1=self.k1, b=self.b)
        self.doc_ids = doc_ids or list(range(len(tokenized_corpus)))
        self.corpus = tokenized_corpus
        return self
    
    def search(self, tokenized_query, top_k=10):
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_indices]
```

For **manual NumPy implementation** enabling full control over the algorithm:

```python
class BM25Manual:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        
    def fit(self, corpus):
        self.corpus_size = len(corpus)
        self.doc_len = np.array([len(doc) for doc in corpus])
        self.avgdl = np.mean(self.doc_len) if self.corpus_size > 0 else 1
        
        # Build term frequencies and document frequencies
        self.tf = [Counter(doc) for doc in corpus]
        self.df = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        
        # Precompute IDF with smoothing to prevent negatives
        self.idf = {
            term: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }
        return self
    
    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        for term in query:
            if term not in self.df:
                continue
            for idx in range(self.corpus_size):
                freq = self.tf[idx].get(term, 0)
                if freq == 0:
                    continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] += self.idf[term] * (numerator / denominator)
        return scores
```

**Parameter tuning guidance**: For standard text, use k1=1.2, b=0.75. For long documents (books, patents), increase k1 to 1.5-2.0. For short text (tweets, titles), decrease both k1 (~1.0) and b (~0.3-0.5). Use BM25L or BM25Plus when very long documents are unfairly penalized.

## Rocchio relevance feedback implementation

Rocchio builds upon TF-IDF by modifying query vectors based on feedback:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RocchioRetriever:
    def __init__(self, alpha=1.0, beta=0.75, gamma=0.15):
        self.alpha = alpha  # Original query weight
        self.beta = beta    # Relevant documents weight
        self.gamma = gamma  # Non-relevant documents weight
        self.vectorizer = TfidfVectorizer(norm='l2', stop_words='english')
        
    def fit(self, documents, doc_ids=None):
        self.documents = documents
        self.doc_ids = doc_ids or list(range(len(documents)))
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        return self
    
    def initial_search(self, query, top_k=10):
        self.query_vector = self.vectorizer.transform([query]).toarray().flatten()
        similarities = cosine_similarity([self.query_vector], self.doc_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.doc_ids[i], similarities[i]) for i in top_indices]
    
    def apply_feedback(self, relevant_ids, non_relevant_ids=None):
        """Apply Rocchio formula: q_new = α*q + β*centroid(rel) - γ*centroid(nonrel)"""
        relevant_indices = [self.doc_ids.index(did) for did in relevant_ids]
        rel_centroid = np.asarray(self.doc_vectors[relevant_indices].mean(axis=0)).flatten()
        
        nonrel_centroid = np.zeros(self.doc_vectors.shape[1])
        if non_relevant_ids:
            nonrel_indices = [self.doc_ids.index(did) for did in non_relevant_ids]
            nonrel_centroid = np.asarray(self.doc_vectors[nonrel_indices].mean(axis=0)).flatten()
        
        # Apply Rocchio formula
        self.modified_query = (
            self.alpha * self.query_vector +
            self.beta * rel_centroid -
            self.gamma * nonrel_centroid
        )
        # Zero out negative weights (standard practice)
        self.modified_query = np.maximum(self.modified_query, 0)
        return self
    
    def rerank(self, top_k=10):
        similarities = cosine_similarity([self.modified_query], self.doc_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.doc_ids[i], similarities[i]) for i in top_indices]
    
    def pseudo_relevance_feedback(self, query, num_feedback=10, top_k=100):
        """Blind feedback: assume top-k initial results are relevant"""
        initial_results = self.initial_search(query, top_k=num_feedback)
        pseudo_relevant = [doc_id for doc_id, _ in initial_results]
        self.apply_feedback(pseudo_relevant, non_relevant_ids=[])
        return self.rerank(top_k=top_k)
```

**Query drift mitigation**: Keep α≥1.0 to maintain original intent; limit feedback to 10-20 documents; consider result fusion combining original and expanded query rankings. Research shows little benefit beyond single-iteration feedback.

## Text preprocessing pipeline

A robust preprocessing pipeline is essential for all three models:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import pos_tag
from nltk.corpus import wordnet

class TextPreprocessor:
    def __init__(self, use_stemming=False, use_lemmatization=True, 
                 remove_stopwords=True, min_token_length=2):
        nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'], quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.min_length = min_token_length
    
    def _get_wordnet_pos(self, tag):
        if tag.startswith('J'): return wordnet.ADJ
        elif tag.startswith('V'): return wordnet.VERB
        elif tag.startswith('R'): return wordnet.ADV
        return wordnet.NOUN
    
    def preprocess(self, text):
        # Clean text
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<[^>]*>', '', text)           # Remove HTML
        text = re.sub(r'[^\w\s]', '', text)           # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()      # Normalize whitespace
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Normalize (lemmatization preferred for quality; stemming for speed)
        if self.use_lemmatization:
            pos_tags = pos_tag(tokens)
            tokens = [self.lemmatizer.lemmatize(w, self._get_wordnet_pos(p)) for w, p in pos_tags]
        elif self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return [t for t in tokens if len(t) >= self.min_length]
```

**Stemming vs. lemmatization trade-offs**: Snowball stemming is faster but produces non-words ("studies" → "studi"); lemmatization is slower but maintains valid words ("studies" → "study"). Use stemming for high-volume systems; lemmatization for quality-focused applications. Always apply identical preprocessing to both documents and queries.

## Evaluation metrics implementation

All standard IR metrics with complete Python implementations:

```python
import numpy as np

def precision_at_k(y_true, y_scores, k):
    """Proportion of relevant items in top-k results"""
    order = np.argsort(y_scores)[::-1][:k]
    return np.sum(y_true[order]) / k

def recall_at_k(y_true, y_scores, k):
    """Proportion of total relevant items found in top-k"""
    total_relevant = np.sum(y_true)
    if total_relevant == 0:
        return 0.0
    order = np.argsort(y_scores)[::-1][:k]
    return np.sum(y_true[order]) / total_relevant

def average_precision(y_true, y_scores):
    """Average of precision values at each relevant document position"""
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    
    num_relevant = np.sum(y_true)
    if num_relevant == 0:
        return 0.0
    
    precisions = []
    relevant_count = 0
    for i, is_rel in enumerate(y_true_sorted):
        if is_rel:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    return np.mean(precisions) if precisions else 0.0

def mean_average_precision(y_true_list, y_scores_list):
    """MAP: Average of AP across all queries"""
    return np.mean([average_precision(yt, ys) for yt, ys in zip(y_true_list, y_scores_list)])

def ndcg_at_k(y_true, y_scores, k, method='exponential'):
    """Normalized Discounted Cumulative Gain"""
    order = np.argsort(y_scores)[::-1][:k]
    y_true_sorted = y_true[order]
    
    discounts = np.log2(np.arange(2, len(y_true_sorted) + 2))
    if method == 'exponential':
        gains = (2 ** y_true_sorted - 1) / discounts
    else:
        gains = y_true_sorted / discounts
    dcg = np.sum(gains)
    
    # Ideal DCG
    ideal_order = np.sort(y_true)[::-1][:k]
    ideal_discounts = np.log2(np.arange(2, len(ideal_order) + 2))
    if method == 'exponential':
        ideal_gains = (2 ** ideal_order - 1) / ideal_discounts
    else:
        ideal_gains = ideal_order / ideal_discounts
    idcg = np.sum(ideal_gains)
    
    return dcg / idcg if idcg > 0 else 0.0
```

For production evaluation, use the **ir-measures** library (`pip install ir-measures`) which provides standardized metric implementations compatible with TREC evaluation standards.

## Recommended datasets for evaluation

**MS MARCO** is the primary benchmark for IR research, featuring 8.8 million passages and ~500K training queries from Bing search logs. Access via the `ir_datasets` library:

```python
import ir_datasets
dataset = ir_datasets.load("msmarco-passage/train")
for doc in dataset.docs_iter():
    print(doc.doc_id, doc.text[:100])
```

**CISI dataset** (available on Kaggle) provides a smaller benchmark with 1,460 documents and 112 queries with complete relevance judgments—ideal for initial development. The **BEIR benchmark** offers 18 diverse datasets for zero-shot evaluation across domains.

## Project structure for comparative evaluation

```
ir_evaluation/
├── configs/
│   ├── tfidf_config.yaml
│   ├── bm25_config.yaml
│   └── rocchio_config.yaml
├── data/
│   ├── raw/                  # Original corpus files
│   ├── processed/            # Tokenized documents
│   └── qrels/               # Relevance judgments (TREC format)
├── src/
│   ├── models/
│   │   ├── base.py          # Abstract RetrievalModel class
│   │   ├── tfidf_model.py
│   │   ├── bm25_model.py
│   │   └── rocchio_model.py
│   ├── preprocessing/
│   │   └── preprocessor.py
│   └── evaluation/
│       ├── metrics.py
│       └── evaluator.py
├── notebooks/
│   └── model_comparison.ipynb
├── results/
│   ├── runs/                # TREC-format output files
│   └── metrics/             # JSON evaluation results
└── scripts/
    └── run_experiment.py
```

An **abstract base class** ensures consistent interfaces across all models:

```python
from abc import ABC, abstractmethod
from typing import List, Tuple

class RetrievalModel(ABC):
    @abstractmethod
    def fit(self, corpus: List[List[str]]) -> 'RetrievalModel': pass
    
    @abstractmethod
    def score(self, query: List[str]) -> np.ndarray: pass
    
    def retrieve(self, query: List[str], top_k: int = 100) -> List[Tuple[str, float]]:
        scores = self.score(query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_idx]
```

## Essential Python libraries for IR systems

- **rank_bm25**: Simple BM25 implementation (`pip install rank-bm25`)
- **bm25s**: High-performance BM25, up to 500x faster than rank_bm25
- **Pyserini**: Academic IR toolkit with prebuilt MSMARCO indexes
- **PyTerrier**: Declarative IR experimentation framework with pipeline operators
- **ir-measures**: Standardized evaluation metrics (nDCG, MAP, MRR)
- **ir_datasets**: Unified access to standard IR benchmarks
- **Gensim**: TF-IDF, LSI, and similarity queries for topic modeling
- **Whoosh**: Pure Python search engine with BM25F scoring

## Common pitfalls and optimization strategies

**TF-IDF pitfalls**: Out-of-vocabulary query terms are silently ignored—use n-grams or character n-grams for fuzzy matching. Avoid converting sparse matrices to dense arrays, which causes memory explosions on large corpora.

**BM25 pitfalls**: Very long documents can be over-penalized with standard parameters—use BM25L or reduce b parameter. Negative IDF values occur for terms in >50% of documents; apply floor of 0 or use the log(1+...) variant.

**Rocchio pitfalls**: Query drift occurs when expanded queries shift away from user intent—keep α≥1.0 and limit iterations. Zero out negative weights after applying the formula (standard practice).

## Conclusion

These three models form a comprehensive baseline for IR evaluation. **TF-IDF** provides fast, interpretable keyword matching suitable for small-to-medium datasets. **BM25** offers state-of-the-art lexical retrieval through term saturation and length normalization—it remains the default in Elasticsearch and Lucene. **Rocchio** enables query refinement through feedback, with pseudo-relevance feedback automating the process. For implementation, start with scikit-learn's TfidfVectorizer and rank_bm25 for rapid prototyping, then consider Pyserini or bm25s for production-scale systems. Evaluate using standardized metrics (nDCG@10, MAP) on MSMARCO or CISI datasets, and structure your codebase with abstract base classes to enable clean model comparison.