# Information Retrieval System Evaluation

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive comparative evaluation of classical Information Retrieval algorithms (TF-IDF, BM25, Rocchio) on large-scale datasets, demonstrating scalability characteristics and performance optimization through preprocessing techniques.

## 🎯 Key Results

- **BM25 achieves 75.1% MAP** on 50,000 documents (MS MARCO dataset)
- **TF-IDF degradation discovered**: -23% performance drop at 100K documents
- **Stemming impact quantified**: +9.8% MAP improvement for BM25
- **Efficient execution**: ~3,000 documents/second on M1 Pro MacBook

## 📊 Datasets

- **CISI**: 1,460 documents (validation baseline)
- **MS MARCO**: 10K, 50K, 100K passages (production-scale evaluation)
- **Total**: 161,460 documents tested

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/[YOUR_USERNAME]/ir-evaluation-project.git
cd ir-evaluation-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r ir_evaluation/requirements.txt
```

### Run Experiments

```bash
# Test on CISI dataset (30 seconds)
./venv/bin/python3 ir_evaluation/scripts/test_cisi_simple.py

# Test on MS MARCO 10K (5 minutes)
./venv/bin/python3 ir_evaluation/scripts/test_msmarco.py
# Choose option 1 when prompted

# Generate visualizations
./venv/bin/python3 ir_evaluation/scripts/create_enhanced_visualizations.py
```

## 📁 Project Structure

```
ir_evaluation/
├── src/
│   ├── models/              # TF-IDF, BM25, Rocchio implementations
│   ├── preprocessing/       # Text processing pipeline
│   ├── evaluation/          # Metrics (MAP, NDCG, P@k)
│   └── data/               # Dataset loaders
├── scripts/
│   ├── test_cisi_simple.py        # CISI evaluation
│   ├── test_msmarco.py            # MS MARCO evaluation
│   └── create_enhanced_visualizations.py  # Generate charts
├── results/
│   ├── metrics/            # JSON result files
│   ├── figures/            # 16 visualizations (2 animated GIFs!)
│   ├── COMPREHENSIVE_TECHNICAL_REPORT.md  # Full documentation
│   └── THE_COMPLETE_STORY.md             # Narrative report
└── requirements.txt
```

## 🎓 Key Findings

### 1. BM25 Demonstrates Superior Performance

- **Stable across scales**: 74-75% MAP from 50K to 100K documents
- **Outperforms alternatives**: +48% vs TF-IDF, +12% vs Rocchio
- **Industry validation**: Used in Elasticsearch, Solr, Lucene

### 2. TF-IDF Exhibits IDF Degradation

- **Critical discovery**: 23% performance drop when scaling
- **Root cause**: IDF compression in large corpora
- **Practical impact**: Unusable for >50K document collections

### 3. Stemming Provides Measurable Improvements

- **BM25**: +9.8% MAP, +16.9% NDCG@10
- **TF-IDF**: +7.5% MAP
- **Rocchio**: +7.6% MAP

### 4. Efficient on Consumer Hardware

- **Processing time**: 33.6 seconds for 100K documents
- **Throughput**: ~3,000 docs/second
- **Memory**: <4GB RAM
- **No GPU required**: CPU-optimized algorithms

## 📊 Performance Summary

| Dataset | Model | MAP | P@5 | NDCG@10 |
|---------|-------|-----|-----|---------|
| **CISI (1.4K)** | TF-IDF | 19.6% | 33.2% | 31.4% |
| | **BM25** | **20.5%** | **39.0%** | **36.7%** |
| | Rocchio | 14.7% | 27.6% | 27.2% |
| **MS MARCO (50K)** | TF-IDF | 50.8% | 13.6% | 55.6% |
| | **BM25** | **75.1%** 🏆 | **18.4%** | **79.1%** |
| | Rocchio | 66.7% | 17.0% | 71.6% |

## 🎬 Visualizations

The project includes 16 professional visualizations:

- **Animated GIFs**: Scalability evolution, model comparison
- **Heatmaps**: Performance across all dimensions
- **3D Plots**: Multi-metric performance space
- **Charts**: Precision-recall, stemming impact, speed analysis

All visualizations available in `ir_evaluation/results/figures/`

## 📖 Documentation

- **[COMPREHENSIVE_TECHNICAL_REPORT.md](ir_evaluation/results/COMPREHENSIVE_TECHNICAL_REPORT.md)**: Complete technical documentation (8,785 words)
- **[THE_COMPLETE_STORY.md](ir_evaluation/results/THE_COMPLETE_STORY.md)**: Narrative journey through the project
- **[FINAL_REPORT.md](ir_evaluation/results/FINAL_REPORT.md)**: Executive summary
- **[STEMMING_IMPACT_ANALYSIS.md](ir_evaluation/results/STEMMING_IMPACT_ANALYSIS.md)**: Preprocessing analysis

## 🛠️ Technologies

- **Python 3.13**
- **scikit-learn**: TF-IDF implementation
- **rank_bm25**: BM25 algorithm
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization
- **ir_datasets**: Dataset management
- **NLTK**: Porter Stemmer

## 🔬 Algorithms Implemented

### TF-IDF (1975)
- Vector space model with cosine similarity
- Sublinear term frequency weighting
- L2 normalization

### BM25 (1994)
- Probabilistic ranking function
- Term frequency saturation (k1=1.5)
- Length normalization (b=0.75)

### Rocchio (1971)
- Relevance feedback algorithm
- Query vector modification
- Pseudo-relevance feedback support

## 📈 Evaluation Metrics

- **MAP** (Mean Average Precision): Primary ranking quality metric
- **P@k** (Precision at k): User-facing relevance
- **NDCG@k** (Normalized DCG): Position-aware quality
- **R@k** (Recall at k): Coverage metric

## 🎯 Use Cases

- **Research**: Baseline for IR experiments
- **Education**: Learning classical algorithms
- **Industry**: Production search system prototyping
- **Benchmarking**: Algorithm comparison framework

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Neural ranking models (BERT, T5)
- Additional datasets (BEIR benchmark)
- Hyperparameter optimization
- Multilingual support

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{ir-evaluation-2025,
  author = {[Your Name]},
  title = {Comparative Evaluation of Classical IR Algorithms},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/[YOUR_USERNAME]/ir-evaluation-project}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: CISI (University of Glasgow), MS MARCO (Microsoft Research)
- **Libraries**: scikit-learn, rank_bm25, NLTK, ir_datasets
- **Inspiration**: Classical IR literature and modern search engines

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **Institution**: [Your University]

---

**⭐ Star this repo if you find it useful!**

Built with ❤️ for the Information Retrieval community

