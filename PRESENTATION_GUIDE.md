# 🎓 Presentation Day Quick Start Guide

## Before Your Presentation

### 1. Test the Demo (5 minutes before)

```bash
cd /path/to/your/project
./ir_evaluation/demo/launch_demo.sh
```

The demo will open at `http://localhost:8501`

### 2. Have Backup Screenshots

If internet/laptop issues occur, you have:
- Static visualizations in `ir_evaluation/results/figures/`
- Comprehensive reports in `ir_evaluation/results/`

## Demo Flow (Recommended 5-7 minutes)

### Part 1: Introduction (30 seconds)
"This is an interactive search system comparing three classical IR algorithms on 1,460 scientific documents."

### Part 2: Model Comparison (2 minutes)

1. Click sidebar → **⚖️ Model Comparison**
2. Enter query: `"information retrieval evaluation"`
3. Click **🚀 Compare Models**

**Talking Points:**
- "Notice BM25 finds documents with scores around 5-6"
- "TF-IDF uses 0-1 normalized scores"
- "All three models rank documents differently"
- "BM25 is fastest - used by Elasticsearch in production"

### Part 3: Sample Query Evaluation (2 minutes)

1. Click sidebar → **📋 Sample Queries**
2. Select any query
3. Click **🚀 Evaluate Query**

**Talking Points:**
- "Green checkmarks show ground-truth relevant documents"
- "P@5 of 60% means 3 out of 5 top results are relevant"
- "NDCG accounts for position - higher is better"
- "These metrics match our comprehensive evaluation"

### Part 4: Custom Search (1-2 minutes)

1. Click sidebar → **🔍 Custom Query**
2. Select **BM25** model
3. Enter: `"document ranking algorithms"`
4. Click **🚀 Search**

**Talking Points:**
- "Search completes in milliseconds"
- "Shows top 10 most relevant documents"
- "System works for any text query"

### Part 5: Dataset Info (30 seconds)

Point to sidebar metrics:
- "1,460 documents from CISI collection"
- "112 test queries"
- "We also evaluated on MS MARCO with 100,000 documents"

## Good Demo Queries

**For Strong Results:**
- `"information retrieval systems"`
- `"document classification methods"`
- `"search engine evaluation"`

**For Model Differences:**
- `"relevance feedback"`
- `"text processing algorithms"`

## If Something Goes Wrong

### Demo Won't Start
```bash
# Reinstall Streamlit
./venv/bin/pip install --upgrade streamlit
```

### NLTK Errors
- The demo now works without preprocessing
- Models handle raw text directly

### Port Already in Use
```bash
streamlit run ir_evaluation/demo/app.py --server.port 8502
```

### Complete Failure
- Switch to static visualizations in `results/figures/`
- Show the comprehensive PDF reports
- The evaluation results are still impressive!

## Presentation Structure

### Slide 1: Title
"Information Retrieval System Evaluation: TF-IDF, BM25, and Rocchio"

### Slide 2: Project Overview
- Implemented 3 classical algorithms
- Evaluated on 2 datasets (1.4K and 100K documents)
- Generated comprehensive analysis

### Slide 3: **LIVE DEMO** ⬅️ Switch to browser

### Slide 4: Key Results
- BM25 achieved 75.1% MAP on 50K documents
- TF-IDF shows degradation at scale
- Stemming improves performance by ~10%

### Slide 5: Visualizations
- Show scalability analysis
- Show model comparison charts
- Show performance heatmaps

### Slide 6: Conclusions & Questions

## Backup Plan

If live demo fails:
1. Show `scalability_animated.gif` - very impressive!
2. Show `model_comparison_animated.gif`
3. Walk through `COMPREHENSIVE_TECHNICAL_REPORT.md`
4. Emphasize the 161K documents tested

## Time Allocation

- Introduction: 1 min
- **Live Demo: 5-7 min** ⭐
- Results Discussion: 3-4 min
- Q&A: Remaining time

## Pro Tips

1. **Practice once** before presenting
2. **Zoom the browser** to 125-150% for visibility
3. **Close other apps** to ensure smooth performance
4. **Have the launch script ready** in terminal
5. **Mention the tech stack**: Python, scikit-learn, rank_bm25, Streamlit

## Commands Cheat Sheet

```bash
# Start demo
./ir_evaluation/demo/launch_demo.sh

# Test demo before presenting
./venv/bin/python3 ir_evaluation/demo/test_demo.py

# If Streamlit needs update
./venv/bin/pip install --upgrade streamlit

# Check what's running
lsof -i :8501
```

## Questions You Might Get

**Q: Why did you choose these algorithms?**
A: They're foundational IR models - TF-IDF (1975), Rocchio (1971), BM25 (1994). Understanding these is essential before neural approaches.

**Q: How does this scale?**
A: We tested up to 100K documents, processing at ~3,000 docs/second on M1 Pro. BM25 scales linearly.

**Q: Could you add neural models like BERT?**
A: Absolutely! This provides a baseline for comparison. Neural models would be the next step.

**Q: What about multilingual support?**
A: These algorithms are language-independent with proper preprocessing. We focused on English for evaluation.

**Q: How accurate is it?**
A: BM25 achieved 75% MAP - competitive with many production systems for this dataset size.

## Good Luck! 🍀

Remember: Your comprehensive evaluation (161K documents, 16 visualizations, 8,785-word technical report) speaks for itself. The demo is just the cherry on top!

---

**Need Help?** Check `ir_evaluation/demo/README.md` for detailed documentation.

