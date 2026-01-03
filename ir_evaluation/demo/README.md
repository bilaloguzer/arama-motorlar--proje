# 🎬 IR System Interactive Demo

An interactive web-based demonstration of your Information Retrieval system, perfect for presentations!

## ✨ Features

### 🔍 Three Search Modes

1. **Custom Query Search**
   - Enter any search query
   - See results ranked by your selected model
   - Real-time performance metrics

2. **Sample Query Evaluation**
   - Test with pre-loaded CISI queries
   - See evaluation metrics (P@5, P@10, NDCG@10, AP)
   - Relevance indicators show which results are actually relevant

3. **Model Comparison**
   - Compare all three models side-by-side
   - See speed differences
   - Compare ranking strategies

### 📊 Key Displays

- **Real-time search** with millisecond timing
- **Interactive results** with expandable document content
- **Evaluation metrics** for sample queries
- **Visual indicators** for relevant documents
- **Model comparison** charts and summaries

## 🚀 Quick Start

### Option 1: Using the Launch Script (Recommended)

```bash
# From project root
chmod +x ir_evaluation/demo/launch_demo.sh
./ir_evaluation/demo/launch_demo.sh
```

### Option 2: Manual Launch

```bash
# Activate virtual environment
source venv/bin/activate

# Install Streamlit if not already installed
pip install streamlit

# Run the app
streamlit run ir_evaluation/demo/app.py
```

The demo will open in your browser at `http://localhost:8501`

## 🎯 Presentation Tips

### For Maximum Impact:

1. **Start with Model Comparison**
   - Show a query like "information retrieval evaluation"
   - Demonstrate how different models rank differently
   - Highlight speed differences

2. **Show Sample Query Evaluation**
   - Pick a query with good metrics
   - Show the relevance indicators
   - Explain the metrics (P@5, NDCG@10, etc.)

3. **Interactive Q&A**
   - Let audience suggest queries
   - Use Custom Query mode for live demonstration
   - Show how the system responds to different query types

### Good Demo Queries:

- `"information retrieval systems"`
- `"document ranking algorithms"`
- `"text processing and indexing"`
- `"search engine evaluation"`
- `"relevance feedback methods"`

## 🎨 UI Features

- **Modern gradient design** - Professional appearance
- **Real-time metrics** - Shows search time, scores, etc.
- **Responsive layout** - Works on different screen sizes
- **Color-coded results** - Green for relevant, red for non-relevant
- **Model information** - Built-in help text

## 📋 Requirements

All requirements are in `ir_evaluation/requirements.txt`. The demo uses:

- **streamlit** - Web interface framework
- All your existing IR system dependencies

## 🐛 Troubleshooting

### "Module not found" errors

Make sure you're in the virtual environment:
```bash
source venv/bin/activate
pip install -r ir_evaluation/requirements.txt
```

### Port already in use

If port 8501 is busy, specify a different port:
```bash
streamlit run ir_evaluation/demo/app.py --server.port 8502
```

### Slow initial load

The first time you run the demo, it needs to:
- Load the CISI dataset
- Build all three model indexes
- This takes ~30 seconds, but is cached afterward

## 🎓 During Your Presentation

### Talking Points:

1. **Architecture**: "This demo runs three classical IR models in real-time"

2. **Dataset**: "We're searching through 1,460 scientific documents from the CISI collection"

3. **Models**:
   - "TF-IDF uses vector space with cosine similarity"
   - "BM25 is the industry standard, used by Elasticsearch"
   - "Rocchio implements relevance feedback"

4. **Performance**: "Notice the millisecond response times - efficient enough for production"

5. **Evaluation**: "The green/red indicators show ground truth relevance from expert judgments"

## 🔧 Customization

Want to modify the demo? Key sections in `app.py`:

- **Line 40-80**: CSS styling
- **Line 82-122**: Model loading and caching
- **Line 270-350**: Custom query search
- **Line 352-450**: Sample query evaluation
- **Line 452-550**: Model comparison

## 📸 Screenshots for Your Report

To capture screenshots during the demo:

1. Use your browser's screenshot tool
2. Or add to your report: "Live demo screenshot from presentation"
3. The visualizations are already beautiful and professional

## 🌟 Pro Tips

- **Practice beforehand**: Run the demo a few times before presenting
- **Have backup queries**: Prepare 3-4 good queries that show interesting results
- **Explain metrics**: Briefly define P@k, NDCG when they appear
- **Show comparisons**: The side-by-side model comparison is very impressive
- **Highlight speed**: Emphasize real-time performance

## 📝 Adding to Your Presentation Slides

Suggested slide structure:

1. **Slide**: "System Architecture"
   - Show the code structure diagram

2. **Slide**: "Live Demo"
   - Switch to the web app
   - Perform 2-3 searches

3. **Slide**: "Performance Results"
   - Show the static visualizations from results/figures/

Good luck with your presentation! 🎉

