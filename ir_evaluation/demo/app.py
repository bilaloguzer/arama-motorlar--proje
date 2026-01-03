"""
Interactive Demo Application for IR System Evaluation
Showcases TF-IDF, BM25, and Rocchio models with real-time search
"""

import streamlit as st
import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.tfidf_model import TFIDFRetriever
from src.models.bm25_model import BM25Retriever
from src.models.rocchio_model import RocchioRetriever
from src.data.msmarco_loader import load_msmarco_subset
from src.evaluation.metrics import ndcg_at_k
import numpy as np

# Page configuration
st.set_page_config(
    page_title="IR System Demo - TF-IDF, BM25, Rocchio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .result-rank {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
        margin-right: 1rem;
    }
    .result-score {
        color: #10b981;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache all models and data"""
    with st.spinner("🔄 Loading MS MARCO dataset and initializing models... (This may take 1-2 minutes)"):
        # Load MS MARCO dataset (10K documents for demo speed)
        docs, queries, qrels = load_msmarco_subset(num_docs=10000, num_queries=100)
        
        # Get document IDs and texts
        doc_ids = list(docs.keys())
        doc_texts = [docs[doc_id] for doc_id in doc_ids]
        
        st.info(f"Loaded {len(docs)} documents and {len(queries)} queries from MS MARCO")
        
        # Initialize models (no preprocessing - models handle text directly)
        tfidf_model = TFIDFRetriever()
        bm25_model = BM25Retriever(k1=1.5, b=0.75)
        rocchio_model = RocchioRetriever(alpha=1.0, beta=0.75, gamma=0.15)
        
        # Build indexes
        tfidf_model.fit(doc_texts, doc_ids)
        bm25_model.fit(doc_texts, doc_ids)
        rocchio_model.fit(doc_texts, doc_ids)
        
        return {
            'models': {
                'TF-IDF': tfidf_model,
                'BM25': bm25_model,
                'Rocchio': rocchio_model
            },
            'docs': docs,
            'queries': queries,
            'qrels': qrels,
            'doc_ids': doc_ids
        }

def calculate_query_metrics(retrieved_doc_ids, qrels, k=10):
    """Calculate metrics for a single query using simple precision/recall"""
    if not retrieved_doc_ids or not qrels:
        return {'P@5': 0.0, 'P@10': 0.0, 'NDCG@10': 0.0, 'Relevant Found': 0}
    
    relevant_docs = set(qrels.keys()) if isinstance(qrels, dict) else set(qrels)
    
    # Calculate precision@5 and precision@10
    top_5 = retrieved_doc_ids[:5]
    top_10 = retrieved_doc_ids[:10]
    
    p_at_5 = sum(1 for doc_id in top_5 if doc_id in relevant_docs) / 5.0
    p_at_10 = sum(1 for doc_id in top_10 if doc_id in relevant_docs) / 10.0
    
    # Calculate NDCG@10
    ndcg_score = ndcg_at_k(retrieved_doc_ids[:10], qrels)
    
    # Count relevant found
    relevant_found = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_docs)
    
    return {
        'P@5': p_at_5,
        'P@10': p_at_10,
        'NDCG@10': ndcg_score,
        'Relevant Found': relevant_found
    }

def search_with_model(model, query, doc_ids, top_k=10):
    """Perform search and return results with timing"""
    start_time = time.time()
    scores = model.score(query)
    search_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Get top k results
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, doc_idx in enumerate(top_indices, 1):
        if doc_idx < len(doc_ids) and scores[doc_idx] > 0:
            results.append({
                'doc_id': doc_ids[doc_idx],
                'score': scores[doc_idx],
                'rank': rank
            })
    
    return results, search_time

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Information Retrieval System Demo</p>', unsafe_allow_html=True)
    st.markdown("### Interactive Search with TF-IDF, BM25, and Rocchio Models")
    st.markdown("---")
    
    # Load models and data
    try:
        data = load_models()
        models = data['models']
        docs = data['docs']
        queries = data['queries']
        qrels = data['qrels']
        doc_ids = data['doc_ids']
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("💡 Make sure you have run the CISI dataset loading script first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/search.png", width=80)
        st.title("⚙️ Configuration")
        
        # Search mode
        search_mode = st.radio(
            "Search Mode:",
            ["🔍 Custom Query", "📋 Sample Queries", "⚖️ Model Comparison"],
            index=0
        )
        
        st.markdown("---")
        
        # Model selection
        if search_mode != "⚖️ Model Comparison":
            selected_model_name = st.selectbox(
                "Select Model:",
                list(models.keys()),
                index=1  # Default to BM25
            )
        
        # Number of results
        top_k = st.slider("Results to Show:", 5, 20, 10)
        
        st.markdown("---")
        
        # Dataset info
        st.markdown("### 📊 Dataset Info")
        st.metric("Documents", f"{len(docs):,}")
        st.metric("Queries", f"{len(queries):,}")
        st.metric("Dataset", "MS MARCO")
        
        st.markdown("---")
        
        # Model info
        with st.expander("ℹ️ Model Information"):
            st.markdown("""
            **TF-IDF** (1975)
            - Vector space model
            - Cosine similarity
            - Fast retrieval
            
            **BM25** (1994)
            - Probabilistic ranking
            - Length normalization
            - Industry standard
            
            **Rocchio** (1971)
            - Relevance feedback
            - Query expansion
            - Iterative refinement
            """)
    
    # Main content area
    if search_mode == "🔍 Custom Query":
        st.subheader("🔍 Custom Query Search")
        
        # Initialize session state for query if not exists
        if 'query_text' not in st.session_state:
            st.session_state.query_text = ""
        
        query_text = st.text_input(
            "Enter your search query:",
            value=st.session_state.query_text,
            placeholder="e.g., machine learning algorithms",
            help="Type any search query to find relevant documents"
        )
        
        # Update session state
        st.session_state.query_text = query_text
        
        if st.button("🚀 Search", key="custom_search_btn") and query_text:
            with st.spinner(f"Searching with {selected_model_name}..."):
                model = models[selected_model_name]
                results, search_time = search_with_model(model, query_text, doc_ids, top_k)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-box"><h3>⏱️ Search Time</h3><h2>{search_time:.2f} ms</h2></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-box"><h3>📄 Results Found</h3><h2>{len(results)}</h2></div>', unsafe_allow_html=True)
                with col3:
                    avg_score = np.mean([r['score'] for r in results]) if results else 0
                    st.markdown(f'<div class="metric-box"><h3>📊 Avg Score</h3><h2>{avg_score:.3f}</h2></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display results
                st.subheader(f"📋 Top {len(results)} Results")
                
                for result in results:
                    doc_id = result['doc_id']
                    doc_text = docs.get(doc_id, "Document not found")
                    
                    # Truncate long documents
                    display_text = doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="result-rank">#{result['rank']}</span>
                        <span style="font-size: 1.2rem; font-weight: bold;">Document {doc_id}</span>
                        <span class="result-score" style="float: right;">Score: {result['score']:.4f}</span>
                        <p style="margin-top: 1rem; color: #475569; line-height: 1.6;">{display_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        elif not query_text:
            st.info("👆 Enter a query above and click 'Search' to see results")
    
    elif search_mode == "📋 Sample Queries":
        st.subheader("📋 Sample Query Evaluation")
        
        # Initialize session state
        if 'selected_query_id' not in st.session_state:
            st.session_state.selected_query_id = list(queries.keys())[0]
        
        # Select a sample query
        sample_query_ids = list(queries.keys())[:20]  # First 20 queries
        selected_query_id = st.selectbox(
            "Select a sample query:",
            sample_query_ids,
            index=sample_query_ids.index(st.session_state.selected_query_id) if st.session_state.selected_query_id in sample_query_ids else 0,
            format_func=lambda x: f"Query {x}: {queries[x][:80]}..."
        )
        
        # Update session state
        st.session_state.selected_query_id = selected_query_id
        
        query_text = queries[selected_query_id]
        st.markdown(f"**Query:** {query_text}")
        
        if st.button("🚀 Evaluate Query", key="sample_query_btn"):
            with st.spinner(f"Evaluating with {selected_model_name}..."):
                model = models[selected_model_name]
                results, search_time = search_with_model(model, query_text, doc_ids, top_k)
                
                # Get relevant docs for this query
                relevant_docs = qrels.get(selected_query_id, {})
                retrieved_doc_ids = [r['doc_id'] for r in results]
                
                # Calculate metrics
                metrics = calculate_query_metrics(retrieved_doc_ids, relevant_docs, top_k)
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown(f'<div class="metric-box"><h4>⏱️ Time</h4><h3>{search_time:.1f}ms</h3></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-box"><h4>P@5</h4><h3>{metrics["P@5"]:.2%}</h3></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-box"><h4>P@10</h4><h3>{metrics["P@10"]:.2%}</h3></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="metric-box"><h4>NDCG@10</h4><h3>{metrics["NDCG@10"]:.2%}</h3></div>', unsafe_allow_html=True)
                with col5:
                    st.markdown(f'<div class="metric-box"><h4>Relevant</h4><h3>{metrics["Relevant Found"]}</h3></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display results with relevance indicators
                st.subheader(f"📋 Top {len(results)} Results")
                
                relevant_count = 0
                for result in results:
                    doc_id = result['doc_id']
                    is_relevant = doc_id in relevant_docs
                    if is_relevant:
                        relevant_count += 1
                    
                    relevance_indicator = "✅ RELEVANT" if is_relevant else "❌ Not Relevant"
                    relevance_color = "#10b981" if is_relevant else "#ef4444"
                    
                    doc_text = docs.get(doc_id, "Document not found")
                    display_text = doc_text[:400] + "..." if len(doc_text) > 400 else doc_text
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="result-rank">#{result['rank']}</span>
                        <span style="font-size: 1.2rem; font-weight: bold;">Document {doc_id}</span>
                        <span style="color: {relevance_color}; font-weight: bold; float: right;">{relevance_indicator}</span>
                        <br>
                        <span class="result-score">Score: {result['score']:.4f}</span>
                        <p style="margin-top: 1rem; color: #475569; line-height: 1.6;">{display_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info(f"💡 Found {relevant_count} relevant documents in top {len(results)} results")
    
    else:  # Model Comparison
        st.subheader("⚖️ Model Comparison")
        
        # Initialize session state for comparison query
        if 'comparison_query' not in st.session_state:
            st.session_state.comparison_query = ""
        
        query_text = st.text_input(
            "Enter query to compare models:",
            value=st.session_state.comparison_query,
            placeholder="e.g., machine learning classification",
            help="Compare how different models rank documents for the same query"
        )
        
        # Update session state
        st.session_state.comparison_query = query_text
        
        if st.button("🚀 Compare Models", key="compare_models_btn") and query_text:
            st.markdown("---")
            
            # Create columns for each model
            cols = st.columns(3)
            
            all_results = {}
            all_times = {}
            
            for idx, (model_name, model) in enumerate(models.items()):
                with cols[idx]:
                    st.markdown(f"### {model_name}")
                    
                    with st.spinner(f"Searching..."):
                        results, search_time = search_with_model(model, query_text, doc_ids, min(top_k, 5))
                        all_results[model_name] = results
                        all_times[model_name] = search_time
                    
                    # Display time
                    st.metric("⏱️ Search Time", f"{search_time:.2f} ms")
                    
                    # Display top results
                    st.markdown("**Top 5 Results:**")
                    for result in results[:5]:
                        doc_id = result['doc_id']
                        doc_text = docs.get(doc_id, "")[:150] + "..."
                        
                        with st.expander(f"#{result['rank']} Doc {doc_id} (Score: {result['score']:.3f})"):
                            st.write(doc_text)
            
            # Summary comparison
            st.markdown("---")
            st.subheader("📊 Comparison Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**⏱️ Speed Ranking:**")
                sorted_times = sorted(all_times.items(), key=lambda x: x[1])
                for rank, (name, time_ms) in enumerate(sorted_times, 1):
                    st.write(f"{rank}. **{name}**: {time_ms:.2f} ms")
            
            with col2:
                st.markdown("**🏆 Highest Scores:**")
                for model_name, results in all_results.items():
                    if results:
                        max_score = max(r['score'] for r in results)
                        st.write(f"**{model_name}**: {max_score:.4f}")
            
            with col3:
                st.markdown("**🎯 Score Ranges:**")
                for model_name, results in all_results.items():
                    if results:
                        scores = [r['score'] for r in results]
                        st.write(f"**{model_name}**: {min(scores):.3f} - {max(scores):.3f}")
        
        elif not query_text:
            st.info("👆 Enter a query above to compare all three models side-by-side")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 0;">
        <p><strong>Information Retrieval System Evaluation Project</strong></p>
        <p>Comparing Classical IR Algorithms: TF-IDF, BM25, and Rocchio</p>
        <p>📊 Dataset: MS MARCO (10,000 documents) | 🎓 Educational Demo</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

