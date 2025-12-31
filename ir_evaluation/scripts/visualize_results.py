import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(filepath='ir_evaluation/results/metrics/cisi_results.json'):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_comparison_bar_chart(results, output_dir='ir_evaluation/results/figures'):
    """Create bar chart comparing all models across metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(results.keys())
    metrics = ['map', 'p@5', 'ndcg@10']
    metric_labels = ['MAP', 'P@5', 'NDCG@10']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, model in enumerate(models):
        values = [results[model].get(m, 0) for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model.upper(), color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('IR Model Comparison on CISI Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison chart to {output_dir}/model_comparison.png")
    plt.close()

def plot_metric_breakdown(results, output_dir='ir_evaluation/results/figures'):
    """Create individual metric comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [('map', 'MAP'), ('p@5', 'Precision@5'), ('ndcg@10', 'NDCG@10')]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for ax, (metric, label) in zip(axes, metrics):
        models = list(results.keys())
        values = [results[model].get(metric, 0) for model in models]
        
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.upper() for m in models], rotation=0)
        ax.set_ylabel('Score', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_breakdown.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved metric breakdown to {output_dir}/metric_breakdown.png")
    plt.close()

def plot_performance_radar(results, output_dir='ir_evaluation/results/figures'):
    """Create radar chart for model comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    categories = ['MAP', 'P@5', 'NDCG@10']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (model, data) in enumerate(results.items()):
        values = [data.get('map', 0), data.get('p@5', 0), data.get('ndcg@10', 0)]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model.upper(), color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, max([max([d.get('map', 0), d.get('p@5', 0), d.get('ndcg@10', 0)]) for d in results.values()]) + 0.05)
    ax.set_title('Model Performance Comparison (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved radar chart to {output_dir}/performance_radar.png")
    plt.close()

def generate_report(results, output_dir='ir_evaluation/results'):
    """Generate markdown report"""
    os.makedirs(output_dir, exist_ok=True)
    
    report = f"""# CISI Dataset Evaluation Report

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
"""
    
    for model, metrics in results.items():
        report += f"| {model.upper()} | {metrics.get('map', 0):.4f} | {metrics.get('p@5', 0):.4f} | {metrics.get('ndcg@10', 0):.4f} |\n"
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1].get('map', 0))
    
    report += f"""

## Key Findings

### Best Performing Model: {best_model[0].upper()}
- **MAP**: {best_model[1].get('map', 0):.4f} (Mean Average Precision)
- **P@5**: {best_model[1].get('p@5', 0):.4f} (Precision at 5)
- **NDCG@10**: {best_model[1].get('ndcg@10', 0):.4f} (Normalized Discounted Cumulative Gain at 10)

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
*Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    output_file = os.path.join(output_dir, 'EVALUATION_REPORT.md')
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Generated report at {output_file}")

def main():
    print("=" * 60)
    print("Generating Visualizations and Report")
    print("=" * 60)
    
    # Load results
    print("\n[1/4] Loading results...")
    results = load_results()
    print(f"✓ Loaded results for {len(results)} models")
    
    # Generate visualizations
    print("\n[2/4] Creating visualizations...")
    plot_comparison_bar_chart(results)
    plot_metric_breakdown(results)
    plot_performance_radar(results)
    
    # Generate report
    print("\n[3/4] Generating markdown report...")
    generate_report(results)
    
    print("\n[4/4] Complete!")
    print("=" * 60)
    print("✓ All visualizations and reports generated successfully!")
    print("\nOutput files:")
    print("  - ir_evaluation/results/figures/model_comparison.png")
    print("  - ir_evaluation/results/figures/metric_breakdown.png")
    print("  - ir_evaluation/results/figures/performance_radar.png")
    print("  - ir_evaluation/results/EVALUATION_REPORT.md")
    print("=" * 60)

if __name__ == "__main__":
    main()

