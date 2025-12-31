import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import os
from datetime import datetime

def load_all_results():
    """Load all result files"""
    results = {}
    cisi_path = 'ir_evaluation/results/metrics/cisi_results.json'
    if os.path.exists(cisi_path):
        with open(cisi_path, 'r') as f:
            results['cisi'] = json.load(f)
    
    for size in [10000, 50000, 100000]:
        path = f'ir_evaluation/results/metrics/msmarco_{size}_results.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[f'msmarco_{size}'] = json.load(f)
    
    return results

def create_animated_scalability(results, output_dir='ir_evaluation/results/figures'):
    """Create animated scalability plot (saved as GIF)"""
    os.makedirs(output_dir, exist_ok=True)
    
    sizes = [1460, 10000, 50000, 100000]
    labels = ['CISI\n1.4K', 'MARCO\n10K', 'MARCO\n50K', 'MARCO\n100K']
    
    bm25_scores = [results['cisi']['bm25']['map']]
    tfidf_scores = [results['cisi']['tfidf']['map']]
    rocchio_scores = [results['cisi']['rocchio']['map']]
    
    for size in [10000, 50000, 100000]:
        key = f'msmarco_{size}'
        if key in results:
            bm25_scores.append(results[key]['bm25']['map'])
            tfidf_scores.append(results[key]['tfidf']['map'])
            rocchio_scores.append(results[key]['rocchio']['map'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    def animate(frame):
        ax.clear()
        current_idx = frame + 1
        
        x = np.arange(current_idx)
        
        # Plot lines up to current frame
        if current_idx > 0:
            ax.plot(x, bm25_scores[:current_idx], 'o-', linewidth=4, markersize=15, 
                   label='BM25', color='#e74c3c', alpha=0.9)
        if current_idx > 0:
            ax.plot(x, rocchio_scores[:current_idx], 's-', linewidth=4, markersize=15,
                   label='Rocchio', color='#2ecc71', alpha=0.9)
        if current_idx > 0:
            ax.plot(x, tfidf_scores[:current_idx], '^-', linewidth=4, markersize=15,
                   label='TF-IDF', color='#3498db', alpha=0.9)
        
        # Add value labels
        if current_idx > 0:
            idx = current_idx - 1
            ax.text(idx, bm25_scores[idx] + 0.03, f'{bm25_scores[idx]:.1%}', 
                   ha='center', fontsize=14, fontweight='bold', color='#e74c3c')
            ax.text(idx, rocchio_scores[idx] + 0.03, f'{rocchio_scores[idx]:.1%}',
                   ha='center', fontsize=14, fontweight='bold', color='#2ecc71')
            ax.text(idx, tfidf_scores[idx] - 0.05, f'{tfidf_scores[idx]:.1%}',
                   ha='center', fontsize=14, fontweight='bold', color='#3498db')
        
        ax.set_xlabel('Dataset Scale', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Average Precision (MAP)', fontsize=16, fontweight='bold')
        ax.set_title(f'Scalability Analysis: Performance vs Corpus Size\n[Step {current_idx}/4]', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=13)
        ax.legend(loc='upper left', fontsize=14, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.set_ylim(0, 0.85)
        
        # Add annotation for TF-IDF drop
        if current_idx == 4:
            ax.annotate('TF-IDF\nDegradation!', 
                       xy=(2.5, tfidf_scores[2]), 
                       xytext=(2.5, 0.35),
                       arrowprops=dict(arrowstyle='->', color='red', lw=3),
                       fontsize=13, ha='center', color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    anim = animation.FuncAnimation(fig, animate, frames=4, interval=1000, repeat=True)
    anim.save(os.path.join(output_dir, 'scalability_animated.gif'), writer='pillow', fps=1)
    plt.close()
    print("✓ Created animated scalability plot (GIF)")

def create_heatmap_visualization(results, output_dir='ir_evaluation/results/figures'):
    """Create comprehensive heatmap of all results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data matrix
    datasets = ['CISI', 'MS MARCO 10K', 'MS MARCO 50K', 'MS MARCO 100K']
    models = ['TF-IDF', 'BM25', 'Rocchio']
    metrics = ['MAP', 'P@5', 'NDCG@10']
    
    # Create 3 subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for metric_idx, (metric_key, metric_name) in enumerate([('map', 'MAP'), ('p@5', 'P@5'), ('ndcg@10', 'NDCG@10')]):
        ax = axes[metric_idx]
        
        # Build data matrix
        data = []
        for model_key in ['tfidf', 'bm25', 'rocchio']:
            row = []
            for dataset_key in ['cisi', 'msmarco_10000', 'msmarco_50000', 'msmarco_100000']:
                if dataset_key in results:
                    row.append(results[dataset_key][model_key][metric_key])
                else:
                    row.append(0)
            data.append(row)
        
        data = np.array(data)
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=np.max(data))
        
        # Set ticks
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(datasets)
        ax.set_yticklabels(models)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values in cells
        for i in range(len(models)):
            for j in range(len(datasets)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold', fontsize=11)
        
        ax.set_title(f'{metric_name} Heatmap', fontsize=14, fontweight='bold', pad=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name, rotation=270, labelpad=20, fontweight='bold')
    
    plt.suptitle('Performance Heatmap: All Models × Datasets × Metrics', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_heatmap_comprehensive.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created comprehensive performance heatmap")

def create_3d_performance_plot(results, output_dir='ir_evaluation/results/figures'):
    """Create 3D performance visualization"""
    from mpl_toolkits.mplot3d import Axes3D
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Data
    datasets = [1.4, 10, 50, 100]  # In thousands
    models = ['TF-IDF', 'BM25', 'Rocchio']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for model_idx, (model_key, model_name, color) in enumerate(zip(['tfidf', 'bm25', 'rocchio'], models, colors)):
        map_scores = []
        p5_scores = []
        ndcg_scores = []
        
        for dataset_key in ['cisi', 'msmarco_10000', 'msmarco_50000', 'msmarco_100000']:
            if dataset_key in results:
                map_scores.append(results[dataset_key][model_key]['map'])
                p5_scores.append(results[dataset_key][model_key]['p@5'])
                ndcg_scores.append(results[dataset_key][model_key]['ndcg@10'])
        
        # Plot as 3D line with markers
        ax.plot(datasets, map_scores, ndcg_scores, 'o-', linewidth=3, markersize=10,
               label=model_name, color=color, alpha=0.8)
    
    ax.set_xlabel('Corpus Size (K docs)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('MAP', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('NDCG@10', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('3D Performance Space: MAP × NDCG × Scale', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12)
    
    plt.savefig(os.path.join(output_dir, 'performance_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created 3D performance visualization")

def create_ranking_quality_viz(results, output_dir='ir_evaluation/results/figures'):
    """Create ranking quality visualization with precision-recall curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    datasets_to_viz = [
        ('cisi', 'CISI (1.4K docs)'),
        ('msmarco_10000', 'MS MARCO (10K docs)'),
        ('msmarco_50000', 'MS MARCO (50K docs)'),
        ('msmarco_100000', 'MS MARCO (100K docs)')
    ]
    
    colors = {'tfidf': '#3498db', 'bm25': '#e74c3c', 'rocchio': '#2ecc71'}
    
    for idx, (dataset_key, dataset_name) in enumerate(datasets_to_viz):
        ax = axes[idx // 2, idx % 2]
        
        if dataset_key not in results:
            continue
        
        # Extract metrics
        for model_key, model_name in [('tfidf', 'TF-IDF'), ('bm25', 'BM25'), ('rocchio', 'Rocchio')]:
            data = results[dataset_key][model_key]
            
            # Create pseudo precision-recall points
            recalls = [data.get('r@5', 0), data.get('r@10', 0.8)]
            precisions = [data.get('p@5', 0), data.get('p@10', 0)]
            
            ax.plot(recalls, precisions, 'o-', linewidth=3, markersize=10,
                   label=f"{model_name} (MAP={data['map']:.3f})",
                   color=colors[model_key], alpha=0.8)
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(dataset_name, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
    
    plt.suptitle('Precision-Recall Trade-offs Across Datasets', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created precision-recall visualization")

def create_model_progression_animation(results, output_dir='ir_evaluation/results/figures'):
    """Create animated comparison of models evolving through datasets"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    datasets = ['CISI', 'MS MARCO\n10K', 'MS MARCO\n50K', 'MS MARCO\n100K']
    models = ['TF-IDF', 'BM25', 'Rocchio']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    def animate(frame):
        ax.clear()
        
        # Get data up to current frame
        dataset_keys = ['cisi', 'msmarco_10000', 'msmarco_50000', 'msmarco_100000']
        current_datasets = datasets[:frame+1]
        current_keys = dataset_keys[:frame+1]
        
        x = np.arange(len(current_datasets))
        width = 0.25
        
        for i, (model_key, model_name, color) in enumerate(zip(['tfidf', 'bm25', 'rocchio'], models, colors)):
            values = []
            for key in current_keys:
                if key in results:
                    values.append(results[key][model_key]['map'])
                else:
                    values.append(0)
            
            bars = ax.bar(x + (i-1)*width, values, width, label=model_name, 
                         color=color, alpha=0.85)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Mean Average Precision', fontsize=14, fontweight='bold')
        ax.set_title(f'Model Comparison Across Scales [Dataset {frame+1}/4]',
                    fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(current_datasets, fontsize=12)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 0.85)
    
    anim = animation.FuncAnimation(fig, animate, frames=4, interval=1200, repeat=True)
    anim.save(os.path.join(output_dir, 'model_comparison_animated.gif'), writer='pillow', fps=1)
    plt.close()
    print("✓ Created animated model comparison (GIF)")

def create_algorithm_explanation_viz(output_dir='ir_evaluation/results/figures'):
    """Create visual explanation of how each algorithm works"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # TF-IDF visualization
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9, 'TF-IDF Algorithm', ha='center', fontsize=16, fontweight='bold')
    
    # Formula
    formula_box = FancyBboxPatch((1, 7), 8, 1.2, boxstyle="round,pad=0.1", 
                                 edgecolor='#3498db', facecolor='#ecf0f1', linewidth=2)
    ax.add_patch(formula_box)
    ax.text(5, 7.6, 'score = TF(term) × IDF(term)', ha='center', fontsize=11, 
           family='monospace', fontweight='bold')
    
    # Steps
    steps = [
        '1. Count term frequency',
        '2. Calculate IDF (rarity)',
        '3. Multiply: TF × IDF',
        '4. Cosine similarity'
    ]
    
    for i, step in enumerate(steps):
        y_pos = 6 - i*1
        step_box = FancyBboxPatch((0.5, y_pos-0.3), 9, 0.6, boxstyle="round,pad=0.05",
                                 edgecolor='#3498db', facecolor='white', linewidth=1.5)
        ax.add_patch(step_box)
        ax.text(5, y_pos, step, ha='center', fontsize=10)
    
    ax.text(5, 1.5, '✓ Simple & Interpretable', ha='center', fontsize=11, 
           color='green', fontweight='bold')
    ax.text(5, 0.8, '✗ IDF degrades at scale', ha='center', fontsize=11,
           color='red', fontweight='bold')
    
    # BM25 visualization
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9, 'BM25 Algorithm', ha='center', fontsize=16, fontweight='bold')
    
    formula_box = FancyBboxPatch((0.5, 7), 9, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='#e74c3c', facecolor='#ecf0f1', linewidth=2)
    ax.add_patch(formula_box)
    ax.text(5, 7.6, 'score = IDF × [TF / (TF + k1×(1-b+b×dl))]', ha='center',
           fontsize=9, family='monospace', fontweight='bold')
    
    steps = [
        '1. Term frequency (saturated)',
        '2. Length normalization',
        '3. IDF weighting',
        '4. Parameter tuning (k1, b)'
    ]
    
    for i, step in enumerate(steps):
        y_pos = 6 - i*1
        step_box = FancyBboxPatch((0.5, y_pos-0.3), 9, 0.6, boxstyle="round,pad=0.05",
                                 edgecolor='#e74c3c', facecolor='white', linewidth=1.5)
        ax.add_patch(step_box)
        ax.text(5, y_pos, step, ha='center', fontsize=10)
    
    ax.text(5, 1.5, '✓ Best Performance', ha='center', fontsize=11,
           color='green', fontweight='bold')
    ax.text(5, 0.8, '✓ Scales Excellently', ha='center', fontsize=11,
           color='green', fontweight='bold')
    
    # Rocchio visualization
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9, 'Rocchio Algorithm', ha='center', fontsize=16, fontweight='bold')
    
    formula_box = FancyBboxPatch((0.5, 7), 9, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='#2ecc71', facecolor='#ecf0f1', linewidth=2)
    ax.add_patch(formula_box)
    ax.text(5, 7.6, 'q_new = α×q + β×rel - γ×nonrel', ha='center',
           fontsize=11, family='monospace', fontweight='bold')
    
    steps = [
        '1. Initial TF-IDF query',
        '2. Get relevant docs',
        '3. Update query vector',
        '4. Re-rank results'
    ]
    
    for i, step in enumerate(steps):
        y_pos = 6 - i*1
        step_box = FancyBboxPatch((0.5, y_pos-0.3), 9, 0.6, boxstyle="round,pad=0.05",
                                 edgecolor='#2ecc71', facecolor='white', linewidth=1.5)
        ax.add_patch(step_box)
        ax.text(5, y_pos, step, ha='center', fontsize=10)
    
    ax.text(5, 1.5, '✓ Learns from feedback', ha='center', fontsize=11,
           color='green', fontweight='bold')
    ax.text(5, 0.8, '⚠ Needs relevance data', ha='center', fontsize=11,
           color='orange', fontweight='bold')
    
    plt.suptitle('Algorithm Comparison: How They Work', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_explanations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created algorithm explanation visualization")

def main():
    print("=" * 70)
    print("Creating Enhanced Visualizations & Animations")
    print("=" * 70)
    
    print("\n[1/7] Loading all results...")
    results = load_all_results()
    print(f"✓ Loaded {len(results)} result files")
    
    print("\n[2/7] Creating animated scalability plot...")
    create_animated_scalability(results)
    
    print("\n[3/7] Creating comprehensive heatmap...")
    create_heatmap_visualization(results)
    
    print("\n[4/7] Creating 3D performance visualization...")
    create_3d_performance_plot(results)
    
    print("\n[5/7] Creating precision-recall curves...")
    create_ranking_quality_viz(results)
    
    print("\n[6/7] Creating animated model comparison...")
    create_model_progression_animation(results)
    
    print("\n[7/7] Creating algorithm explanations...")
    create_algorithm_explanation_viz()
    
    print("\n" + "=" * 70)
    print("✓ All Enhanced Visualizations Created!")
    print("=" * 70)
    print("\nNew files created:")
    print("  🎬 scalability_animated.gif")
    print("  🎬 model_comparison_animated.gif")
    print("  🔥 performance_heatmap_comprehensive.png")
    print("  📊 performance_3d.png")
    print("  📈 precision_recall_curves.png")
    print("  🎓 algorithm_explanations.png")
    print("=" * 70)

if __name__ == "__main__":
    main()

