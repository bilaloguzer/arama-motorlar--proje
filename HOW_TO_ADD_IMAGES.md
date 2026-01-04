# How to Add Images to Your LaTeX Report

## Quick Reference

### Basic Image Syntax

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{path/to/image.png}
\caption{Your caption here}
\label{fig:yourLabel}
\end{figure}
```

## Example: Add Your Visualizations

### 1. After Section 6.6 (Statistical Significance)

Add this code RIGHT AFTER the "Statistical Significance" subsection (around line 665):

```latex
\subsection{Visualizations}

Figure~\ref{fig:scalability} shows the scalability analysis across different corpus sizes.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{ir_evaluation/results/figures/scalability_analysis.png}
\caption{Scalability Analysis: MAP performance across corpus sizes showing BM25's stability and TF-IDF's degradation.}
\label{fig:scalability}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{ir_evaluation/results/figures/comprehensive_comparison.png}
\caption{Comprehensive Comparison: All metrics across datasets demonstrating BM25's consistent superiority.}
\label{fig:comprehensive}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{ir_evaluation/results/figures/stemming_impact.png}
\caption{Stemming Impact: Performance improvements with Porter Stemmer (+9.8\% MAP for BM25).}
\label{fig:stemming}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{ir_evaluation/results/figures/speed_analysis.png}
\caption{Speed Analysis: Processing time and throughput demonstrating linear scaling.}
\label{fig:speed}
\end{figure}
```

### 2. Add Side-by-Side Images (Optional)

For two images side by side:

```latex
\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{ir_evaluation/results/figures/model_comparison.png}
    \caption{Model Comparison}
    \label{fig:model_comp}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{ir_evaluation/results/figures/results_heatmap.png}
    \caption{Results Heatmap}
    \label{fig:heatmap}
\end{subfigure}
\caption{Additional Performance Visualizations}
\label{fig:additional}
\end{figure}
```

### 3. Add More Figures in Appendix

At the end of the Appendix section (around line 930), add:

```latex
\section{Additional Visualizations}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{ir_evaluation/results/figures/performance_heatmap_comprehensive.png}
\caption{Comprehensive Performance Heatmap across all metrics and datasets.}
\label{fig:heatmap_full}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{ir_evaluation/results/figures/precision_recall_curves.png}
\caption{Precision-Recall Curves for all three algorithms.}
\label{fig:pr_curves}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{ir_evaluation/results/figures/performance_3d.png}
\caption{3D Performance Space visualization.}
\label{fig:3d}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{ir_evaluation/results/figures/algorithm_explanations.png}
\caption{Visual explanation of how each algorithm works.}
\label{fig:algo_explain}
\end{figure}
```

## Image Size Options

- `width=0.5\textwidth` - Half page width
- `width=0.8\textwidth` - 80% of page width (recommended for single images)
- `width=\textwidth` - Full page width
- `height=10cm` - Fixed height
- `scale=0.5` - 50% of original size

## Placement Options

- `[H]` - Place HERE (exactly where you write it) - **recommended**
- `[h]` - Place approximately here
- `[t]` - Place at top of page
- `[b]` - Place at bottom of page
- `[p]` - Place on separate page

## Reference Images in Text

After defining a figure with `\label{fig:scalability}`, you can reference it:

```latex
As shown in Figure~\ref{fig:scalability}, BM25 maintains stable performance...
```

This will automatically insert the figure number (e.g., "Figure 1").

## Your Available Images

Located in `ir_evaluation/results/figures/`:

1. ✅ `scalability_analysis.png` - Main performance chart
2. ✅ `comprehensive_comparison.png` - All metrics
3. ✅ `stemming_impact.png` - Preprocessing impact
4. ✅ `speed_analysis.png` - Timing analysis
5. `performance_heatmap_comprehensive.png` - Heatmap
6. `precision_recall_curves.png` - PR curves
7. `performance_3d.png` - 3D visualization
8. `algorithm_explanations.png` - How algorithms work
9. `model_comparison.png` - CISI results
10. `results_heatmap.png` - Results heatmap
11. `performance_radar.png` - Radar chart
12. `metric_breakdown.png` - Individual metrics
13. `speed_comparison.png` - Speed comparison
14. `msmarco_50k_comprehensive.png` - Large dataset results

## Compilation Note

When compiling with Overleaf or local LaTeX:
- Make sure to upload the entire `ir_evaluation/results/figures/` folder
- Or adjust paths to where your images are located
- If using Overleaf: zip the figures folder and upload it

## Quick Fix: If Images Don't Show

If you get errors about missing images:

1. **On Overleaf**: Upload the `figures` folder to your project
2. **Locally**: Make sure paths are correct relative to your `.tex` file
3. **Relative path**: Use `ir_evaluation/results/figures/imagename.png`
4. **Absolute path**: Not recommended, but possible with full path

## Example: Complete Section with Images

```latex
\section{Results}

\subsection{Overall Performance}

Table~\ref{tab:results} shows the numerical results, while Figure~\ref{fig:scalability}
provides a visual comparison across corpus sizes.

\begin{table}[H]
% ... your table ...
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{ir_evaluation/results/figures/scalability_analysis.png}
\caption{Performance across different corpus sizes.}
\label{fig:scalability}
\end{figure}

As Figure~\ref{fig:scalability} demonstrates, BM25 maintains consistent performance...
```

---

## Ready-to-Use Code Block

Copy this entire block and paste it after line 665 in your `.tex` file:

```latex
\subsection{Visual Results}

The following figures provide visual representation of our key findings.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{ir_evaluation/results/figures/scalability_analysis.png}
\caption{Scalability Analysis: MAP performance progression from 1.4K to 100K documents. BM25 demonstrates remarkable stability (74-75\% MAP) while TF-IDF suffers 23\% degradation due to IDF compression.}
\label{fig:scalability}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{ir_evaluation/results/figures/comprehensive_comparison.png}
\caption{Comprehensive Performance Comparison: BM25 outperforms alternatives across all metrics (MAP, P@5, P@10, NDCG@10) on all datasets.}
\label{fig:comprehensive}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{ir_evaluation/results/figures/stemming_impact.png}
\caption{Stemming Impact Analysis: Porter Stemmer improves all models, with BM25 benefiting most (+9.8\% MAP, +16.9\% NDCG@10).}
\label{fig:stemming}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{ir_evaluation/results/figures/speed_analysis.png}
\caption{Computational Performance: Linear scaling demonstrated across all models, with throughput of $\sim$3,000 documents/second on M1 Pro.}
\label{fig:speed}
\end{figure}
```

Save and compile!
