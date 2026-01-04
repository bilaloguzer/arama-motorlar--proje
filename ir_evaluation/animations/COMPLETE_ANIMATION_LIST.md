## 🎬 Complete Animation Catalog

All Manim animations for your IR project, organized by report section.

---

## 📋 Quick Render Commands

### Render ALL animations in high quality (1080p)
```bash
cd ir_evaluation/animations

# Background & Introduction
manim -pqh methodology_animations.py DatasetOverview
manim -pqh algorithm_comparison.py AlgorithmShowdown

# Methodology Section
manim -pqh methodology_animations.py PreprocessingPipeline
manim -pqh methodology_animations.py EvaluationMetrics
manim -pqh methodology_animations.py ExperimentalDesign

# Results Section
manim -pqh scalability_animation.py ScalabilityEvolution
manim -pqh scalability_animation.py ScalabilityBarChart
manim -pqh comprehensive_visualizations.py ComprehensiveComparison
manim -pqh comprehensive_visualizations.py MetricBreakdown
manim -pqh comprehensive_visualizations.py PerformanceHeatmap

# TF-IDF Degradation
manim -pqh tfidf_degradation.py IDFCompressionExplanation
manim -pqh tfidf_degradation.py BM25Resistance

# Stemming Analysis
manim -pqh algorithm_comparison.py StemmingImpact

# Speed Analysis
manim -pqh speed_and_efficiency.py SpeedAnalysis
manim -pqh speed_and_efficiency.py LinearScaling
manim -pqh speed_and_efficiency.py MemoryUsage
manim -pqh speed_and_efficiency.py EfficiencyComparison
```

---

## 📊 Animations by Report Section

### 1. Introduction / Background

#### **DatasetOverview** (`methodology_animations.py`)
- **What it shows:** CISI and MS MARCO dataset cards with specifications
- **Duration:** ~20 seconds
- **Use for:** Introducing your benchmark datasets
- **Key visuals:** Side-by-side dataset comparison cards

#### **AlgorithmShowdown** (`algorithm_comparison.py`)
- **What it shows:** Beautiful cards for each algorithm with pros/cons
- **Duration:** ~40 seconds
- **Use for:** Introducing TF-IDF, BM25, and Rocchio
- **Key visuals:** Winner badges, performance comparison

---

### 2. Methodology

#### **PreprocessingPipeline** (`methodology_animations.py`)
- **What it shows:** Step-by-step text transformation
- **Example:** "Information Retrieval Systems" → ["inform", "retriev", "system"]
- **Duration:** ~25 seconds
- **Use for:** Explaining your preprocessing steps
- **Key visuals:** Arrow-connected pipeline stages

#### **EvaluationMetrics** (`methodology_animations.py`)
- **What it shows:** MAP, P@k, NDCG explained with examples
- **Duration:** ~60 seconds
- **Use for:** Explaining your evaluation approach
- **Key visuals:** Example rankings with precision calculations

#### **ExperimentalDesign** (`methodology_animations.py`)
- **What it shows:** Independent vs dependent variables
- **Duration:** ~20 seconds
- **Use for:** Describing your experimental setup
- **Key visuals:** Variable breakdown, total experiments count

---

### 3. Results - Main Findings

#### **ScalabilityEvolution** (`scalability_animation.py`) ⭐ **PRIMARY**
- **What it shows:** Performance across 1.4K → 100K documents
- **Duration:** ~30 seconds
- **Use for:** Your main finding - BM25 wins, TF-IDF degrades
- **Key visuals:**
  - Line graphs for all three algorithms
  - Yellow arrow highlighting TF-IDF's -23% degradation
  - Blue box highlighting BM25's stability at 75%
  - Color-coded legend

#### **ScalabilityBarChart** (`scalability_animation.py`)
- **What it shows:** Animated bars for each dataset size
- **Duration:** ~40 seconds
- **Use for:** Alternative visualization of same data
- **Key visuals:** Growing bars, winner highlighting per dataset

#### **ComprehensiveComparison** (`comprehensive_visualizations.py`)
- **What it shows:** All metrics (MAP, P@5, NDCG) across all datasets
- **Duration:** ~50 seconds
- **Use for:** Complete results overview
- **Key visuals:** Grouped bar charts transitioning through datasets

#### **MetricBreakdown** (`comprehensive_visualizations.py`)
- **What it shows:** Individual metric explanations with BM25 winning all
- **Duration:** ~25 seconds
- **Use for:** Emphasizing BM25's dominance across metrics
- **Key visuals:** Metric definitions, conclusion slide

#### **PerformanceHeatmap** (`comprehensive_visualizations.py`)
- **What it shows:** Color-coded grid of all results
- **Duration:** ~35 seconds
- **Use for:** At-a-glance performance comparison
- **Key visuals:** Animated cell-by-cell heatmap, BM25 column highlighted

---

### 4. Results - TF-IDF Degradation

#### **IDFCompressionExplanation** (`tfidf_degradation.py`) ⭐ **EDUCATIONAL**
- **What it shows:** Mathematical explanation of IDF compression
- **Duration:** ~45 seconds
- **Use for:** Explaining WHY TF-IDF fails at scale
- **Key visuals:**
  - IDF formula breakdown
  - Side-by-side 50K vs 100K comparison
  - Same IDF value highlighting
  - Performance collapse bar chart

#### **BM25Resistance** (`tfidf_degradation.py`)
- **What it shows:** Why BM25 doesn't degrade
- **Duration:** ~30 seconds
- **Use for:** Explaining BM25's three advantages
- **Key visuals:** Three checkmarked advantages, stability stats

---

### 5. Results - Stemming Impact

#### **StemmingImpact** (`algorithm_comparison.py`)
- **What it shows:** Porter Stemmer performance improvements
- **Duration:** ~35 seconds
- **Use for:** Demonstrating preprocessing benefits
- **Key visuals:**
  - Term conflation example ("retrieve" variants → "retriev")
  - Improvement bars (+7.5% to +9.8%)
  - BM25 highlighted as biggest beneficiary

---

### 6. Results - Computational Performance

#### **SpeedAnalysis** (`speed_and_efficiency.py`)
- **What it shows:** Processing time for each algorithm at each scale
- **Duration:** ~40 seconds
- **Use for:** Showing computational efficiency
- **Key visuals:**
  - Animated bar charts for each corpus size
  - Rocchio highlighted as fastest
  - Throughput counter animation (~2,973 docs/sec)

#### **LinearScaling** (`speed_and_efficiency.py`)
- **What it shows:** O(n) complexity demonstration
- **Duration:** ~25 seconds
- **Use for:** Proving linear scalability
- **Key visuals:**
  - Line plot through data points
  - O(n) formula
  - "5× data = 5× time" annotation

#### **MemoryUsage** (`speed_and_efficiency.py`)
- **What it shows:** RAM requirements across scales
- **Duration:** ~25 seconds
- **Use for:** Hardware requirements discussion
- **Key visuals:**
  - Horizontal bars showing memory usage
  - 16GB system capacity
  - Projection to 500K documents

#### **EfficiencyComparison** (`speed_and_efficiency.py`)
- **What it shows:** Speed vs Accuracy trade-off plot
- **Duration:** ~30 seconds
- **Use for:** Showing BM25 as optimal balance
- **Key visuals:**
  - 2D scatter plot
  - BM25 highlighted as "Optimal Balance"
  - Rocchio: fast but less accurate
  - TF-IDF: slow AND inaccurate

---

## 📐 Animation Statistics

| Category | Animations | Total Duration |
|----------|-----------|----------------|
| **Introduction** | 2 | ~60 seconds |
| **Methodology** | 3 | ~105 seconds |
| **Results - Main** | 5 | ~180 seconds |
| **Results - Degradation** | 2 | ~75 seconds |
| **Results - Stemming** | 1 | ~35 seconds |
| **Results - Speed** | 4 | ~120 seconds |
| **TOTAL** | **17 animations** | **~10 minutes** |

---

## 🎯 Recommended Presentation Flow

### Option 1: Complete Presentation (10 minutes)
Play all animations in order for a full project walkthrough.

### Option 2: Key Findings Only (3 minutes)
1. `AlgorithmShowdown` - Introduce algorithms (40s)
2. `ScalabilityEvolution` - Main finding (30s)
3. `IDFCompressionExplanation` - Why TF-IDF fails (45s)
4. `SpeedAnalysis` - Computational efficiency (40s)

### Option 3: Technical Deep Dive (7 minutes)
1. `DatasetOverview` (20s)
2. `PreprocessingPipeline` (25s)
3. `EvaluationMetrics` (60s)
4. `ScalabilityEvolution` (30s)
5. `ComprehensiveComparison` (50s)
6. `IDFCompressionExplanation` (45s)
7. `BM25Resistance` (30s)
8. `StemmingImpact` (35s)
9. `EfficiencyComparison` (30s)

---

## 🎬 Batch Rendering Scripts

### Render all core animations (high quality)
```bash
#!/bin/bash
cd ir_evaluation/animations

echo "Rendering core animations..."

manim -qh scalability_animation.py ScalabilityEvolution
manim -qh tfidf_degradation.py IDFCompressionExplanation
manim -qh algorithm_comparison.py AlgorithmShowdown
manim -qh speed_and_efficiency.py SpeedAnalysis

echo "Core animations complete!"
```

### Render all methodology animations
```bash
manim -qh methodology_animations.py PreprocessingPipeline
manim -qh methodology_animations.py EvaluationMetrics
manim -qh methodology_animations.py ExperimentalDesign
manim -qh methodology_animations.py DatasetOverview
```

### Render all results animations
```bash
manim -qh scalability_animation.py ScalabilityEvolution
manim -qh scalability_animation.py ScalabilityBarChart
manim -qh comprehensive_visualizations.py ComprehensiveComparison
manim -qh comprehensive_visualizations.py MetricBreakdown
manim -qh comprehensive_visualizations.py PerformanceHeatmap
```

---

## 📦 File Locations After Rendering

All videos saved to:
```
ir_evaluation/animations/media/videos/[script_name]/1080p60/[SceneName].mp4
```

Example:
```
media/videos/scalability_animation/1080p60/ScalabilityEvolution.mp4
media/videos/tfidf_degradation/1080p60/IDFCompressionExplanation.mp4
```

---

## 💡 Usage Tips

### For LaTeX Report
1. Render high quality: `manim -qh script.py SceneName`
2. Extract key frame: `ffmpeg -i video.mp4 -ss 00:00:05 -frames:v 1 frame.png`
3. Include in LaTeX: `\includegraphics[width=0.9\textwidth]{frame.png}`

### For Presentation Slides
1. Render high quality
2. Insert → Video → Choose .mp4
3. Set to auto-play or click-to-play
4. Test playback before presenting!

### For GitHub README
1. Render as GIF: `manim -pqh --format=gif script.py SceneName`
2. Upload to repo
3. Embed: `![Animation](animations/media/SceneName.gif)`

---

## 🎨 Customization

All animations use your actual data. To modify:

1. **Change colors:** Edit color constants (BLUE, RED, GREEN, YELLOW)
2. **Adjust timing:** Modify `run_time` parameters
3. **Update data:** Change data arrays in each script
4. **Add scenes:** Copy existing scene structure

---

**You now have 17 professional animations covering every aspect of your IR project!** 🚀

Each animation is designed to explain one key concept clearly and beautifully, just like 3Blue1Brown's educational videos.
