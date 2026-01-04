#!/bin/bash

# Batch Render All IR Animations
# Renders all 17 animations in high quality (1080p60)

set -e  # Exit on error

cd "$(dirname "$0")"

echo "=========================================="
echo "IR Project - Batch Animation Rendering"
echo "=========================================="
echo ""
echo "Total animations: 17"
echo "Quality: High (1080p60)"
echo "Estimated time: 20-30 minutes"
echo ""
read -p "Press Enter to start rendering..."
echo ""

# Activate virtual environment
source ../../venv/bin/activate

# Counter
total=17
current=0

render() {
    current=$((current + 1))
    echo ""
    echo "=========================================="
    echo "[$current/$total] Rendering: $2"
    echo "File: $1"
    echo "=========================================="
    manim -qh "$1" "$2"
    echo "✓ Complete: $2"
}

echo "=== SECTION 1: Introduction & Background (2 animations) ==="
render "methodology_animations.py" "DatasetOverview"
render "algorithm_comparison.py" "AlgorithmShowdown"

echo ""
echo "=== SECTION 2: Methodology (3 animations) ==="
render "methodology_animations.py" "PreprocessingPipeline"
render "methodology_animations.py" "EvaluationMetrics"
render "methodology_animations.py" "ExperimentalDesign"

echo ""
echo "=== SECTION 3: Results - Main Findings (5 animations) ==="
render "scalability_animation.py" "ScalabilityEvolution"
render "scalability_animation.py" "ScalabilityBarChart"
render "comprehensive_visualizations.py" "ComprehensiveComparison"
render "comprehensive_visualizations.py" "MetricBreakdown"
render "comprehensive_visualizations.py" "PerformanceHeatmap"

echo ""
echo "=== SECTION 4: Results - TF-IDF Degradation (2 animations) ==="
render "tfidf_degradation.py" "IDFCompressionExplanation"
render "tfidf_degradation.py" "BM25Resistance"

echo ""
echo "=== SECTION 5: Results - Stemming (1 animation) ==="
render "algorithm_comparison.py" "StemmingImpact"

echo ""
echo "=== SECTION 6: Results - Speed Analysis (4 animations) ==="
render "speed_and_efficiency.py" "SpeedAnalysis"
render "speed_and_efficiency.py" "LinearScaling"
render "speed_and_efficiency.py" "MemoryUsage"
render "speed_and_efficiency.py" "EfficiencyComparison"

echo ""
echo "=========================================="
echo "✓ ALL ANIMATIONS RENDERED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Total: $current/$total animations"
echo ""
echo "Output location: media/videos/[script_name]/1080p60/"
echo ""
echo "Next steps:"
echo "1. Check videos: open media/videos/"
echo "2. Test playback with QuickTime or VLC"
echo "3. Add to presentation slides"
echo ""
echo "For LaTeX: Extract frames with:"
echo "  ffmpeg -i video.mp4 -ss 00:00:05 -frames:v 1 frame.png"
echo ""
