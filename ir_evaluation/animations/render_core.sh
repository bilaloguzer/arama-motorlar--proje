#!/bin/bash

# Quick Render - Core 4 Animations Only
# Renders the most important animations for quick presentations

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Rendering CORE 4 Animations (Quick Mode)"
echo "=========================================="
echo ""
echo "This renders only the essential animations:"
echo "  1. ScalabilityEvolution - Main findings"
echo "  2. IDFCompressionExplanation - Why TF-IDF fails"
echo "  3. AlgorithmShowdown - Introduction"
echo "  4. SpeedAnalysis - Computational performance"
echo ""
echo "Estimated time: 5-8 minutes"
echo ""
read -p "Press Enter to start..."
echo ""

source ../../venv/bin/activate

echo "[1/4] Main Finding: Scalability Evolution"
manim -qh scalability_animation.py ScalabilityEvolution
echo "✓ Complete"
echo ""

echo "[2/4] Educational: TF-IDF Degradation"
manim -qh tfidf_degradation.py IDFCompressionExplanation
echo "✓ Complete"
echo ""

echo "[3/4] Introduction: Algorithm Showdown"
manim -qh algorithm_comparison.py AlgorithmShowdown
echo "✓ Complete"
echo ""

echo "[4/4] Performance: Speed Analysis"
manim -qh speed_and_efficiency.py SpeedAnalysis
echo "✓ Complete"
echo ""

echo "=========================================="
echo "✓ CORE 4 ANIMATIONS COMPLETE!"
echo "=========================================="
echo ""
echo "Videos ready at: media/videos/*/1080p60/"
echo ""
echo "Presentation order:"
echo "  1. AlgorithmShowdown.mp4 (intro)"
echo "  2. ScalabilityEvolution.mp4 (main finding)"
echo "  3. IDFCompressionExplanation.mp4 (explanation)"
echo "  4. SpeedAnalysis.mp4 (performance)"
echo ""
