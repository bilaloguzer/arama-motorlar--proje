#!/bin/bash

# Manim Installation Script for IR Animations
# This script installs Manim and dependencies for creating 3Blue1Brown-style animations

echo "=========================================="
echo "Installing Manim for IR Visualizations"
echo "=========================================="

# Check Python version
python3 --version

# Install Manim Community Edition (easier to use than manimgl)
echo "\n[1/4] Installing Manim Community Edition..."
pip install manim

# Install additional dependencies
echo "\n[2/4] Installing additional dependencies..."
pip install pandas numpy matplotlib

# Check if FFmpeg is installed (required for video rendering)
echo "\n[3/4] Checking FFmpeg installation..."
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "⚠️  Please install Homebrew first: https://brew.sh"
        echo "Then run: brew install ffmpeg"
        exit 1
    fi
else
    echo "✓ FFmpeg is already installed"
fi

# Check if LaTeX is installed (optional, for better text rendering)
echo "\n[4/4] Checking LaTeX installation..."
if ! command -v latex &> /dev/null; then
    echo "⚠️  LaTeX not found (optional but recommended)"
    echo "To install: brew install --cask mactex-no-gui"
else
    echo "✓ LaTeX is already installed"
fi

echo "\n=========================================="
echo "✓ Manim installation complete!"
echo "=========================================="
echo "\nTest the installation with:"
echo "  cd ir_evaluation/animations"
echo "  manim -pql test_scene.py CircleExample"
echo "\nRender options:"
echo "  -pql = Preview, Quality Low (fast)"
echo "  -pqh = Preview, Quality High"
echo "  -qk  = Quality 4K (no preview)"
echo "=========================================="
