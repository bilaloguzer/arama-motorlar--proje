# IR Project Animations - 3Blue1Brown Style

Professional mathematical animations for your Information Retrieval project using Manim (the library used by 3Blue1Brown).

## 📦 Installation

### Step 1: Install Manim and Dependencies

```bash
cd ir_evaluation/animations
chmod +x setup_manim.sh
./setup_manim.sh
```

This will install:
- Manim Community Edition
- FFmpeg (video rendering)
- LaTeX (optional, for better text)
- Required Python packages

### Step 2: Test Installation

```bash
manim -pql test_scene.py CircleExample
```

If you see a blue circle animation, you're ready to go!

## 🎬 Available Animations

### 1. Scalability Evolution (`scalability_animation.py`)

**Scenes:**
- `ScalabilityEvolution` - Main animation showing performance across corpus sizes
- `ScalabilityBarChart` - Animated bar chart progression

**What it shows:**
- BM25's stable 75% MAP performance
- TF-IDF's -23% degradation at 100K docs
- Rocchio's middle-ground performance
- Visual highlighting of key findings

**Render:**
```bash
# Low quality preview (fast)
manim -pql scalability_animation.py ScalabilityEvolution

# High quality (for presentations)
manim -pqh scalability_animation.py ScalabilityEvolution

# 4K quality (for video/publication)
manim -qk scalability_animation.py ScalabilityEvolution
```

### 2. TF-IDF Degradation (`tfidf_degradation.py`)

**Scenes:**
- `IDFCompressionExplanation` - Educational explanation of IDF compression
- `BM25Resistance` - Why BM25 doesn't degrade

**What it shows:**
- IDF formula breakdown
- Side-by-side comparison at 50K vs 100K
- Mathematical explanation of compression
- BM25's three advantages

**Render:**
```bash
manim -pqh tfidf_degradation.py IDFCompressionExplanation
manim -pqh tfidf_degradation.py BM25Resistance
```

### 3. Algorithm Comparison (`algorithm_comparison.py`)

**Scenes:**
- `AlgorithmShowdown` - Head-to-head comparison of all three algorithms
- `StemmingImpact` - Visual demonstration of stemming benefits

**What it shows:**
- Algorithm cards with strengths/weaknesses
- Performance comparison bars
- Winner highlighting
- Stemming term conflation example

**Render:**
```bash
manim -pqh algorithm_comparison.py AlgorithmShowdown
manim -pqh algorithm_comparison.py StemmingImpact
```

## 🎨 Rendering Options

### Quality Settings

| Flag | Quality | Resolution | Use Case | Speed |
|------|---------|------------|----------|-------|
| `-ql` | Low | 480p | Quick preview | Fast ⚡ |
| `-qm` | Medium | 720p | Testing | Medium |
| `-qh` | High | 1080p | Presentations | Slow |
| `-qk` | 4K | 2160p | Publication | Very Slow |

### Other Flags

- `-p` = Preview (opens video when done)
- `-s` = Save last frame as image
- `-a` = Render all scenes in file
- `--format=gif` = Output as GIF instead of mp4

### Examples

```bash
# Preview in low quality (fastest)
manim -pql scalability_animation.py ScalabilityEvolution

# High quality without preview (for batch rendering)
manim -qh scalability_animation.py ScalabilityEvolution

# Save as GIF
manim -pqh --format=gif algorithm_comparison.py AlgorithmShowdown

# Render all scenes in a file
manim -pqh -a scalability_animation.py

# Save last frame as PNG
manim -sqh scalability_animation.py ScalabilityEvolution
```

## 📁 Output Location

All rendered videos are saved to:
```
ir_evaluation/animations/media/videos/
```

Structure:
```
media/
└── videos/
    ├── scalability_animation/
    │   ├── 480p15/  (low quality)
    │   ├── 1080p60/ (high quality)
    │   └── 2160p60/ (4K)
    ├── tfidf_degradation/
    └── algorithm_comparison/
```

## 🎯 Usage in Presentation

### For LaTeX Report

1. Render high quality:
```bash
manim -qh scalability_animation.py ScalabilityEvolution
```

2. Find output:
```
media/videos/scalability_animation/1080p60/ScalabilityEvolution.mp4
```

3. Convert to images for LaTeX:
```bash
# Extract key frames
ffmpeg -i ScalabilityEvolution.mp4 -vf "select='eq(n,0)'" -vsync 0 frame1.png
ffmpeg -i ScalabilityEvolution.mp4 -vf "select='eq(n,100)'" -vsync 0 frame2.png
```

### For Presentation Slides

1. Render and use directly:
```bash
manim -pqh scalability_animation.py ScalabilityEvolution
```

2. Embed in PowerPoint/Keynote:
   - Insert → Video
   - Choose the .mp4 file
   - Set to auto-play on click

### For GitHub README

1. Render as GIF:
```bash
manim -pqm --format=gif scalability_animation.py ScalabilityEvolution
```

2. Add to markdown:
```markdown
![Scalability Analysis](animations/media/videos/scalability_animation/ScalabilityEvolution.gif)
```

## 🎓 Customization

### Modify Data

Edit the data arrays in each file:

```python
# In scalability_animation.py
corpus_sizes = ["1.4K", "10K", "50K", "100K"]
tfidf_map = [0.196, 0.497, 0.508, 0.392]
bm25_map = [0.205, 0.646, 0.751, 0.747]
rocchio_map = [0.147, 0.610, 0.667, 0.626]
```

### Change Colors

Manim color constants:
- `RED`, `BLUE`, `GREEN`, `YELLOW`, `PURPLE`, `ORANGE`
- `GREY`, `WHITE`, `BLACK`
- Custom: `rgb_to_color([0.5, 0.2, 0.8])`

### Adjust Timing

```python
# Slower animation
self.play(Create(line), run_time=3)  # 3 seconds

# Faster
self.play(Create(line), run_time=0.5)  # 0.5 seconds

# Wait between animations
self.wait(2)  # Wait 2 seconds
```

## 🔧 Troubleshooting

### "manim: command not found"

```bash
pip install --upgrade manim
```

### LaTeX errors

Either install LaTeX:
```bash
brew install --cask mactex-no-gui
```

Or use simple text (no LaTeX):
```python
# Replace MathTex with Text
Text("IDF = log(N/df)", font_size=36)
```

### Slow rendering

- Use `-ql` for quick previews
- Only use `-qh` or `-qk` for final renders
- Close other applications

### Video won't play

Make sure you have VLC or QuickTime installed:
```bash
brew install --cask vlc
```

## 📊 Performance Tips

1. **Preview first**: Always test with `-ql` before high-quality render
2. **Batch rendering**: Use `-a` to render all scenes at once
3. **Scene selection**: Only render the scenes you need
4. **Cache usage**: Manim caches intermediate results (in `media/` folder)

## 🎬 Advanced: Create Your Own Scene

```python
from manim import *

class MyCustomScene(Scene):
    def construct(self):
        # Your animation code here
        title = Text("My IR Animation", font_size=48)
        self.play(Write(title))
        self.wait(1)
```

Render:
```bash
manim -pql my_animation.py MyCustomScene
```

## 📚 Resources

- **Manim Documentation**: https://docs.manim.community/
- **3Blue1Brown's Manim**: https://github.com/3b1b/manim
- **Example Gallery**: https://docs.manim.community/en/stable/examples.html
- **Tutorial Series**: https://www.youtube.com/watch?v=rUsUrbWb2D4

## 🎯 Recommended Workflow

1. **Test**: `manim -pql test_scene.py CircleExample`
2. **Preview**: `manim -pql your_scene.py YourScene`
3. **Refine**: Edit code, repeat step 2
4. **Final render**: `manim -pqh your_scene.py YourScene`
5. **For publication**: `manim -qk your_scene.py YourScene`

## 💡 Tips for Best Results

1. **Keep animations smooth**: 2-3 seconds per main transition
2. **Use color consistently**: Same colors for same concepts
3. **Add pauses**: `self.wait(1)` between major sections
4. **Highlight key points**: Use yellow boxes/arrows
5. **End with summary**: Always conclude with main finding

---

**Ready to create stunning visualizations!** 🎨

Start with the test scene, then move to your data visualizations. The animations will make your project stand out! 🚀
