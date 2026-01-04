"""
Test Scene - Verify Manim Installation
Run: manim -pql test_scene.py CircleExample
"""

from manim import *


class CircleExample(Scene):
    def construct(self):
        # Create title
        title = Text("Manim Test - IR Animations", font_size=48)
        title.to_edge(UP)

        # Create a circle
        circle = Circle(radius=1, color=BLUE)

        # Create text
        text = Text("Installation Successful!", font_size=36)
        text.next_to(circle, DOWN)

        # Animations
        self.play(Write(title))
        self.play(Create(circle))
        self.play(Write(text))
        self.play(circle.animate.set_fill(BLUE, opacity=0.5))
        self.wait(2)


class SimpleBarChart(Scene):
    def construct(self):
        # Title
        title = Text("IR Performance Comparison", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        # Create bars
        bar_tfidf = Rectangle(height=2, width=0.5, color=RED, fill_opacity=0.7)
        bar_bm25 = Rectangle(height=3.5, width=0.5, color=BLUE, fill_opacity=0.7)
        bar_rocchio = Rectangle(height=3, width=0.5, color=GREEN, fill_opacity=0.7)

        # Position bars
        bars = VGroup(bar_tfidf, bar_bm25, bar_rocchio).arrange(RIGHT, buff=0.5)
        bars.move_to(ORIGIN)

        # Labels
        label_tfidf = Text("TF-IDF", font_size=24).next_to(bar_tfidf, DOWN)
        label_bm25 = Text("BM25", font_size=24).next_to(bar_bm25, DOWN)
        label_rocchio = Text("Rocchio", font_size=24).next_to(bar_rocchio, DOWN)

        # Animate
        self.play(GrowFromEdge(bar_tfidf, DOWN))
        self.play(Write(label_tfidf))
        self.play(GrowFromEdge(bar_bm25, DOWN))
        self.play(Write(label_bm25))
        self.play(GrowFromEdge(bar_rocchio, DOWN))
        self.play(Write(label_rocchio))

        self.wait(2)
