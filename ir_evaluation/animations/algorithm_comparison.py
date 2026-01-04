"""
Algorithm Comparison Animation - Visual explanation of how each algorithm works
Educational 3Blue1Brown-style visualization

Run: manim -pqh algorithm_comparison.py AlgorithmShowdown
"""

from manim import *


class AlgorithmShowdown(Scene):
    def construct(self):
        # Title
        title = Text("IR Algorithms: Head-to-Head", font_size=52, weight=BOLD)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Show all three algorithms
        self.show_tfidf()
        self.wait(1)
        self.show_bm25()
        self.wait(1)
        self.show_rocchio()
        self.wait(1)
        self.show_comparison()
        self.wait(2)

    def show_tfidf(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

        # TF-IDF Card
        card = self.create_algorithm_card(
            "TF-IDF (1975)",
            RED,
            [
                "Vector Space Model",
                "Cosine Similarity",
                "Simple & Fast"
            ],
            [
                "✗ No length normalization",
                "✗ IDF degrades at scale",
                "✗ Linear term weighting"
            ],
            "50.8% MAP @ 50K"
        )
        card.move_to(ORIGIN)

        self.play(FadeIn(card), run_time=1.5)
        self.wait(2)

    def show_bm25(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

        # BM25 Card
        card = self.create_algorithm_card(
            "BM25 (1994)",
            BLUE,
            [
                "Probabilistic Model",
                "Term Saturation (k1=1.5)",
                "Length Normalization (b=0.75)"
            ],
            [
                "✓ Handles long documents",
                "✓ Stable across scales",
                "✓ Industry standard"
            ],
            "75.1% MAP @ 50K 🏆"
        )
        card.move_to(ORIGIN)

        self.play(FadeIn(card), run_time=1.5)

        # Add winner badge
        badge = Text("WINNER", font_size=48, color=YELLOW, weight=BOLD)
        badge.to_edge(UP).shift(DOWN * 1.5)
        star = Text("⭐", font_size=60)
        star.next_to(badge, LEFT)

        self.play(
            Write(badge),
            Write(star),
            run_time=1
        )
        self.wait(2)

    def show_rocchio(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

        # Rocchio Card
        card = self.create_algorithm_card(
            "Rocchio (1971)",
            GREEN,
            [
                "Relevance Feedback",
                "Query Refinement",
                "Centroid-based"
            ],
            [
                "~ Requires user feedback",
                "~ Potential query drift",
                "~ Behind BM25"
            ],
            "66.7% MAP @ 50K"
        )
        card.move_to(ORIGIN)

        self.play(FadeIn(card), run_time=1.5)
        self.wait(2)

    def create_algorithm_card(self, name, color, strengths, details, performance):
        # Card background
        card = Rectangle(
            height=5,
            width=10,
            color=color,
            fill_opacity=0.1,
            stroke_width=4
        )

        # Name
        name_text = Text(name, font_size=40, color=color, weight=BOLD)
        name_text.move_to(card.get_top() + DOWN * 0.5)

        # Strengths
        strength_items = VGroup(*[
            Text(f"• {s}", font_size=24, color=WHITE)
            for s in strengths
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        strength_items.next_to(name_text, DOWN, buff=0.5)

        # Details
        detail_items = VGroup(*[
            Text(d, font_size=22, color=GREY)
            for d in details
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        detail_items.next_to(strength_items, DOWN, buff=0.4)

        # Performance
        perf_text = Text(performance, font_size=32, color=color, weight=BOLD)
        perf_text.move_to(card.get_bottom() + UP * 0.5)

        return VGroup(card, name_text, strength_items, detail_items, perf_text)

    def show_comparison(self):
        # Clear all
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

        # Final comparison table
        comparison_title = Text(
            "Performance Summary (MS MARCO 50K)",
            font_size=36,
            weight=BOLD
        )
        comparison_title.to_edge(UP).shift(DOWN * 0.5)
        self.play(Write(comparison_title))

        # Create bars
        algorithms = ["TF-IDF", "BM25", "Rocchio"]
        maps = [0.508, 0.751, 0.667]
        colors = [RED, BLUE, GREEN]

        bars = []
        labels = []
        values = []

        for i, (alg, map_val, color) in enumerate(zip(algorithms, maps, colors)):
            # Bar
            bar = Rectangle(
                height=map_val * 4,
                width=1.5,
                color=color,
                fill_opacity=0.8,
                stroke_width=3
            )

            # Label
            label = Text(alg, font_size=28, color=WHITE)

            # Value
            value = Text(
                f"{map_val:.1%}",
                font_size=32,
                color=color,
                weight=BOLD
            )

            bars.append(bar)
            labels.append(label)
            values.append(value)

        # Arrange
        bar_group = VGroup(*bars).arrange(RIGHT, buff=1.2)
        bar_group.move_to(ORIGIN + DOWN * 0.5)

        for bar, label, value in zip(bars, labels, values):
            label.next_to(bar, DOWN, buff=0.3)
            value.next_to(bar, UP, buff=0.3)

        # Animate bars
        for bar, label, value in zip(bars, labels, values):
            self.play(GrowFromEdge(bar, DOWN), run_time=1)
            self.play(Write(label), Write(value), run_time=0.5)

        self.wait(1)

        # Highlight winner
        winner_box = SurroundingRectangle(
            bars[1],  # BM25
            color=YELLOW,
            buff=0.3,
            stroke_width=8
        )
        winner_badge = Text("BEST", font_size=36, color=YELLOW, weight=BOLD)
        winner_badge.next_to(bars[1], RIGHT, buff=0.5)

        self.play(
            Create(winner_box),
            Write(winner_badge),
            run_time=1.5
        )

        self.wait(1)

        # Add advantage text
        advantage = Text(
            "+48% better than TF-IDF",
            font_size=28,
            color=YELLOW
        )
        advantage.next_to(winner_badge, DOWN, buff=0.2)

        self.play(Write(advantage), run_time=1)
        self.wait(2)


class StemmingImpact(Scene):
    def construct(self):
        # Title
        title = Text("Impact of Stemming", font_size=48, weight=BOLD)
        subtitle = Text("Porter Stemmer Performance Boost", font_size=32, color=GREEN)
        subtitle.next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(1)

        self.play(
            title.animate.to_edge(UP),
            FadeOut(subtitle)
        )

        # Show example
        example_title = Text("Example: Term Conflation", font_size=32, color=YELLOW)
        example_title.next_to(title, DOWN, buff=0.5)
        self.play(Write(example_title))

        # Original terms
        original = VGroup(
            Text('"retrieval"', font_size=28, color=RED),
            Text('"retrieve"', font_size=28, color=RED),
            Text('"retrieved"', font_size=28, color=RED),
            Text('"retrieves"', font_size=28, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        original.shift(LEFT * 3)

        original_label = Text("Original Terms", font_size=24, weight=BOLD)
        original_label.next_to(original, UP, buff=0.3)

        self.play(
            Write(original_label),
            Write(original),
            run_time=2
        )
        self.wait(1)

        # Arrow
        arrow = Arrow(
            start=original.get_right() + RIGHT * 0.3,
            end=original.get_right() + RIGHT * 2.5,
            color=GREEN,
            stroke_width=8
        )
        arrow_label = Text("Stemming", font_size=24, color=GREEN)
        arrow_label.next_to(arrow, UP)

        self.play(
            GrowArrow(arrow),
            Write(arrow_label),
            run_time=1
        )

        # Stemmed term
        stemmed = Text('"retriev"', font_size=36, color=GREEN, weight=BOLD)
        stemmed.shift(RIGHT * 3)

        stemmed_label = Text("Stemmed Form", font_size=24, weight=BOLD, color=GREEN)
        stemmed_label.next_to(stemmed, UP, buff=0.3)

        self.play(
            Write(stemmed_label),
            Write(stemmed),
            run_time=1
        )
        self.wait(2)

        # Clear for results
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

        # Show improvements
        improvements_title = Text(
            "MAP Improvements with Stemming",
            font_size=36,
            weight=BOLD
        )
        improvements_title.to_edge(UP).shift(DOWN * 0.5)
        self.play(Write(improvements_title))

        # Data
        algorithms = ["TF-IDF", "BM25", "Rocchio"]
        improvements = [7.5, 9.8, 7.6]
        colors = [RED, BLUE, GREEN]

        y_pos = 1
        for alg, improvement, color in zip(algorithms, improvements, colors):
            # Algorithm name
            alg_text = Text(alg, font_size=32, color=color, weight=BOLD)
            alg_text.shift(LEFT * 4 + UP * y_pos)

            # Arrow
            arrow_length = improvement / 10 * 4  # Scale
            arrow = Arrow(
                start=ORIGIN + UP * y_pos,
                end=RIGHT * arrow_length + UP * y_pos,
                color=color,
                stroke_width=8,
                buff=0
            )

            # Value
            value = Text(
                f"+{improvement}%",
                font_size=32,
                color=color,
                weight=BOLD
            )
            value.next_to(arrow, RIGHT, buff=0.3)

            self.play(
                Write(alg_text),
                GrowArrow(arrow),
                Write(value),
                run_time=1
            )
            self.wait(0.5)

            y_pos -= 1.5

        self.wait(1)

        # Highlight BM25 as biggest beneficiary
        highlight = Text(
            "BM25 benefits MOST from stemming!",
            font_size=28,
            color=YELLOW,
            weight=BOLD
        )
        highlight.to_edge(DOWN)

        self.play(Write(highlight), run_time=1.5)
        self.wait(2)
