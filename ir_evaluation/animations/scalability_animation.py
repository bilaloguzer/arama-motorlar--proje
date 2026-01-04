"""
Scalability Animation - Shows how algorithms perform across different corpus sizes
Demonstrates BM25 stability vs TF-IDF degradation

Run: manim -pqh scalability_animation.py ScalabilityEvolution
"""

from manim import *
import json


class ScalabilityEvolution(Scene):
    def construct(self):
        # Title
        title = Text("Algorithm Scalability Analysis", font_size=48, weight=BOLD)
        subtitle = Text("Performance across 1.4K → 100K documents", font_size=28)
        subtitle.next_to(title, DOWN)

        title_group = VGroup(title, subtitle)
        title_group.to_edge(UP)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle), run_time=1)
        self.wait(0.5)

        # Data (from your results)
        corpus_sizes = ["1.4K", "10K", "50K", "100K"]
        tfidf_map = [0.196, 0.497, 0.508, 0.392]
        bm25_map = [0.205, 0.646, 0.751, 0.747]
        rocchio_map = [0.147, 0.610, 0.667, 0.626]

        # Create axes
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 0.8, 0.2],
            x_length=10,
            y_length=5,
            axis_config={"color": GREY},
            tips=False,
        )

        # Labels
        x_labels = VGroup(*[
            Text(corpus_sizes[i], font_size=24).move_to(
                axes.c2p(i + 0.5, -0.05)
            )
            for i in range(4)
        ])

        y_label = Text("MAP Score", font_size=28)
        y_label.rotate(90 * DEGREES)
        y_label.next_to(axes, LEFT)

        axes_group = VGroup(axes, x_labels, y_label)
        axes_group.shift(DOWN * 0.5)

        self.play(Create(axes), Write(y_label))
        self.play(Write(x_labels))
        self.wait(0.5)

        # Create line graphs
        def create_line_graph(data, color, name):
            points = [axes.c2p(i + 0.5, data[i]) for i in range(4)]
            line = VMobject(color=color, stroke_width=6)
            line.set_points_smoothly(points)

            dots = VGroup(*[
                Dot(point, color=color, radius=0.1) for point in points
            ])

            # Value labels
            value_labels = VGroup(*[
                Text(f"{data[i]:.1%}", font_size=20, color=color).next_to(
                    dots[i], UP, buff=0.15
                )
                for i in range(4)
            ])

            return line, dots, value_labels

        # BM25 (Blue) - Winner
        bm25_line, bm25_dots, bm25_labels = create_line_graph(bm25_map, BLUE, "BM25")

        # TF-IDF (Red) - Degrading
        tfidf_line, tfidf_dots, tfidf_labels = create_line_graph(tfidf_map, RED, "TF-IDF")

        # Rocchio (Green) - Middle
        rocchio_line, rocchio_dots, rocchio_labels = create_line_graph(
            rocchio_map, GREEN, "Rocchio"
        )

        # Legend
        legend_items = VGroup(
            self.create_legend_item("BM25 (Best)", BLUE),
            self.create_legend_item("TF-IDF (Degrades)", RED),
            self.create_legend_item("Rocchio", GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        legend_items.to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)

        # Animate appearance
        self.play(Write(legend_items[0]))
        self.play(Create(bm25_line), run_time=2)
        self.play(
            *[GrowFromCenter(dot) for dot in bm25_dots],
            run_time=1
        )
        self.play(Write(bm25_labels), run_time=1)
        self.wait(0.5)

        self.play(Write(legend_items[1]))
        self.play(Create(tfidf_line), run_time=2)
        self.play(
            *[GrowFromCenter(dot) for dot in tfidf_dots],
            run_time=1
        )
        self.play(Write(tfidf_labels), run_time=1)
        self.wait(0.5)

        self.play(Write(legend_items[2]))
        self.play(Create(rocchio_line), run_time=2)
        self.play(
            *[GrowFromCenter(dot) for dot in rocchio_dots],
            run_time=1
        )
        self.play(Write(rocchio_labels), run_time=1)

        self.wait(1)

        # Highlight TF-IDF degradation
        degradation_arrow = Arrow(
            start=tfidf_dots[2].get_center() + UP * 0.5,
            end=tfidf_dots[3].get_center() + DOWN * 0.5,
            color=YELLOW,
            stroke_width=8,
        )
        degradation_text = Text(
            "-23% Degradation!",
            font_size=28,
            color=YELLOW,
            weight=BOLD
        )
        degradation_text.next_to(degradation_arrow, RIGHT, buff=0.3)

        self.play(
            GrowArrow(degradation_arrow),
            Write(degradation_text),
            run_time=1.5
        )
        self.wait(0.5)

        # Highlight BM25 stability
        stability_box = SurroundingRectangle(
            VGroup(bm25_dots[2], bm25_dots[3]),
            color=BLUE,
            buff=0.3,
            corner_radius=0.2
        )
        stability_text = Text(
            "Stable at 75%",
            font_size=28,
            color=BLUE,
            weight=BOLD
        )
        stability_text.next_to(stability_box, LEFT, buff=0.5)

        self.play(
            Create(stability_box),
            Write(stability_text),
            run_time=1.5
        )

        self.wait(3)

        # Conclusion
        conclusion = Text(
            "BM25: Superior and Stable Across All Scales",
            font_size=36,
            color=BLUE,
            weight=BOLD
        )
        conclusion.to_edge(DOWN)

        self.play(Write(conclusion), run_time=2)
        self.wait(2)

    def create_legend_item(self, label, color):
        line = Line(ORIGIN, RIGHT * 0.5, color=color, stroke_width=6)
        dot = Dot(color=color, radius=0.08)
        text = Text(label, font_size=24, color=WHITE)

        group = VGroup(line, dot, text)
        dot.move_to(line.get_center())
        text.next_to(line, RIGHT, buff=0.2)

        return group


class ScalabilityBarChart(Scene):
    def construct(self):
        # Title
        title = Text("MAP Performance by Corpus Size", font_size=42, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Data
        corpus_sizes = ["1.4K\nCISI", "10K\nMS MARCO", "50K\nMS MARCO", "100K\nMS MARCO"]
        algorithms = ["TF-IDF", "BM25", "Rocchio"]
        colors = [RED, BLUE, GREEN]

        data = {
            "TF-IDF": [0.196, 0.497, 0.508, 0.392],
            "BM25": [0.205, 0.646, 0.751, 0.747],
            "Rocchio": [0.147, 0.610, 0.667, 0.626]
        }

        # Create bar chart for each corpus size
        for size_idx in range(4):
            if size_idx > 0:
                # Fade out previous chart
                self.play(*[FadeOut(mob) for mob in self.mobjects if mob != title])

            size_label = Text(corpus_sizes[size_idx], font_size=36)
            size_label.next_to(title, DOWN, buff=0.5)
            self.play(Write(size_label))

            # Create bars for this size
            max_height = 4
            bars = []
            labels = []
            values = []

            for alg_idx, alg in enumerate(algorithms):
                height = data[alg][size_idx] * 5  # Scale to fit screen
                bar = Rectangle(
                    height=height,
                    width=1.5,
                    color=colors[alg_idx],
                    fill_opacity=0.8,
                    stroke_width=3
                )

                # Label
                label = Text(alg, font_size=28)

                # Value
                value = Text(
                    f"{data[alg][size_idx]:.1%}",
                    font_size=32,
                    weight=BOLD,
                    color=colors[alg_idx]
                )

                bars.append(bar)
                labels.append(label)
                values.append(value)

            # Arrange bars
            bar_group = VGroup(*bars).arrange(RIGHT, buff=0.8)
            bar_group.move_to(ORIGIN)

            # Position labels and values
            for i, (bar, label, value) in enumerate(zip(bars, labels, values)):
                label.next_to(bar, DOWN)
                value.next_to(bar, UP)

            # Animate
            for bar, label, value in zip(bars, labels, values):
                self.play(
                    GrowFromEdge(bar, DOWN),
                    run_time=0.8
                )
                self.play(
                    Write(label),
                    Write(value),
                    run_time=0.5
                )

            # Highlight winner
            winner_idx = 1  # BM25
            winner_rect = SurroundingRectangle(
                bars[winner_idx],
                color=YELLOW,
                buff=0.2,
                stroke_width=6
            )
            winner_text = Text("Winner!", font_size=28, color=YELLOW, weight=BOLD)
            winner_text.next_to(bars[winner_idx], RIGHT, buff=0.3)

            self.play(
                Create(winner_rect),
                Write(winner_text),
                run_time=1
            )

            self.wait(2)

        # Final summary
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != title])

        summary_lines = [
            "Key Findings:",
            "",
            "✓ BM25: 75.1% MAP at 50K docs",
            "✗ TF-IDF: -23% degradation at 100K",
            "~ Rocchio: Stable but behind BM25"
        ]

        summary_text = VGroup(*[
            Text(line, font_size=32 if i == 0 else 28, weight=BOLD if i == 0 else NORMAL)
            for i, line in enumerate(summary_lines)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        summary_text.move_to(ORIGIN)

        self.play(Write(summary_text), run_time=3)
        self.wait(3)
