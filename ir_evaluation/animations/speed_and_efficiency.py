"""
Speed and Efficiency Animations - Computational performance analysis
Shows processing time, throughput, and scalability

Run: manim -pqh speed_and_efficiency.py SpeedAnalysis
"""

from manim import *


class SpeedAnalysis(Scene):
    """Processing time and throughput visualization"""

    def construct(self):
        title = Text("Computational Performance Analysis", font_size=48, weight=BOLD)
        subtitle = Text("Apple M1 Pro | 16GB RAM", font_size=28, color=GREY)
        subtitle.next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(0.5)

        self.play(
            title.animate.to_edge(UP),
            FadeOut(subtitle)
        )

        # Processing time data (from actual results)
        corpus_sizes = ["10K", "50K", "100K"]
        times = {
            "TF-IDF": [1.29, 7.05, 14.50],
            "BM25": [1.11, 5.63, 11.69],
            "Rocchio": [0.77, 3.91, 7.44]
        }
        colors = {"TF-IDF": RED, "BM25": BLUE, "Rocchio": GREEN}

        # Create bar chart for processing times
        bar_width = 0.8
        max_height = 4

        for size_idx, size in enumerate(corpus_sizes):
            if size_idx > 0:
                self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

            size_title = Text(f"{size} Documents", font_size=40, color=YELLOW)
            size_title.next_to(title, DOWN, buff=0.5)
            self.play(Write(size_title))

            bars = []
            labels = []

            for alg_idx, (alg, color) in enumerate([
                ("TF-IDF", RED),
                ("BM25", BLUE),
                ("Rocchio", GREEN)
            ]):
                time_val = times[alg][size_idx]
                height = (time_val / 15) * max_height  # Normalize

                bar = Rectangle(
                    height=height,
                    width=bar_width,
                    color=color,
                    fill_opacity=0.8,
                    stroke_width=3
                )

                # Algorithm label
                alg_label = Text(alg, font_size=28, color=color)

                # Time label
                time_label = Text(
                    f"{time_val:.2f}s",
                    font_size=24,
                    color=color,
                    weight=BOLD
                )

                bars.append(bar)
                labels.append((alg_label, time_label))

            # Arrange bars
            bar_group = VGroup(*bars).arrange(RIGHT, buff=1)
            bar_group.move_to(ORIGIN)

            # Position labels
            for bar, (alg_label, time_label) in zip(bars, labels):
                alg_label.next_to(bar, DOWN, buff=0.3)
                time_label.next_to(bar, UP, buff=0.3)

            # Animate
            for bar, (alg_label, time_label) in zip(bars, labels):
                self.play(GrowFromEdge(bar, DOWN), run_time=0.8)
                self.play(Write(alg_label), Write(time_label), run_time=0.5)

            # Highlight fastest (Rocchio)
            fastest_box = SurroundingRectangle(
                bars[2],
                color=YELLOW,
                buff=0.2,
                stroke_width=6
            )
            fastest_text = Text("Fastest", font_size=28, color=YELLOW, weight=BOLD)
            fastest_text.next_to(bars[2], RIGHT, buff=0.5)

            self.play(Create(fastest_box), Write(fastest_text), run_time=1)
            self.wait(1.5)

        # Show throughput
        self.show_throughput()

    def show_throughput(self):
        """Show documents processed per second"""
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

        throughput_title = Text(
            "Throughput: Documents/Second",
            font_size=42,
            weight=BOLD
        )
        throughput_title.to_edge(UP).shift(DOWN * 0.5)
        self.play(Write(throughput_title))

        # Calculate average throughput
        avg_throughput = 2973

        # Animated counter
        counter = Integer(0, font_size=120, color=BLUE)
        counter.move_to(ORIGIN)

        docs_label = Text("docs/sec", font_size=36, color=WHITE)
        docs_label.next_to(counter, DOWN, buff=0.5)

        self.play(Write(docs_label))
        self.play(
            ChangeDecimalToValue(counter, avg_throughput),
            run_time=2,
            rate_func=linear
        )

        self.wait(1)

        # Add checkmark
        check = Text("✓", font_size=100, color=GREEN)
        check.next_to(counter, RIGHT, buff=0.5)

        sufficient_text = Text(
            "Sufficient for Academic Use",
            font_size=32,
            color=GREEN
        )
        sufficient_text.to_edge(DOWN)

        self.play(Write(check), run_time=0.5)
        self.play(Write(sufficient_text), run_time=1)
        self.wait(2)


class LinearScaling(Scene):
    """Demonstrates linear O(n) scaling"""

    def construct(self):
        title = Text("Linear Scalability: O(n)", font_size=48, weight=BOLD, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))

        # Create axes
        axes = Axes(
            x_range=[0, 100, 25],
            y_range=[0, 15, 5],
            x_length=10,
            y_length=5,
            axis_config={"color": GREY},
            tips=False,
        )

        x_label = Text("Documents (thousands)", font_size=24)
        x_label.next_to(axes, DOWN)

        y_label = Text("Time (seconds)", font_size=24)
        y_label.rotate(90 * DEGREES)
        y_label.next_to(axes, LEFT)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Actual data points
        data_points = [
            (10, 1.11),
            (50, 5.63),
            (100, 11.69)
        ]

        # Create line
        line = axes.plot_line_graph(
            x_values=[10, 50, 100],
            y_values=[1.11, 5.63, 11.69],
            line_color=BLUE,
            vertex_dot_radius=0.1,
            stroke_width=4
        )

        self.play(Create(line), run_time=2)

        # Show formula
        formula = MathTex(
            r"\text{Time} \propto n",
            font_size=48,
            color=BLUE
        )
        formula.to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)

        self.play(Write(formula))

        # Add annotation
        annotation = Text(
            "5× data = 5× time",
            font_size=32,
            color=YELLOW,
            weight=BOLD
        )
        annotation.next_to(formula, DOWN, buff=0.5)

        self.play(Write(annotation), run_time=1)
        self.wait(3)


class MemoryUsage(Scene):
    """Show memory requirements"""

    def construct(self):
        title = Text("Memory Requirements", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Memory data
        memory_data = [
            ("10K docs", "800 MB", 0.8, BLUE),
            ("50K docs", "2.1 GB", 2.1, GREEN),
            ("100K docs", "3.8 GB", 3.8, YELLOW),
            ("Projection: 500K", "~15 GB", 15, RED),
        ]

        # Create memory bars
        y_pos = 2
        for label, mem_text, mem_val, color in memory_data:
            # Label
            label_obj = Text(label, font_size=28, color=WHITE)
            label_obj.shift(LEFT * 4 + UP * y_pos)

            # Bar (scaled)
            bar_length = (mem_val / 16) * 6  # Scale to max 16GB available
            bar = Rectangle(
                width=bar_length,
                height=0.4,
                color=color,
                fill_opacity=0.7,
                stroke_width=2
            )
            bar.next_to(label_obj, RIGHT, buff=0.5)

            # Value
            value = Text(mem_text, font_size=24, color=color, weight=BOLD)
            value.next_to(bar, RIGHT, buff=0.2)

            self.play(
                Write(label_obj),
                GrowFromEdge(bar, LEFT),
                run_time=0.8
            )
            self.play(Write(value), run_time=0.4)

            y_pos -= 1

        self.wait(1)

        # Add system specs
        specs = Text(
            "M1 Pro: 16GB Unified Memory",
            font_size=28,
            color=GREY
        )
        specs.to_edge(DOWN)

        checkmark = Text("✓ Sufficient", font_size=32, color=GREEN, weight=BOLD)
        checkmark.next_to(specs, UP, buff=0.3)

        self.play(Write(specs), Write(checkmark), run_time=1)
        self.wait(2)


class EfficiencyComparison(Scene):
    """Compare all three algorithms on efficiency"""

    def construct(self):
        title = Text("Speed vs Accuracy Trade-off", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Create 2D plot: Speed (x) vs Accuracy (y)
        axes = Axes(
            x_range=[0, 8, 2],
            y_range=[0, 0.8, 0.2],
            x_length=8,
            y_length=5,
            axis_config={"color": GREY},
            tips=True,
        )
        axes.shift(DOWN * 0.5)

        x_label = Text("Speed (faster →)", font_size=24)
        x_label.next_to(axes, DOWN)

        y_label = Text("Accuracy (MAP)", font_size=24)
        y_label.rotate(90 * DEGREES)
        y_label.next_to(axes, LEFT)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Data points (speed inverted so faster = higher x)
        # 50K dataset: time and MAP
        algorithms = [
            ("Rocchio", 3.91, 0.667, GREEN, "Fast but less accurate"),
            ("BM25", 5.63, 0.751, BLUE, "Best accuracy, good speed"),
            ("TF-IDF", 7.05, 0.508, RED, "Slow AND inaccurate"),
        ]

        for alg, time_val, map_val, color, description in algorithms:
            # Invert time for x-axis (8 - time_val)
            x_pos = 8 - time_val
            y_pos = map_val

            point = Dot(axes.c2p(x_pos, y_pos), color=color, radius=0.15)

            label = Text(alg, font_size=24, color=color, weight=BOLD)
            label.next_to(point, UP + RIGHT, buff=0.2)

            self.play(GrowFromCenter(point), run_time=0.5)
            self.play(Write(label), run_time=0.5)
            self.wait(0.5)

        # Highlight BM25 as optimal
        bm25_point = Dot(axes.c2p(8 - 5.63, 0.751), color=BLUE, radius=0.15)
        optimal_circle = Circle(radius=0.5, color=YELLOW, stroke_width=6)
        optimal_circle.move_to(bm25_point)

        optimal_text = Text(
            "Optimal Balance",
            font_size=32,
            color=YELLOW,
            weight=BOLD
        )
        optimal_text.to_edge(DOWN)

        self.play(Create(optimal_circle), run_time=1)
        self.play(Write(optimal_text), run_time=1)
        self.wait(3)
