"""
Comprehensive Visualizations - All metrics across datasets
Beautiful animated comparisons for the Results section

Run: manim -pqh comprehensive_visualizations.py ComprehensiveComparison
"""

from manim import *


class ComprehensiveComparison(Scene):
    """Multi-metric comparison showing MAP, P@5, NDCG across all datasets"""

    def construct(self):
        # Title
        title = Text("Complete Performance Analysis", font_size=48, weight=BOLD)
        subtitle = Text("All Metrics • All Datasets • All Algorithms", font_size=28)
        subtitle.next_to(title, DOWN)

        title_group = VGroup(title, subtitle)
        title_group.to_edge(UP)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle))
        self.wait(0.5)

        # Data from actual results
        datasets = ["CISI\n1.4K", "MARCO\n10K", "MARCO\n50K", "MARCO\n100K"]
        metrics = ["MAP", "P@5", "NDCG@10"]

        # BM25, TF-IDF, Rocchio
        map_data = {
            "BM25": [0.205, 0.646, 0.751, 0.747],
            "TF-IDF": [0.196, 0.497, 0.508, 0.392],
            "Rocchio": [0.147, 0.610, 0.667, 0.626]
        }

        p5_data = {
            "BM25": [0.390, 0.162, 0.184, 0.174],
            "TF-IDF": [0.332, 0.132, 0.136, 0.102],
            "Rocchio": [0.276, 0.156, 0.170, 0.154]
        }

        ndcg_data = {
            "BM25": [0.367, 0.692, 0.791, 0.787],
            "TF-IDF": [0.314, 0.546, 0.556, 0.443],
            "Rocchio": [0.272, 0.667, 0.716, 0.664]
        }

        # Show each dataset
        for dataset_idx, dataset_name in enumerate(datasets):
            if dataset_idx > 0:
                self.play(*[FadeOut(mob) for mob in self.mobjects if mob not in [title, subtitle]])

            dataset_title = Text(dataset_name, font_size=40, color=YELLOW, weight=BOLD)
            dataset_title.next_to(subtitle, DOWN, buff=0.8)
            self.play(Write(dataset_title))

            # Create grouped bar chart for three metrics
            bar_groups = self.create_metric_comparison(
                dataset_idx,
                map_data,
                p5_data,
                ndcg_data
            )

            self.play(FadeIn(bar_groups), run_time=1.5)
            self.wait(2)

        # Final summary
        self.show_summary()

    def create_metric_comparison(self, dataset_idx, map_data, p5_data, ndcg_data):
        """Create grouped bars for all three metrics"""
        colors = {"BM25": BLUE, "TF-IDF": RED, "Rocchio": GREEN}

        # Get values for this dataset
        map_vals = [map_data[alg][dataset_idx] for alg in ["BM25", "TF-IDF", "Rocchio"]]
        p5_vals = [p5_data[alg][dataset_idx] for alg in ["BM25", "TF-IDF", "Rocchio"]]
        ndcg_vals = [ndcg_data[alg][dataset_idx] for alg in ["BM25", "TF-IDF", "Rocchio"]]

        all_groups = []

        for metric_idx, (metric_name, values) in enumerate([
            ("MAP", map_vals),
            ("P@5", p5_vals),
            ("NDCG@10", ndcg_vals)
        ]):
            # Create bars for this metric
            bars = []
            labels = []

            for i, (alg, val, color) in enumerate(zip(
                ["BM25", "TF-IDF", "Rocchio"],
                values,
                [BLUE, RED, GREEN]
            )):
                height = val * 3  # Scale
                bar = Rectangle(
                    height=height,
                    width=0.4,
                    color=color,
                    fill_opacity=0.8,
                    stroke_width=2
                )

                value_label = Text(f"{val:.2f}", font_size=18, color=color)
                value_label.next_to(bar, UP, buff=0.1)

                bars.append(bar)
                labels.append(value_label)

            # Arrange bars for this metric
            bar_group = VGroup(*bars).arrange(RIGHT, buff=0.2)
            label_group = VGroup(*labels)

            # Position labels
            for bar, label in zip(bars, labels):
                label.next_to(bar, UP, buff=0.1)

            # Metric title
            metric_title = Text(metric_name, font_size=24, weight=BOLD)
            metric_title.next_to(bar_group, DOWN, buff=0.3)

            full_group = VGroup(bar_group, label_group, metric_title)
            all_groups.append(full_group)

        # Arrange all metric groups
        final_group = VGroup(*all_groups).arrange(RIGHT, buff=1.5)
        final_group.shift(DOWN * 0.5)

        return final_group

    def show_summary(self):
        """Final summary with key findings"""
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        title = Text("Key Performance Insights", font_size=48, weight=BOLD, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))

        findings = [
            ("BM25: Consistent Winner", "75.1% MAP @ 50K docs", BLUE),
            ("TF-IDF: Degrades at Scale", "-23% from 50K to 100K", RED),
            ("Rocchio: Stable Middle Ground", "66.7% MAP @ 50K docs", GREEN),
        ]

        y_offset = 1.5
        for i, (main, detail, color) in enumerate(findings):
            main_text = Text(main, font_size=36, color=color, weight=BOLD)
            detail_text = Text(detail, font_size=28, color=WHITE)

            group = VGroup(main_text, detail_text).arrange(DOWN, buff=0.2)
            group.shift(UP * (y_offset - i * 1.5))

            checkmark = Text("✓" if i != 1 else "✗", font_size=60, color=color)
            checkmark.next_to(group, LEFT)

            self.play(
                FadeIn(checkmark),
                Write(main_text),
                run_time=1
            )
            self.play(Write(detail_text), run_time=0.8)
            self.wait(0.5)

        self.wait(2)


class MetricBreakdown(Scene):
    """Detailed breakdown of individual metrics"""

    def construct(self):
        title = Text("Metric Analysis: MAP vs P@5 vs NDCG", font_size=44, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Show what each metric measures
        metrics_info = [
            ("MAP", "Mean Average Precision", "Ranking quality across all results", BLUE),
            ("P@5", "Precision at 5", "Relevance of top 5 results", GREEN),
            ("NDCG@10", "Normalized DCG at 10", "Position-aware relevance", YELLOW),
        ]

        y_pos = 1.5
        for metric, full_name, description, color in metrics_info:
            # Metric abbreviation
            abbrev = Text(metric, font_size=48, color=color, weight=BOLD)
            abbrev.shift(LEFT * 4 + UP * y_pos)

            # Full name
            name = Text(full_name, font_size=28, color=WHITE)
            name.next_to(abbrev, RIGHT, buff=0.5)

            # Description
            desc = Text(description, font_size=20, color=GREY)
            desc.next_to(name, DOWN, aligned_edge=LEFT, buff=0.1)

            self.play(
                Write(abbrev),
                Write(name),
                run_time=1
            )
            self.play(FadeIn(desc), run_time=0.5)
            self.wait(0.5)

            y_pos -= 1.5

        self.wait(2)

        # Clear and show BM25 winning on all metrics
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

        conclusion = Text(
            "BM25 Dominates ALL Metrics",
            font_size=48,
            color=BLUE,
            weight=BOLD
        )
        conclusion.move_to(ORIGIN)

        self.play(Write(conclusion), run_time=2)
        self.wait(2)


class PerformanceHeatmap(Scene):
    """Animated heatmap showing all results"""

    def construct(self):
        title = Text("Performance Heatmap: All Results", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Datasets (rows)
        datasets = ["CISI", "10K", "50K", "100K"]
        algorithms = ["TF-IDF", "BM25", "Rocchio"]

        # MAP values for heatmap
        data = [
            [0.196, 0.205, 0.147],  # CISI
            [0.497, 0.646, 0.610],  # 10K
            [0.508, 0.751, 0.667],  # 50K
            [0.392, 0.747, 0.626],  # 100K
        ]

        # Create grid
        cell_width = 2
        cell_height = 1

        grid = VGroup()

        for row_idx, (dataset, row_data) in enumerate(zip(datasets, data)):
            for col_idx, (algorithm, value) in enumerate(zip(algorithms, row_data)):
                # Color intensity based on value
                if value > 0.7:
                    color = BLUE
                    opacity = 0.9
                elif value > 0.5:
                    color = GREEN
                    opacity = 0.7
                else:
                    color = RED
                    opacity = 0.5

                cell = Rectangle(
                    width=cell_width,
                    height=cell_height,
                    color=color,
                    fill_opacity=opacity,
                    stroke_width=2,
                    stroke_color=WHITE
                )

                value_text = Text(f"{value:.1%}", font_size=24, weight=BOLD, color=WHITE)
                value_text.move_to(cell.get_center())

                cell_group = VGroup(cell, value_text)
                cell_group.move_to([
                    (col_idx - 1) * cell_width,
                    (1.5 - row_idx) * cell_height,
                    0
                ])

                # Animate cell appearance
                self.play(
                    FadeIn(cell),
                    Write(value_text),
                    run_time=0.3
                )

                grid.add(cell_group)

        # Add labels
        row_labels = VGroup(*[
            Text(dataset, font_size=24)
            for dataset in datasets
        ])

        col_labels = VGroup(*[
            Text(alg, font_size=24, color=[RED, BLUE, GREEN][i])
            for i, alg in enumerate(algorithms)
        ])

        for i, label in enumerate(row_labels):
            label.next_to(grid[i * 3], LEFT, buff=0.5)

        for i, label in enumerate(col_labels):
            label.next_to(grid[i], UP, buff=0.5)

        self.play(Write(row_labels), Write(col_labels), run_time=1)

        # Highlight BM25 column
        bm25_highlight = Rectangle(
            width=cell_width + 0.2,
            height=cell_height * 4 + 0.3,
            color=YELLOW,
            stroke_width=6,
            fill_opacity=0
        )
        bm25_highlight.move_to([0, 0.25, 0])

        self.play(Create(bm25_highlight), run_time=1)

        winner_text = Text("WINNER!", font_size=32, color=YELLOW, weight=BOLD)
        winner_text.next_to(bm25_highlight, DOWN, buff=0.5)

        self.play(Write(winner_text), run_time=1)
        self.wait(3)
