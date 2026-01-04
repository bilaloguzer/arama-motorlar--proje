"""
TF-IDF Degradation Animation - Explains IDF compression phenomenon
Educational visualization showing why TF-IDF fails at scale

Run: manim -pqh tfidf_degradation.py IDFCompressionExplanation
"""

from manim import *


class IDFCompressionExplanation(Scene):
    def construct(self):
        # Title
        title = Text("Why Does TF-IDF Fail at Scale?", font_size=48, weight=BOLD)
        subtitle = Text("The IDF Compression Phenomenon", font_size=32, color=YELLOW)
        subtitle.next_to(title, DOWN)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle), run_time=1)
        self.wait(1)

        self.play(
            FadeOut(title),
            FadeOut(subtitle)
        )

        # Part 1: Show IDF formula
        self.show_idf_formula()
        self.wait(1)

        # Part 2: Demonstrate compression
        self.demonstrate_compression()
        self.wait(1)

        # Part 3: Show the impact
        self.show_impact()
        self.wait(2)

    def show_idf_formula(self):
        # IDF Formula
        formula_title = Text("IDF Formula:", font_size=36, color=BLUE)
        formula_title.to_edge(UP)

        formula = MathTex(
            r"\text{IDF}(t) = \log\left(\frac{N}{\text{df}(t)}\right)",
            font_size=60
        )
        formula.next_to(formula_title, DOWN, buff=0.5)

        # Explanation
        n_exp = Text("N = Total documents", font_size=28)
        df_exp = Text("df(t) = Documents containing term t", font_size=28)
        explanations = VGroup(n_exp, df_exp).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        explanations.next_to(formula, DOWN, buff=0.8)

        self.play(Write(formula_title))
        self.play(Write(formula), run_time=2)
        self.play(Write(explanations), run_time=1.5)
        self.wait(1)

        self.formula_group = VGroup(formula_title, formula, explanations)

    def demonstrate_compression(self):
        # Fade out formula
        self.play(FadeOut(self.formula_group))

        # Title
        title = Text("IDF Compression in Action", font_size=40, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Example term: "search"
        term_title = Text('Example Term: "search"', font_size=32, color=YELLOW)
        term_title.next_to(title, DOWN, buff=0.5)
        self.play(Write(term_title))

        # Create two scenarios side by side
        scenario_50k = self.create_scenario(
            "50K Documents",
            "N = 50,000",
            "df = 5,000",
            "IDF = log(10) = 2.30",
            BLUE,
            LEFT * 3.5
        )

        scenario_100k = self.create_scenario(
            "100K Documents",
            "N = 100,000",
            "df = 10,000",
            "IDF = log(10) = 2.30",
            RED,
            RIGHT * 3.5
        )

        # Animate scenarios
        self.play(FadeIn(scenario_50k), run_time=1.5)
        self.wait(0.5)
        self.play(FadeIn(scenario_100k), run_time=1.5)
        self.wait(1)

        # Highlight the problem
        problem_box = SurroundingRectangle(
            VGroup(
                scenario_50k[-1],  # IDF value
                scenario_100k[-1]   # IDF value
            ),
            color=YELLOW,
            buff=0.3,
            stroke_width=6
        )

        problem_text = Text(
            "Same IDF! No discrimination!",
            font_size=32,
            color=YELLOW,
            weight=BOLD
        )
        problem_text.to_edge(DOWN, buff=0.5)

        self.play(Create(problem_box), run_time=1)
        self.play(Write(problem_text), run_time=1.5)
        self.wait(2)

        # Fade out for next part
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

    def create_scenario(self, title, n_text, df_text, idf_text, color, position):
        title_obj = Text(title, font_size=28, color=color, weight=BOLD)
        n_obj = MathTex(n_text, font_size=24)
        df_obj = MathTex(df_text, font_size=24)
        idf_obj = MathTex(idf_text, font_size=28, color=color)

        box = Rectangle(
            height=3,
            width=3.5,
            color=color,
            stroke_width=3
        )

        group = VGroup(title_obj, n_obj, df_obj, idf_obj)
        group.arrange(DOWN, buff=0.3)
        group.move_to(position)

        box.move_to(group)

        return VGroup(box, title_obj, n_obj, df_obj, idf_obj)

    def show_impact(self):
        # Title
        title = Text("The Result: Performance Collapse", font_size=42, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Create simple bar chart
        bar_50k = Rectangle(height=3.5, width=1, color=BLUE, fill_opacity=0.8)
        bar_100k = Rectangle(height=2.5, width=1, color=RED, fill_opacity=0.8)

        bars = VGroup(bar_50k, bar_100k).arrange(RIGHT, buff=2)
        bars.move_to(ORIGIN)

        label_50k = Text("50K docs\n50.8% MAP", font_size=24, color=BLUE)
        label_100k = Text("100K docs\n39.2% MAP", font_size=24, color=RED)

        label_50k.next_to(bar_50k, DOWN)
        label_100k.next_to(bar_100k, DOWN)

        # Animate bars growing
        self.play(GrowFromEdge(bar_50k, DOWN), run_time=1)
        self.play(Write(label_50k))
        self.wait(0.5)

        self.play(GrowFromEdge(bar_100k, DOWN), run_time=1)
        self.play(Write(label_100k))
        self.wait(1)

        # Show degradation
        arrow = Arrow(
            start=bar_50k.get_top() + RIGHT * 0.5,
            end=bar_100k.get_top() + LEFT * 0.5,
            color=YELLOW,
            stroke_width=8
        )
        degradation_text = Text(
            "-23%",
            font_size=48,
            color=YELLOW,
            weight=BOLD
        )
        degradation_text.next_to(arrow, UP)

        self.play(
            GrowArrow(arrow),
            Write(degradation_text),
            run_time=1.5
        )
        self.wait(1)

        # Conclusion
        conclusion = Text(
            "TF-IDF unusable for large corpora (>50K docs)",
            font_size=32,
            color=RED,
            weight=BOLD
        )
        conclusion.to_edge(DOWN)

        self.play(Write(conclusion), run_time=2)
        self.wait(2)


class BM25Resistance(Scene):
    def construct(self):
        # Title
        title = Text("Why BM25 Doesn't Degrade", font_size=48, weight=BOLD, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Show three advantages
        advantages = [
            ("1. Robust IDF Formula", "Different smoothing prevents compression"),
            ("2. Term Saturation", "Limits impact of term frequency"),
            ("3. Length Normalization", "Compensates for IDF variations")
        ]

        for i, (main, sub) in enumerate(advantages):
            main_text = Text(main, font_size=36, color=BLUE, weight=BOLD)
            sub_text = Text(sub, font_size=28, color=WHITE)

            group = VGroup(main_text, sub_text).arrange(DOWN, buff=0.3)
            group.move_to(ORIGIN + UP * (1 - i * 1.5))

            checkmark = Text("✓", font_size=60, color=GREEN)
            checkmark.next_to(group, LEFT, buff=0.5)

            self.play(
                FadeIn(checkmark),
                Write(main_text),
                run_time=1
            )
            self.play(Write(sub_text), run_time=0.8)
            self.wait(0.5)

        self.wait(1)

        # Conclusion
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != title])

        conclusion_lines = [
            "Result:",
            "",
            "BM25 maintains 74-75% MAP",
            "from 50K to 100K documents",
            "",
            "(-0.5% change vs TF-IDF's -23%)"
        ]

        conclusion = VGroup(*[
            Text(
                line,
                font_size=32 if i == 0 else 28,
                weight=BOLD if i == 0 else NORMAL,
                color=BLUE if i in [2, 3] else WHITE
            )
            for i, line in enumerate(conclusion_lines)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        conclusion.move_to(ORIGIN)

        self.play(Write(conclusion), run_time=3)
        self.wait(3)
