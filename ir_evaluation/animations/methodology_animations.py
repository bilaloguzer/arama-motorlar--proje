"""
Methodology Animations - Visual explanations for the report
Preprocessing pipeline, evaluation metrics, experimental design

Run: manim -pqh methodology_animations.py PreprocessingPipeline
"""

from manim import *


class PreprocessingPipeline(Scene):
    """Show the text preprocessing pipeline step by step"""

    def construct(self):
        title = Text("Text Preprocessing Pipeline", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Original text
        original = Text(
            '"Information Retrieval Systems"',
            font_size=32,
            color=WHITE
        )
        original.shift(UP * 2)

        original_label = Text("Original Query", font_size=24, color=GREY)
        original_label.next_to(original, UP, buff=0.3)

        self.play(Write(original_label), Write(original))
        self.wait(1)

        # Step 1: Tokenization
        step1_arrow = Arrow(original.get_bottom(), original.get_bottom() + DOWN * 0.8, color=BLUE)
        step1_label = Text("1. Tokenization", font_size=28, color=BLUE)
        step1_label.next_to(step1_arrow, LEFT, buff=0.5)

        tokens_text = Text(
            '["Information", "Retrieval", "Systems"]',
            font_size=28,
            color=BLUE
        )
        tokens_text.next_to(step1_arrow, DOWN, buff=0.3)

        self.play(
            GrowArrow(step1_arrow),
            Write(step1_label),
            run_time=1
        )
        self.play(Write(tokens_text), run_time=1)
        self.wait(0.5)

        # Step 2: Normalization
        step2_arrow = Arrow(tokens_text.get_bottom(), tokens_text.get_bottom() + DOWN * 0.8, color=GREEN)
        step2_label = Text("2. Lowercase", font_size=28, color=GREEN)
        step2_label.next_to(step2_arrow, LEFT, buff=0.5)

        normalized_text = Text(
            '["information", "retrieval", "systems"]',
            font_size=28,
            color=GREEN
        )
        normalized_text.next_to(step2_arrow, DOWN, buff=0.3)

        self.play(
            GrowArrow(step2_arrow),
            Write(step2_label),
            run_time=1
        )
        self.play(Write(normalized_text), run_time=1)
        self.wait(0.5)

        # Step 3: Stopword Removal (systems is not a stopword, so all kept)
        step3_arrow = Arrow(normalized_text.get_bottom(), normalized_text.get_bottom() + DOWN * 0.8, color=YELLOW)
        step3_label = Text("3. Stopword Filter", font_size=28, color=YELLOW)
        step3_label.next_to(step3_arrow, LEFT, buff=0.5)

        filtered_text = Text(
            '["information", "retrieval", "systems"]',
            font_size=28,
            color=YELLOW
        )
        filtered_text.next_to(step3_arrow, DOWN, buff=0.3)

        self.play(
            GrowArrow(step3_arrow),
            Write(step3_label),
            run_time=1
        )
        self.play(Write(filtered_text), run_time=1)
        self.wait(0.5)

        # Step 4: Stemming
        step4_arrow = Arrow(filtered_text.get_bottom(), filtered_text.get_bottom() + DOWN * 0.8, color=RED)
        step4_label = Text("4. Porter Stemmer", font_size=28, color=RED)
        step4_label.next_to(step4_arrow, LEFT, buff=0.5)

        stemmed_text = Text(
            '["inform", "retriev", "system"]',
            font_size=32,
            color=RED,
            weight=BOLD
        )
        stemmed_text.next_to(step4_arrow, DOWN, buff=0.3)

        final_label = Text("Final Processed Query", font_size=24, color=RED, weight=BOLD)
        final_label.next_to(stemmed_text, DOWN, buff=0.3)

        self.play(
            GrowArrow(step4_arrow),
            Write(step4_label),
            run_time=1
        )
        self.play(Write(stemmed_text), run_time=1)
        self.play(Write(final_label), run_time=0.8)

        self.wait(3)


class EvaluationMetrics(Scene):
    """Explain what each evaluation metric measures"""

    def construct(self):
        title = Text("Evaluation Metrics Explained", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Show MAP
        self.explain_map()
        self.wait(1)

        # Show Precision@k
        self.explain_precision_at_k()
        self.wait(1)

        # Show NDCG
        self.explain_ndcg()
        self.wait(2)

    def explain_map(self):
        """Explain Mean Average Precision"""
        map_title = Text("MAP: Mean Average Precision", font_size=36, color=BLUE, weight=BOLD)
        map_title.shift(UP * 2.5)
        self.play(Write(map_title))

        # Show example ranking
        results = [
            ("Doc 1", True, GREEN),
            ("Doc 2", False, RED),
            ("Doc 3", True, GREEN),
            ("Doc 4", True, GREEN),
            ("Doc 5", False, RED),
        ]

        result_group = VGroup()
        for i, (doc, relevant, color) in enumerate(results):
            rank_num = Text(f"{i+1}.", font_size=24, color=WHITE)
            doc_text = Text(doc, font_size=24, color=color)
            rel_icon = Text("✓" if relevant else "✗", font_size=28, color=color)

            item = VGroup(rank_num, doc_text, rel_icon).arrange(RIGHT, buff=0.3)
            result_group.add(item)

        result_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        result_group.shift(UP * 0.5 + LEFT * 2)

        self.play(Write(result_group), run_time=2)

        # Calculate precisions
        precisions = [
            "P@1 = 1/1 = 1.00",
            "P@3 = 2/3 = 0.67",
            "P@4 = 3/4 = 0.75",
        ]

        calc_group = VGroup()
        for prec in precisions:
            calc_text = Text(prec, font_size=20, color=YELLOW)
            calc_group.add(calc_text)

        calc_group.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        calc_group.next_to(result_group, RIGHT, buff=1)

        self.play(Write(calc_group), run_time=2)

        # Final AP
        ap_calc = Text("AP = (1.00 + 0.67 + 0.75) / 3 = 0.81", font_size=24, color=BLUE, weight=BOLD)
        ap_calc.to_edge(DOWN).shift(UP * 0.5)

        self.play(Write(ap_calc), run_time=1.5)
        self.wait(2)

        # Clear
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

    def explain_precision_at_k(self):
        """Explain Precision at k"""
        pk_title = Text("P@5: Precision at 5", font_size=36, color=GREEN, weight=BOLD)
        pk_title.shift(UP * 2)
        self.play(Write(pk_title))

        explanation = Text(
            "What percentage of top 5 results are relevant?",
            font_size=28,
            color=WHITE
        )
        explanation.next_to(pk_title, DOWN, buff=0.5)
        self.play(Write(explanation))

        # Show top 5 results
        top5 = VGroup()
        relevance = [True, False, True, True, False]

        for i, rel in enumerate(relevance):
            box = Rectangle(
                width=1.5,
                height=1.5,
                color=GREEN if rel else RED,
                fill_opacity=0.7,
                stroke_width=3
            )
            num = Text(str(i+1), font_size=32, color=WHITE, weight=BOLD)
            num.move_to(box.get_center())

            icon = Text("✓" if rel else "✗", font_size=28, color=WHITE)
            icon.next_to(box, DOWN, buff=0.2)

            item = VGroup(box, num, icon)
            top5.add(item)

        top5.arrange(RIGHT, buff=0.5)
        top5.move_to(ORIGIN)

        self.play(FadeIn(top5), run_time=1.5)

        # Calculate
        calc = Text("P@5 = 3/5 = 60%", font_size=32, color=GREEN, weight=BOLD)
        calc.to_edge(DOWN).shift(UP * 0.5)

        self.play(Write(calc), run_time=1)
        self.wait(2)

        # Clear
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])

    def explain_ndcg(self):
        """Explain NDCG"""
        ndcg_title = Text("NDCG: Position-Aware Metric", font_size=36, color=YELLOW, weight=BOLD)
        ndcg_title.to_edge(UP).shift(DOWN * 0.5)
        self.play(Write(ndcg_title))

        # Key idea
        idea = Text(
            "Higher positions count MORE",
            font_size=32,
            color=YELLOW
        )
        idea.next_to(ndcg_title, DOWN, buff=0.5)
        self.play(Write(idea))

        # Show discount
        positions = [1, 2, 3, 4, 5]
        discounts = [1.0, 0.63, 0.5, 0.43, 0.39]

        discount_group = VGroup()
        for pos, disc in zip(positions, discounts):
            pos_text = Text(f"Pos {pos}", font_size=24, color=WHITE)
            disc_text = Text(f"Weight: {disc:.2f}", font_size=20, color=YELLOW)

            # Bar showing weight
            bar = Rectangle(
                width=disc * 3,
                height=0.3,
                color=YELLOW,
                fill_opacity=0.7
            )

            item = VGroup(pos_text, bar, disc_text).arrange(RIGHT, buff=0.3)
            discount_group.add(item)

        discount_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        discount_group.move_to(ORIGIN)

        self.play(FadeIn(discount_group), run_time=2)

        conclusion = Text(
            "Earlier results matter more!",
            font_size=28,
            color=YELLOW,
            weight=BOLD
        )
        conclusion.to_edge(DOWN)

        self.play(Write(conclusion), run_time=1)
        self.wait(2)


class ExperimentalDesign(Scene):
    """Show the experimental setup"""

    def construct(self):
        title = Text("Experimental Design", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Show variables
        iv_title = Text("Independent Variables:", font_size=32, color=BLUE)
        iv_title.shift(UP * 2 + LEFT * 3)
        self.play(Write(iv_title))

        ivs = [
            "• Algorithm (TF-IDF, BM25, Rocchio)",
            "• Corpus Size (1.4K, 10K, 50K, 100K)",
            "• Preprocessing (with/without stemming)"
        ]

        iv_group = VGroup()
        for iv_text in ivs:
            iv = Text(iv_text, font_size=24, color=WHITE)
            iv_group.add(iv)

        iv_group.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        iv_group.next_to(iv_title, DOWN, aligned_edge=LEFT, buff=0.3)

        self.play(Write(iv_group), run_time=2)

        # Show dependent variables
        dv_title = Text("Dependent Variables:", font_size=32, color=GREEN)
        dv_title.shift(UP * 2 + RIGHT * 3)
        self.play(Write(dv_title))

        dvs = [
            "• MAP (primary)",
            "• Precision@k",
            "• NDCG@k",
            "• Processing Time"
        ]

        dv_group = VGroup()
        for dv_text in dvs:
            dv = Text(dv_text, font_size=24, color=WHITE)
            dv_group.add(dv)

        dv_group.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        dv_group.next_to(dv_title, DOWN, aligned_edge=LEFT, buff=0.3)

        self.play(Write(dv_group), run_time=2)

        # Show total experiments
        total = Text(
            "Total Experiments: 3 algorithms × 4 scales = 12 experiments",
            font_size=28,
            color=YELLOW,
            weight=BOLD
        )
        total.to_edge(DOWN)

        self.play(Write(total), run_time=1.5)
        self.wait(3)


class DatasetOverview(Scene):
    """Show datasets used"""

    def construct(self):
        title = Text("Benchmark Datasets", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # CISI
        cisi_card = self.create_dataset_card(
            "CISI (1960s)",
            [
                "1,460 documents",
                "112 queries",
                "Computer Science abstracts",
                "Baseline validation"
            ],
            BLUE,
            LEFT * 4
        )

        self.play(FadeIn(cisi_card), run_time=1.5)
        self.wait(1)

        # MS MARCO
        marco_card = self.create_dataset_card(
            "MS MARCO (2016)",
            [
                "8.8M passages (subsampled)",
                "100 queries per subset",
                "Real Bing search logs",
                "Production-scale evaluation"
            ],
            GREEN,
            RIGHT * 4
        )

        self.play(FadeIn(marco_card), run_time=1.5)
        self.wait(1)

        # Show scales used
        scales_title = Text("MS MARCO Scales Tested:", font_size=32, weight=BOLD, color=YELLOW)
        scales_title.to_edge(DOWN).shift(UP * 1.5)

        scales = Text("10K • 50K • 100K documents", font_size=28, color=WHITE)
        scales.next_to(scales_title, DOWN)

        self.play(Write(scales_title), Write(scales), run_time=1.5)
        self.wait(2)

    def create_dataset_card(self, name, features, color, position):
        """Create a dataset info card"""
        # Border
        card = Rectangle(
            width=5,
            height=4,
            color=color,
            stroke_width=4,
            fill_opacity=0.1
        )

        # Name
        name_text = Text(name, font_size=28, color=color, weight=BOLD)
        name_text.move_to(card.get_top() + DOWN * 0.5)

        # Features
        feature_group = VGroup()
        for feature in features:
            feature_text = Text(f"• {feature}", font_size=20, color=WHITE)
            feature_group.add(feature_text)

        feature_group.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        feature_group.next_to(name_text, DOWN, buff=0.5)

        full_card = VGroup(card, name_text, feature_group)
        full_card.move_to(position)

        return full_card
