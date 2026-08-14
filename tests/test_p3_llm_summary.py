from __future__ import annotations

import unittest

from scripts.summarize_p3_llm_experiment import (
    _aggregate,
    _by_seed_rows,
    _markdown,
)


def _classification(accuracy: float) -> dict:
    return {
        "samples": 20,
        "accuracy": accuracy,
        "balanced_accuracy": accuracy,
        "candidate_coverage": {
            "top_1": accuracy,
            "top_2": min(1.0, accuracy + 0.1),
            "top_3": 1.0,
        },
        "domain_metrics": {
            "0": {
                "accuracy": accuracy,
                "balanced_accuracy": accuracy,
            }
        },
    }


class P3LlmSummaryTests(unittest.TestCase):
    def test_summary_preserves_small_model_and_paired_llm_metrics(self) -> None:
        reports = []
        for seed, llm_accuracy in ((42, 0.8), (52, 0.9), (62, 1.0)):
            report = {
                "status": "ok",
                "llm_tuning": "lora",
                "input_contract": {"context_mode": "full"},
                "continuous_prompt_metrics": {
                    **_classification(llm_accuracy),
                    "domain_metrics": {
                        "0": {
                            "accuracy": llm_accuracy,
                            "balanced_accuracy": llm_accuracy,
                        }
                    },
                },
                "upstream_semantic_baselines": {
                    "global_fault_identity": _classification(0.7),
                    "hierarchical_fusion": _classification(0.75),
                },
                "llm_upstream_paired_comparison": {
                    "qwen_minus_upstream_accuracy": llm_accuracy - 0.75,
                    "correction_rate": 0.15,
                    "corruption_rate": 0.05,
                    "net_correction_rate": 0.1,
                    "prediction_agreement_rate": 0.8,
                    "mcnemar_exact_p_value": 0.03,
                },
            }
            reports.append(("continuous_full_lora", seed, report))

        rows, domain_rows = _by_seed_rows(reports)
        summary = _aggregate(rows)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(domain_rows), 3)
        self.assertEqual(summary[0]["num_seeds"], 3)
        self.assertAlmostEqual(summary[0]["llm_accuracy_mean"], 0.9)
        self.assertAlmostEqual(summary[0]["fused_accuracy_mean"], 0.75)
        self.assertAlmostEqual(summary[0]["net_correction_rate_mean"], 0.1)
        self.assertEqual(summary[0]["mcnemar_significant_seeds"], 3)
        self.assertAlmostEqual(summary[0]["mcnemar_p_value_max"], 0.03)
        markdown = _markdown(summary)
        self.assertIn("continuous_full_lora", markdown)
        self.assertIn("0.9000", markdown)


if __name__ == "__main__":
    unittest.main()
