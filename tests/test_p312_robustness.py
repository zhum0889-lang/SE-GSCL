from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_p31_robustness import (  # noqa: E402
    _domain_accuracy_stats,
    main,
    _nested_stats,
    _stats,
)


class P312RobustnessTests(unittest.TestCase):
    def test_stats_reports_population_variation(self) -> None:
        result = _stats([0.8, 0.9, 1.0])
        self.assertAlmostEqual(result["mean"], 0.9)
        self.assertAlmostEqual(result["min"], 0.8)
        self.assertAlmostEqual(result["max"], 1.0)
        self.assertGreater(result["std"], 0.0)

    def test_nested_class_and_domain_metrics_are_aggregated(self) -> None:
        reports = [
            {
                "continuous_prompt_metrics": {
                    "per_class_recall": {"Normal": 1.0, "Ball": 0.8},
                    "domain_metrics": {
                        "0": {"accuracy": 1.0},
                        "1": {"accuracy": 0.8},
                    },
                }
            },
            {
                "continuous_prompt_metrics": {
                    "per_class_recall": {"Normal": 0.9, "Ball": 1.0},
                    "domain_metrics": {
                        "0": {"accuracy": 0.9},
                        "1": {"accuracy": 1.0},
                    },
                }
            },
        ]
        classes = _nested_stats(
            reports,
            "continuous_prompt_metrics",
            "per_class_recall",
        )
        domains = _domain_accuracy_stats(
            reports,
            "continuous_prompt_metrics",
        )
        self.assertAlmostEqual(classes["Normal"]["mean"], 0.95)
        self.assertAlmostEqual(classes["Ball"]["mean"], 0.9)
        self.assertAlmostEqual(domains["0"]["mean"], 0.95)
        self.assertAlmostEqual(domains["1"]["mean"], 0.9)

    def test_main_writes_json_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_paths = []
            for seed, accuracy in ((42, 1.0), (52, 0.9)):
                report = {
                    "status": "ok",
                    "training": {"seed": seed},
                    "continuous_prompt_metrics": {
                        "samples": 10,
                        "valid_label_rate": 1.0,
                        "accuracy": accuracy,
                        "balanced_accuracy": accuracy,
                        "per_class_recall": {
                            "Normal": accuracy,
                            "Ball": accuracy,
                        },
                        "domain_metrics": {
                            "0": {"accuracy": accuracy},
                            "1": {"accuracy": accuracy},
                        },
                    },
                    "training_only_auxiliary_probe_metrics": {
                        "accuracy": 1.0
                    },
                    "upstream_fused_baseline": {"accuracy": 0.9},
                    "qwen_upstream_paired_comparison": {
                        "prediction_agreement_rate": 0.9,
                        "qwen_only_correct": 1,
                        "upstream_only_correct": 0,
                        "qwen_minus_upstream_accuracy": accuracy - 0.9,
                    },
                }
                path = root / f"seed_{seed}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                report_paths.append(path)
            output_dir = root / "summary"
            arguments = [
                "summarize_p31_robustness.py",
                "--reports",
                *(str(path) for path in report_paths),
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", arguments):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
            summary = json.loads(
                (output_dir / "p312_robustness_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["num_runs"], 2)
            self.assertAlmostEqual(
                summary["aggregate"]["qwen_accuracy"]["mean"],
                0.95,
            )
            self.assertTrue(
                (output_dir / "p312_robustness_summary.md").is_file()
            )


if __name__ == "__main__":
    unittest.main()
