from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_p32_continuous_explanations import (  # noqa: E402
    _assemble_continuous_inputs,
    _continuous_packet,
    _failure_audit,
    _preservation_metrics,
)


class P32ContinuousExplanationTests(unittest.TestCase):
    def test_batch_assembly_left_pads_before_continuous_tokens(self) -> None:
        embedding = nn.Embedding(16, 6)
        prompts = torch.randn(2, 2, 6)
        token_rows = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4]),
        ]
        with torch.inference_mode():
            inputs, attention = _assemble_continuous_inputs(
                prompts,
                token_rows,
                embedding,
                pad_token_id=0,
            )
        self.assertEqual(inputs.shape, (2, 5, 6))
        self.assertEqual(attention.tolist()[0], [1, 1, 1, 1, 1])
        self.assertEqual(attention.tolist()[1], [0, 0, 1, 1, 1])
        torch.testing.assert_close(inputs[1, 2:4], prompts[1])

    def test_continuous_packet_uses_direct_prediction_without_losing_p2(self) -> None:
        packet = {
            "predicted_class_id": 3,
            "predicted_class_name": "OuterRace",
            "top_candidates": [
                {
                    "class_id": 1,
                    "class_name": "InnerRace",
                    "probability": 0.7,
                },
                {
                    "class_id": 3,
                    "class_name": "OuterRace",
                    "probability": 0.3,
                },
            ],
        }
        updated = _continuous_packet(
            packet,
            {
                "predicted_class_id": 1,
                "predicted_class_name": "InnerRace",
            },
        )
        self.assertEqual(updated["predicted_class_name"], "InnerRace")
        self.assertEqual(updated["p2_predicted_class_name"], "OuterRace")
        self.assertAlmostEqual(updated["confidence"], 0.7)

    def test_preservation_metrics_compare_both_diagnostic_stages(self) -> None:
        records = [
            {
                "sample_id": 1,
                "packet": {
                    "ground_truth_class_name": "InnerRace",
                    "p2_predicted_class_name": "OuterRace",
                },
                "direct_prediction": {
                    "predicted_class_name": "InnerRace"
                },
                "parsed_output": {"diagnosis": "InnerRace"},
                "controlled_output": {"semantic_control_repairs": []},
            },
            {
                "sample_id": 2,
                "packet": {
                    "ground_truth_class_name": "Ball",
                    "p2_predicted_class_name": "Ball",
                },
                "direct_prediction": {"predicted_class_name": "Ball"},
                "parsed_output": {"diagnosis": "OuterRace"},
                "controlled_output": {"semantic_control_repairs": []},
            },
            {
                "sample_id": 3,
                "packet": {
                    "ground_truth_class_name": "Normal",
                    "p2_predicted_class_name": "Normal",
                },
                "direct_prediction": {"predicted_class_name": "Normal"},
                "parsed_output": None,
                "controlled_output": {
                    "semantic_control_repairs": [
                        "diagnosis_restored_to_direct_prompt"
                    ]
                },
            },
        ]
        metrics = _preservation_metrics(records)
        self.assertAlmostEqual(metrics["direct_prompt_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["p2_fused_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["diagnosis_preservation_rate"], 1 / 3)
        self.assertAlmostEqual(
            metrics["valid_output_diagnosis_preservation_rate"],
            0.5,
        )
        audit = _failure_audit(records)
        self.assertEqual(audit["unparseable_sample_ids"], [3])
        self.assertEqual(
            audit["raw_diagnosis_drift_sample_ids"],
            [2],
        )
        self.assertEqual(
            audit["semantic_control_repaired_sample_ids"],
            [3],
        )


if __name__ == "__main__":
    unittest.main()
