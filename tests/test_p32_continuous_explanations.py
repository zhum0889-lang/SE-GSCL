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
                "packet": {
                    "ground_truth_class_name": "InnerRace",
                    "p2_predicted_class_name": "OuterRace",
                },
                "direct_prediction": {
                    "predicted_class_name": "InnerRace"
                },
                "parsed_output": {"diagnosis": "InnerRace"},
            },
            {
                "packet": {
                    "ground_truth_class_name": "Ball",
                    "p2_predicted_class_name": "Ball",
                },
                "direct_prediction": {"predicted_class_name": "Ball"},
                "parsed_output": {"diagnosis": "OuterRace"},
            },
        ]
        metrics = _preservation_metrics(records)
        self.assertAlmostEqual(metrics["direct_prompt_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["p2_fused_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["diagnosis_preservation_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
