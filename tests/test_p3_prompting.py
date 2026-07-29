from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.llm import (  # noqa: E402
    ALLOWED_MAINTENANCE_ACTIONS,
    build_diagnostic_messages,
    evaluate_llm_outputs,
    parse_diagnostic_json,
    select_evaluation_rows,
)


def _packet(sample_id: int = 1, domain_id: int = 0) -> dict:
    return {
        "sample_id": sample_id,
        "domain_id": domain_id,
        "predicted_class_id": 1,
        "predicted_class_name": "InnerRace",
        "confidence": 0.8,
        "normalized_entropy": 0.4,
        "top1_top2_margin": 0.5,
        "global_local_agreement": True,
        "top_candidates": [
            {
                "class_id": 1,
                "class_name": "InnerRace",
                "probability": 0.8,
                "global_probability": 0.85,
                "local_probability": 0.7,
            },
            {
                "class_id": 3,
                "class_name": "OuterRace",
                "probability": 0.2,
                "global_probability": 0.15,
                "local_probability": 0.3,
            },
        ],
        "top_symptoms": [
            {
                "symptom_id": 3,
                "symptom_name": "BPFI impact train",
                "class_id": 1,
                "probability": 0.7,
            }
        ],
        "ground_truth_class_id": 1,
        "ground_truth_class_name": "InnerRace",
        "is_correct": True,
    }


class P3PromptingTests(unittest.TestCase):
    def test_prompt_does_not_expose_ground_truth(self) -> None:
        text = str(build_diagnostic_messages(_packet()))
        self.assertNotIn("ground_truth", text)
        self.assertNotIn("is_correct", text)
        self.assertIn("BPFI impact train", text)

    def test_json_parser_accepts_fenced_response(self) -> None:
        parsed = parse_diagnostic_json(
            '```json\n{"diagnosis":"InnerRace"}\n```'
        )
        self.assertEqual(parsed, {"diagnosis": "InnerRace"})

    def test_selection_covers_each_domain(self) -> None:
        rows = [
            _packet(sample_id=index, domain_id=index % 4)
            for index in range(20)
        ]
        selected = select_evaluation_rows(rows, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual({row["domain_id"] for row in selected}, {0, 1, 2, 3})

    def test_metrics_reject_unsupported_evidence(self) -> None:
        packet = _packet()
        record = {
            "packet": packet,
            "parsed_output": {
                "diagnosis": "InnerRace",
                "confidence_level": "high",
                "supporting_evidence": ["invented frequency"],
                "counter_evidence": [],
                "explanation": "test",
                "uncertainty_acknowledged": False,
                "maintenance_action": ALLOWED_MAINTENANCE_ACTIONS[2],
            },
        }
        metrics = evaluate_llm_outputs([record])
        self.assertEqual(metrics["json_parse_rate"], 1.0)
        self.assertEqual(metrics["candidate_label_valid_rate"], 1.0)
        self.assertEqual(metrics["evidence_item_grounded_rate"], 0.0)

    def test_metrics_detect_class_inconsistent_support(self) -> None:
        packet = _packet()
        record = {
            "packet": packet,
            "parsed_output": {
                "diagnosis": "InnerRace",
                "confidence_level": "high",
                "supporting_evidence": ["Outer-race symptom"],
                "counter_evidence": [],
                "explanation": "test",
                "uncertainty_acknowledged": False,
                "maintenance_action": ALLOWED_MAINTENANCE_ACTIONS[2],
            },
        }
        packet["top_symptoms"].append(
            {
                "symptom_id": 9,
                "symptom_name": "Outer-race symptom",
                "class_id": 3,
                "probability": 0.2,
            }
        )
        metrics = evaluate_llm_outputs([record])
        self.assertEqual(metrics["evidence_item_grounded_rate"], 1.0)
        self.assertEqual(
            metrics["supporting_evidence_class_consistency_rate"],
            0.0,
        )
        self.assertEqual(metrics["contradictory_support_rate"], 1.0)

    def test_metrics_detect_unsafe_maintenance_policy(self) -> None:
        packet = _packet()
        record = {
            "packet": packet,
            "parsed_output": {
                "diagnosis": "InnerRace",
                "confidence_level": "high",
                "supporting_evidence": ["BPFI impact train"],
                "counter_evidence": [],
                "explanation": "test",
                "uncertainty_acknowledged": False,
                "maintenance_action": ALLOWED_MAINTENANCE_ACTIONS[0],
            },
        }
        metrics = evaluate_llm_outputs([record])
        self.assertEqual(metrics["maintenance_action_valid_rate"], 1.0)
        self.assertEqual(
            metrics["maintenance_policy_consistency_rate"],
            0.0,
        )

    def test_unparseable_output_counts_as_end_to_end_failure(self) -> None:
        packet = _packet()
        packet["normalized_entropy"] = 0.9
        metrics = evaluate_llm_outputs(
            [{"packet": packet, "parsed_output": None}]
        )
        self.assertEqual(metrics["json_parse_rate"], 0.0)
        self.assertEqual(metrics["candidate_label_valid_rate"], 0.0)
        self.assertEqual(metrics["llm_diagnosis_accuracy"], 0.0)
        self.assertEqual(metrics["uncertain_samples"], 1)
        self.assertEqual(metrics["uncertainty_acknowledgement_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
