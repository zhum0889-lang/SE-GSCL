from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.diagnostics import (  # noqa: E402
    fit_reliability_gate,
    fuse_probabilities,
)
from se_gscl.losses import within_class_symptom_distribution_loss  # noqa: E402
from se_gscl.semantics import (  # noqa: E402
    FrozenSymptomPrototypeBank,
    ResidualSymptomPrototypeBank,
)


class P22SemanticGuardTests(unittest.TestCase):
    def _base_bank(self) -> FrozenSymptomPrototypeBank:
        return FrozenSymptomPrototypeBank(
            torch.randn(6, 8),
            torch.tensor([0, 0, 0, 1, 1, 1]),
            ("Inner", "Outer"),
            tuple(f"s{i}" for i in range(6)),
            tuple(f"S{i}" for i in range(6)),
            tuple(f"p{i}" for i in range(6)),
            "unit-test",
        )

    def test_residual_bank_limits_semantic_drift(self) -> None:
        base = self._base_bank()
        bank = ResidualSymptomPrototypeBank(
            base,
            max_residual_scale=0.2,
        )
        torch.testing.assert_close(bank(), base.prototypes)
        with torch.no_grad():
            bank.residual.copy_(torch.randn_like(bank.residual) * 100.0)
        adapted = bank()
        cosine = torch.sum(adapted * base.prototypes, dim=1)
        self.assertTrue(torch.all(cosine > 0.95))
        adapted.sum().backward()
        self.assertIsNotNone(bank.residual.grad)

    def test_distribution_loss_rewards_correct_within_class_order(self) -> None:
        targets = torch.tensor([[0.9, 0.6, 0.3]])
        weights = torch.ones_like(targets)
        labels = torch.tensor([0])
        class_ids = torch.tensor([0, 0, 0])
        aligned = within_class_symptom_distribution_loss(
            torch.tensor([[0.9, 0.6, 0.3]]),
            targets,
            weights,
            labels,
            class_ids,
        )
        reversed_order = within_class_symptom_distribution_loss(
            torch.tensor([[0.3, 0.6, 0.9]]),
            targets,
            weights,
            labels,
            class_ids,
        )
        self.assertLess(float(aligned), float(reversed_order))

    def test_validation_gate_can_select_reliable_local_overrides(self) -> None:
        labels = np.asarray([0, 1, 2, 3])
        global_probabilities = np.asarray(
            [
                [0.40, 0.45, 0.10, 0.05],
                [0.45, 0.40, 0.10, 0.05],
                [0.05, 0.10, 0.40, 0.45],
                [0.05, 0.10, 0.45, 0.40],
            ]
        )
        local_probabilities = np.asarray(
            [
                [0.90, 0.04, 0.03, 0.03],
                [0.04, 0.90, 0.03, 0.03],
                [0.03, 0.04, 0.90, 0.03],
                [0.03, 0.04, 0.03, 0.90],
            ]
        )
        gate = fit_reliability_gate(
            global_probabilities,
            local_probabilities,
            labels,
        )
        weights = gate.local_weights(
            global_probabilities,
            local_probabilities,
        )
        fused = fuse_probabilities(
            global_probabilities,
            local_probabilities,
            weights,
        )
        self.assertTrue(np.array_equal(fused.argmax(axis=1), labels))
        self.assertEqual(gate.validation_balanced_accuracy, 1.0)
        self.assertTrue(np.all(weights >= 0.5))


if __name__ == "__main__":
    unittest.main()
