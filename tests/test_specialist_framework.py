from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.losses import (  # noqa: E402
    cross_condition_supervised_contrastive_loss,
    cross_covariance_loss,
    global_prototype_alignment_loss,
)
from se_gscl.models import SEGSCLSpecialist  # noqa: E402
from se_gscl.semantics import FrozenPrototypeBank  # noqa: E402


class SpecialistFrameworkTests(unittest.TestCase):
    def test_specialist_emits_expected_tokens_and_heads(self) -> None:
        model = SEGSCLSpecialist(
            input_channels=3,
            token_dim=64,
            branch_dim=8,
            num_tokens=16,
            kernels=(5, 9),
            temporal_dilations=(1, 2, 4),
            temporal_dropout=0.1,
            normalization="group",
            num_domains=4,
            condition_dim=3,
        )
        output = model(torch.randn(5, 3, 2048))
        self.assertEqual(output.signal_tokens.shape, (5, 16, 64))
        self.assertEqual(output.fault_tokens.shape, (5, 16, 64))
        self.assertEqual(output.condition_tokens.shape, (5, 16, 64))
        self.assertEqual(output.fault_embedding.shape, (5, 64))
        self.assertEqual(output.condition_embedding.shape, (5, 64))
        self.assertEqual(output.domain_logits.shape, (5, 4))
        self.assertEqual(output.condition_values.shape, (5, 3))
        self.assertEqual(len(model.encoder.branches), 2)
        self.assertEqual(len(model.encoder.temporal_mixer), 3)
        self.assertTrue(
            any(
                isinstance(module, torch.nn.GroupNorm)
                for module in model.encoder.modules()
            )
        )
        torch.testing.assert_close(
            output.fault_embedding.norm(dim=1),
            torch.ones(5),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_global_and_cross_condition_losses_backpropagate(self) -> None:
        model = SEGSCLSpecialist(
            input_channels=1,
            token_dim=32,
            branch_dim=8,
            num_tokens=8,
            num_domains=2,
        )
        output = model(torch.randn(6, 1, 1024))
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        domains = torch.tensor([0, 1, 0, 1, 0, 1])
        bank = FrozenPrototypeBank(torch.randn(3, 32), ["A", "B", "C"], "test-v1")
        global_loss, logits = global_prototype_alignment_loss(
            output.fault_embedding,
            bank.prototypes,
            labels,
        )
        cross_condition = cross_condition_supervised_contrastive_loss(
            output.fault_embedding,
            labels,
            domains,
        )
        decorrelation = cross_covariance_loss(
            output.fault_embedding,
            output.condition_embedding,
        )
        loss = global_loss + cross_condition + 0.01 * decorrelation
        loss.backward()

        self.assertEqual(logits.shape, (6, 3))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )
        self.assertIsNone(bank.prototypes.grad)

    def test_temporal_mixer_receives_gradients(self) -> None:
        model = SEGSCLSpecialist(
            token_dim=32,
            branch_dim=8,
            num_tokens=8,
            temporal_dilations=(1, 2),
        )
        output = model(torch.randn(4, 1, 1024))
        output.fault_tokens.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.encoder.temporal_mixer.parameters()
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))

    def test_cross_condition_loss_skips_batches_without_valid_pairs(self) -> None:
        embeddings = torch.randn(4, 16, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])
        domains = torch.tensor([0, 0, 1, 1])
        loss = cross_condition_supervised_contrastive_loss(
            embeddings,
            labels,
            domains,
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(embeddings.grad)


if __name__ == "__main__":
    unittest.main()
