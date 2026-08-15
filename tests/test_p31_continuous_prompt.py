from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.llm import (  # noqa: E402
    LowRankContinuousPromptAdapter,
    build_continuous_context,
)
from scripts.train_p31_continuous_prompt import (  # noqa: E402
    _paired_comparison,
    _training_batch,
)


class _TinyFrozenDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 24)
        self.head = nn.Linear(24, 32, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, labels, use_cache):
        del attention_mask, use_cache
        logits = self.head(inputs_embeds)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return type("Output", (), {"loss": loss})()


class P31ContinuousPromptTests(unittest.TestCase):
    def test_context_contains_semantics_posterior_and_reliability(self) -> None:
        arrays = {
            "fuzzy_symptom_embeddings": np.ones((3, 8), dtype=np.float32),
            "fused_probabilities": np.asarray(
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                ],
                dtype=np.float32,
            ),
            "global_probabilities": np.asarray(
                [[0.7, 0.1, 0.1, 0.1]] * 3,
                dtype=np.float32,
            ),
            "local_probabilities": np.asarray(
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                ],
                dtype=np.float32,
            ),
        }
        context = build_continuous_context(arrays)
        self.assertEqual(context.shape, (3, 15))
        self.assertTrue(np.isfinite(context).all())
        np.testing.assert_allclose(
            np.linalg.norm(context[:, :8], axis=1),
            np.ones(3),
            rtol=1e-5,
        )

    def test_full_context_adds_only_observable_condition_features(self) -> None:
        arrays = {
            "fuzzy_symptom_embeddings": np.ones((2, 6), dtype=np.float32),
            "fused_probabilities": np.asarray(
                [[0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]],
                dtype=np.float32,
            ),
            "global_probabilities": np.asarray(
                [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]],
                dtype=np.float32,
            ),
            "local_probabilities": np.asarray(
                [[0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]],
                dtype=np.float32,
            ),
            "condition_features": np.asarray(
                [[0.5, 0.64, 0.1, 0.2, 1, 1, 1, 1]] * 2,
                dtype=np.float32,
            ),
        }
        no_condition = build_continuous_context(
            arrays,
            mode="no_condition",
        )
        full = build_continuous_context(arrays, mode="full")
        identity = build_continuous_context(
            arrays,
            mode="fault_identity_only",
        )
        self.assertEqual(no_condition.shape, (2, 13))
        self.assertEqual(full.shape, (2, 21))
        self.assertEqual(identity.shape, (2, 7))
        np.testing.assert_allclose(full[:, -8:], arrays["condition_features"])

    def test_context_retains_fuzzy_identity_and_description_posterior(self) -> None:
        arrays = {
            "fuzzy_identity_embeddings": np.ones((2, 5), dtype=np.float32),
            "identity_description_probabilities": np.asarray(
                [[0.4, 0.1, 0.2, 0.3], [0.1, 0.4, 0.3, 0.2]],
                dtype=np.float32,
            ),
            "fuzzy_symptom_embeddings": np.ones((2, 6), dtype=np.float32),
            "fused_probabilities": np.asarray(
                [[0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]],
                dtype=np.float32,
            ),
            "global_probabilities": np.asarray(
                [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]],
                dtype=np.float32,
            ),
            "local_probabilities": np.asarray(
                [[0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]],
                dtype=np.float32,
            ),
        }
        full_semantics = build_continuous_context(arrays, mode="no_condition")
        without_identity = build_continuous_context(
            arrays,
            mode="no_fuzzy_identity",
        )
        identity = build_continuous_context(
            arrays,
            mode="fault_identity_only",
        )
        self.assertEqual(full_semantics.shape, (2, 22))
        self.assertEqual(without_identity.shape, (2, 13))
        self.assertEqual(identity.shape, (2, 16))
        np.testing.assert_allclose(
            np.linalg.norm(full_semantics[:, :5], axis=1),
            np.ones(2),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            full_semantics[:, 5:9],
            arrays["identity_description_probabilities"],
        )

    def test_adapter_emits_prompt_tokens_and_gradients(self) -> None:
        adapter = LowRankContinuousPromptAdapter(
            input_dim=15,
            hidden_size=32,
            num_prompt_tokens=4,
            rank=8,
            num_classes=4,
        )
        context = torch.randn(3, 15)
        tokens = adapter(context)
        self.assertEqual(tokens.shape, (3, 4, 32))
        logits = adapter.classification_logits(tokens)
        self.assertEqual(logits.shape, (3, 4))
        (tokens.square().mean() + logits.square().mean()).backward()
        self.assertIsNotNone(adapter.down.weight.grad)
        self.assertIsNotNone(adapter.up.weight.grad)
        self.assertIsNotNone(adapter.token_codes.grad)
        self.assertIsNotNone(adapter.semantic_classifier.weight.grad)

    def test_loss_backpropagates_through_frozen_decoder_inputs(self) -> None:
        model = _TinyFrozenDecoder().requires_grad_(False)
        adapter = LowRankContinuousPromptAdapter(
            input_dim=15,
            hidden_size=24,
            num_prompt_tokens=4,
            rank=8,
        )
        inputs_embeds, attention, labels, prompt_embeddings = _training_batch(
            torch.randn(2, 15),
            torch.tensor([0, 1]),
            adapter,
            model,
            torch.tensor([1, 2, 3]),
            [torch.tensor([4, 5]), torch.tensor([6, 5])],
            pad_token_id=0,
        )
        self.assertEqual(prompt_embeddings.shape, (2, 4, 24))
        output = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            labels=labels,
            use_cache=False,
        )
        output.loss.backward()
        self.assertIsNotNone(adapter.down.weight.grad)
        self.assertIsNone(model.embedding.weight.grad)
        self.assertIsNone(model.head.weight.grad)

    def test_paired_comparison_separates_corrected_and_new_errors(self) -> None:
        metrics = _paired_comparison(
            generated=np.asarray([0, 1, 2, 0]),
            upstream=np.asarray([0, 2, 2, 3]),
            labels=np.asarray([0, 1, 3, 3]),
        )
        self.assertEqual(metrics["both_correct"], 1)
        self.assertEqual(metrics["qwen_only_correct"], 1)
        self.assertEqual(metrics["upstream_only_correct"], 1)
        self.assertEqual(metrics["both_wrong"], 1)
        self.assertAlmostEqual(metrics["qwen_minus_upstream_accuracy"], 0.0)
        self.assertAlmostEqual(metrics["correction_rate"], 0.25)
        self.assertAlmostEqual(metrics["corruption_rate"], 0.25)
        self.assertAlmostEqual(metrics["net_correction_rate"], 0.0)
        self.assertAlmostEqual(metrics["mcnemar_exact_p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
