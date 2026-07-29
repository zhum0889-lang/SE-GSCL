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
from scripts.train_p31_continuous_prompt import _training_batch  # noqa: E402


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

    def test_adapter_emits_prompt_tokens_and_gradients(self) -> None:
        adapter = LowRankContinuousPromptAdapter(
            input_dim=15,
            hidden_size=32,
            num_prompt_tokens=4,
            rank=8,
        )
        context = torch.randn(3, 15)
        tokens = adapter(context)
        self.assertEqual(tokens.shape, (3, 4, 32))
        tokens.square().mean().backward()
        self.assertIsNotNone(adapter.down.weight.grad)
        self.assertIsNotNone(adapter.up.weight.grad)
        self.assertIsNotNone(adapter.token_codes.grad)

    def test_loss_backpropagates_through_frozen_decoder_inputs(self) -> None:
        model = _TinyFrozenDecoder().requires_grad_(False)
        adapter = LowRankContinuousPromptAdapter(
            input_dim=15,
            hidden_size=24,
            num_prompt_tokens=4,
            rank=8,
        )
        inputs_embeds, attention, labels = _training_batch(
            torch.randn(2, 15),
            torch.tensor([0, 1]),
            adapter,
            model,
            torch.tensor([1, 2, 3]),
            [torch.tensor([4, 5]), torch.tensor([6, 5])],
            pad_token_id=0,
        )
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


if __name__ == "__main__":
    unittest.main()
