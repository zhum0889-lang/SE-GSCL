"""Low-rank adapters for direct semantic-vector prompting of a frozen LLM."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import nn


def build_continuous_context(
    arrays: Mapping[str, np.ndarray],
    *,
    mode: str = "no_condition",
) -> np.ndarray:
    """Combine P2 semantic evidence and reliability into one adapter input.

    The returned vector deliberately preserves soft diagnostic evidence instead
    of a top-1 class label.  It is the only sample-level interface supplied to
    the continuous-prompt adapter in P3.1.
    """

    supported_modes = {
        "full",
        "no_condition",
        "fault_identity_only",
    }
    if mode not in supported_modes:
        raise ValueError(
            f"Unsupported continuous context mode {mode!r}; "
            f"choose from {sorted(supported_modes)}."
        )
    fuzzy = np.asarray(
        arrays["fuzzy_symptom_embeddings"],
        dtype=np.float32,
    )
    # Keep semantic directions comparable across samples while retaining the
    # posterior and reliability scalars below in their native [0, 1] ranges.
    fuzzy_norm = np.linalg.norm(fuzzy, axis=1, keepdims=True)
    fuzzy = fuzzy / np.maximum(fuzzy_norm, 1e-6)
    fused = np.asarray(arrays["fused_probabilities"], dtype=np.float32)
    global_probabilities = np.asarray(
        arrays["global_probabilities"],
        dtype=np.float32,
    )
    local_probabilities = np.asarray(
        arrays["local_probabilities"],
        dtype=np.float32,
    )
    if not (
        len(fuzzy)
        == len(fused)
        == len(global_probabilities)
        == len(local_probabilities)
    ):
        raise ValueError("Continuous-prompt arrays must share sample count.")
    # Entropy, Top-1/Top-2 margin, and global-local agreement tell the adapter
    # how decisive and internally consistent the upstream P2 evidence is.
    clipped = np.clip(fused, 1e-8, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    entropy = entropy / math.log(fused.shape[1])
    ordered = np.sort(fused, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    agreement = (
        global_probabilities.argmax(axis=1)
        == local_probabilities.argmax(axis=1)
    ).astype(np.float32)
    reliability = np.concatenate(
        [
            entropy[:, None].astype(np.float32),
            margin[:, None].astype(np.float32),
            agreement[:, None],
        ],
        axis=1,
    )
    if mode == "fault_identity_only":
        return np.concatenate(
            [global_probabilities, reliability],
            axis=1,
        ).astype(np.float32)

    blocks = [fuzzy, fused, reliability]
    if mode == "full":
        if "condition_features" not in arrays:
            raise ValueError(
                "Full continuous prompting requires label-free observable "
                "condition_features exported by the current P2 pipeline."
            )
        condition = np.asarray(
            arrays["condition_features"],
            dtype=np.float32,
        )
        if len(condition) != len(fuzzy):
            raise ValueError("Condition features must share sample count.")
        blocks.append(condition)
    return np.concatenate(blocks, axis=1).astype(np.float32)


class LowRankContinuousPromptAdapter(nn.Module):
    """Map one semantic context to multiple frozen-LLM input embeddings.

    This module is intentionally much smaller than the LLM.  It learns the
    semantic-to-token interface while the pretrained language-model parameters
    remain fixed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        *,
        num_prompt_tokens: int = 4,
        rank: int = 64,
        initial_scale: float = 0.02,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_size, num_prompt_tokens, rank) <= 0:
            raise ValueError("All adapter dimensions must be positive.")
        if num_classes is not None and num_classes <= 0:
            raise ValueError("num_classes must be positive when provided.")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.rank = int(rank)
        self.num_classes = (
            int(num_classes) if num_classes is not None else None
        )
        # The low-rank bottleneck limits trainable parameters and separates the
        # continuous engineering-semantic interface from the frozen LLM.
        self.down = nn.Linear(self.input_dim, self.rank)
        self.token_codes = nn.Parameter(
            torch.empty(self.num_prompt_tokens, self.rank)
        )
        self.up = nn.Linear(self.rank, self.hidden_size, bias=False)
        self.output_norm = nn.LayerNorm(
            self.hidden_size,
            elementwise_affine=False,
        )
        self.output_scale = nn.Parameter(
            torch.tensor(float(initial_scale))
        )
        # This head is used only as an auxiliary training regularizer; the
        # diagnostic label used at inference is generated by the frozen LLM.
        self.semantic_classifier = (
            nn.Linear(self.hidden_size, self.num_classes)
            if self.num_classes is not None
            else None
        )
        nn.init.normal_(self.token_codes, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.input_dim:
            raise ValueError(
                f"context must have shape [B,{self.input_dim}]."
            )
        # A shared semantic base plus token-specific codes yields Q distinct
        # continuous prompt tokens for each sample.
        base = torch.tanh(self.down(context))
        latent = base[:, None, :] + self.token_codes[None, :, :]
        tokens = self.up(torch.tanh(latent))
        # Normalization and bounded scale keep learned token magnitudes close
        # to the numerical range expected by the frozen token embedding space.
        scale = self.output_scale.abs().clamp(min=1e-4, max=1.0)
        return self.output_norm(tokens) * scale

    def classification_logits(
        self,
        prompt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Classify pooled prompt tokens for training-time regularization."""

        if self.semantic_classifier is None:
            raise RuntimeError(
                "The adapter was created without a semantic classifier."
            )
        expected = (
            self.num_prompt_tokens,
            self.hidden_size,
        )
        if prompt_tokens.ndim != 3 or tuple(prompt_tokens.shape[1:]) != expected:
            raise ValueError(
                "prompt_tokens must have shape "
                f"[B,{self.num_prompt_tokens},{self.hidden_size}]."
            )
        return self.semantic_classifier(prompt_tokens.mean(dim=1))
