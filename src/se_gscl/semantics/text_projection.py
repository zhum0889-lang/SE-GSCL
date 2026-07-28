"""Trainable projection from frozen LLM states to fault semantic prototypes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .prototype_bank import FrozenPrototypeBank
from .text_cache import TextEmbeddingCache


class ProjectedTextPrototypeBank(nn.Module):
    """Aggregate projected multi-description embeddings into class prototypes."""

    def __init__(
        self,
        cache: TextEmbeddingCache,
        semantic_dim: int,
    ) -> None:
        super().__init__()
        self.register_buffer("text_embeddings", cache.embeddings.float())
        self.register_buffer("class_ids", cache.class_ids.long())
        self.class_names = cache.class_names
        self.version = cache.version
        self.projection = nn.Sequential(
            nn.LayerNorm(cache.hidden_size),
            nn.Linear(cache.hidden_size, semantic_dim),
        )

    def forward(self) -> torch.Tensor:
        # Decoder hidden states share a strong common direction. Removing the
        # ontology-only mean preserves class differences without using signals.
        centered = self.text_embeddings - self.text_embeddings.mean(
            dim=0,
            keepdim=True,
        )
        projected = self.projection(centered)
        prototypes = [
            projected[self.class_ids == class_id].mean(dim=0)
            for class_id in range(len(self.class_names))
        ]
        return F.normalize(torch.stack(prototypes, dim=0), dim=-1)

    @torch.no_grad()
    def freeze(self, version: str | None = None) -> FrozenPrototypeBank:
        return FrozenPrototypeBank(
            self(),
            self.class_names,
            version or self.version,
        )
