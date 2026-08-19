"""Versioned frozen prototype bank for P1 experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FrozenPrototypeBank(nn.Module):
    def __init__(
        self,
        prototypes: torch.Tensor,
        class_names: Sequence[str],
        version: str,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(prototypes, dtype=torch.float32)
        if values.ndim != 2:
            raise ValueError("prototypes must have shape [K,d].")
        if len(class_names) != values.shape[0]:
            raise ValueError("class_names and prototypes must have equal length.")
        if not version:
            raise ValueError("prototype version must be non-empty.")
        self.register_buffer("prototypes", F.normalize(values, dim=-1))
        self.class_names = tuple(str(name) for name in class_names)
        self.version = str(version)

    @property
    def semantic_dim(self) -> int:
        return int(self.prototypes.shape[1])

    def similarities(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"Expected semantic dim {self.semantic_dim}, got {embeddings.shape[-1]}"
            )
        return F.normalize(embeddings, dim=-1) @ self.prototypes.T


class LearnedPrototypeBank(nn.Module):
    """Class anchors learned from the initial signal domain without text."""

    def __init__(
        self,
        class_names: Sequence[str],
        semantic_dim: int,
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in class_names)
        if not names:
            raise ValueError("class_names must be non-empty.")
        if semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive.")
        self.class_names = names
        self.values = nn.Parameter(torch.empty(len(names), semantic_dim))
        # A normalized Gaussian initialization keeps this ablation semantic-free
        # and avoids the LAPACK-backed QR decomposition used by orthogonal_.
        # Some cloud PyTorch builds intentionally omit CPU LAPACK support.
        with torch.no_grad():
            nn.init.normal_(self.values, mean=0.0, std=semantic_dim**-0.5)
            self.values.copy_(F.normalize(self.values, dim=-1))

    def forward(self) -> torch.Tensor:
        return F.normalize(self.values, dim=-1)

    @torch.no_grad()
    def freeze(
        self,
        version: str = "learned-after-initial-domain",
    ) -> FrozenPrototypeBank:
        return FrozenPrototypeBank(self(), self.class_names, version)
