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
