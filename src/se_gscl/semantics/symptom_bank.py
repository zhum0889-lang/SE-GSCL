"""Projected and frozen local symptom prototype banks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .symptom_cache import SymptomEmbeddingCache


class FrozenSymptomPrototypeBank(nn.Module):
    def __init__(
        self,
        prototypes: torch.Tensor,
        class_ids: torch.Tensor,
        class_names: Sequence[str],
        symptom_ids: Sequence[str],
        symptom_names: Sequence[str],
        physics_keys: Sequence[str],
        version: str,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(prototypes, dtype=torch.float32)
        ids = torch.as_tensor(class_ids, dtype=torch.long)
        if values.ndim != 2:
            raise ValueError("prototypes must have shape [R,d].")
        if ids.shape != (values.shape[0],):
            raise ValueError("class_ids must have shape [R].")
        if not (
            len(symptom_ids)
            == len(symptom_names)
            == len(physics_keys)
            == values.shape[0]
        ):
            raise ValueError("Symptom metadata must match prototype count.")
        observed = sorted(set(int(value) for value in ids.tolist()))
        if observed != list(range(len(class_names))):
            raise ValueError("Every class must be represented in class_ids.")
        self.register_buffer("prototypes", F.normalize(values, dim=-1))
        self.register_buffer("class_ids", ids)
        self.class_names = tuple(str(value) for value in class_names)
        self.symptom_ids = tuple(str(value) for value in symptom_ids)
        self.symptom_names = tuple(str(value) for value in symptom_names)
        self.physics_keys = tuple(str(value) for value in physics_keys)
        self.version = str(version)

    @property
    def semantic_dim(self) -> int:
        return int(self.prototypes.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_symptoms(self) -> int:
        return int(self.prototypes.shape[0])

    def forward(self) -> torch.Tensor:
        return self.prototypes


class ResidualSymptomPrototypeBank(nn.Module):
    """Adapt frozen symptom prototypes with a norm-bounded residual."""

    def __init__(
        self,
        base: FrozenSymptomPrototypeBank,
        *,
        max_residual_scale: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0.0 < max_residual_scale < 1.0:
            raise ValueError("max_residual_scale must be in (0,1).")
        self.register_buffer("base_prototypes", base.prototypes.detach().clone())
        self.register_buffer("class_ids", base.class_ids.detach().clone())
        self.residual = nn.Parameter(torch.zeros_like(self.base_prototypes))
        self.max_residual_scale = float(max_residual_scale)
        self.class_names = base.class_names
        self.symptom_ids = base.symptom_ids
        self.symptom_names = base.symptom_names
        self.physics_keys = base.physics_keys
        self.version = f"{base.version}-bounded-residual"

    @property
    def semantic_dim(self) -> int:
        return int(self.base_prototypes.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_symptoms(self) -> int:
        return int(self.base_prototypes.shape[0])

    def forward(self) -> torch.Tensor:
        residual_norm = self.residual.norm(dim=-1, keepdim=True)
        bounded = self.residual / (1.0 + residual_norm)
        return F.normalize(
            self.base_prototypes + self.max_residual_scale * bounded,
            dim=-1,
        )


class ProjectedSymptomPrototypeBank(nn.Module):
    """Project symptom descriptions with the P1 global text projector."""

    def __init__(
        self,
        cache: SymptomEmbeddingCache,
        projection: nn.Module,
        text_center: torch.Tensor,
    ) -> None:
        super().__init__()
        center = torch.as_tensor(text_center, dtype=torch.float32)
        if center.shape != (1, cache.hidden_size):
            raise ValueError(
                f"text_center must have shape [1,{cache.hidden_size}], "
                f"got {tuple(center.shape)}."
            )
        self.register_buffer("text_embeddings", cache.embeddings.float())
        self.register_buffer("class_ids", cache.class_ids.long())
        self.register_buffer("text_center", center)
        self.projection = projection
        self.class_names = cache.class_names
        self.symptom_ids = cache.symptom_ids
        self.symptom_names = cache.symptom_names
        self.physics_keys = cache.physics_keys
        self.version = cache.version

    @property
    def semantic_dim(self) -> int:
        linear = next(
            module
            for module in reversed(list(self.projection.modules()))
            if isinstance(module, nn.Linear)
        )
        return int(linear.out_features)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_symptoms(self) -> int:
        return int(self.text_embeddings.shape[0])

    def forward(self) -> torch.Tensor:
        centered = self.text_embeddings - self.text_center
        return F.normalize(self.projection(centered), dim=-1)

    @torch.no_grad()
    def freeze(self, version: str | None = None) -> FrozenSymptomPrototypeBank:
        return FrozenSymptomPrototypeBank(
            self(),
            self.class_ids,
            self.class_names,
            self.symptom_ids,
            self.symptom_names,
            self.physics_keys,
            version or self.version,
        )
