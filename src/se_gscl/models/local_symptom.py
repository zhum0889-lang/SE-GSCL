"""Token-to-symptom matching for hierarchical local semantic evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from se_gscl.semantics import FrozenSymptomPrototypeBank


@dataclass
class LocalSymptomOutput:
    symptom_prototypes: torch.Tensor
    token_similarities: torch.Tensor
    symptom_scores: torch.Tensor
    symptom_weights: torch.Tensor
    class_scores: torch.Tensor
    class_probabilities: torch.Tensor
    conditional_symptom_probabilities: torch.Tensor
    joint_probabilities: torch.Tensor
    fuzzy_symptom_embedding: torch.Tensor


class LocalSymptomMatcher(nn.Module):
    """Aggregate strongest local token responses into class evidence."""

    def __init__(
        self,
        bank: FrozenSymptomPrototypeBank,
        *,
        top_tokens: int = 4,
        temperature: float = 0.1,
        learnable_symptom_weights: bool = False,
    ) -> None:
        super().__init__()
        if top_tokens <= 0:
            raise ValueError("top_tokens must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.bank = bank
        self.top_tokens = int(top_tokens)
        self.temperature = float(temperature)
        semantic_dim = bank.semantic_dim
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, semantic_dim, bias=False),
        )
        nn.init.eye_(self.token_adapter[1].weight)
        if learnable_symptom_weights:
            self.symptom_weight_logits = nn.Parameter(
                torch.zeros(bank.num_symptoms)
            )
        else:
            self.register_parameter("symptom_weight_logits", None)

    def _class_symptom_weights(
        self,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.zeros(
            self.bank.num_symptoms,
            device=reference.device,
            dtype=reference.dtype,
        )
        for class_id in range(self.bank.num_classes):
            mask = self.bank.class_ids == class_id
            count = int(mask.sum())
            if self.symptom_weight_logits is None:
                weights[mask] = 1.0 / count
            else:
                weights[mask] = torch.softmax(
                    self.symptom_weight_logits[mask],
                    dim=0,
                )
        return weights

    def forward(self, fault_tokens: torch.Tensor) -> LocalSymptomOutput:
        if fault_tokens.ndim != 3:
            raise ValueError("fault_tokens must have shape [B,N,d].")
        if fault_tokens.shape[-1] != self.bank.semantic_dim:
            raise ValueError(
                f"Expected token dim {self.bank.semantic_dim}, "
                f"got {fault_tokens.shape[-1]}."
            )
        tokens = F.normalize(self.token_adapter(fault_tokens), dim=-1)
        prototypes = self.bank()
        token_similarities = torch.einsum(
            "bnd,rd->bnr",
            tokens,
            prototypes,
        )
        top_count = min(self.top_tokens, int(fault_tokens.shape[1]))
        symptom_scores = token_similarities.topk(
            top_count,
            dim=1,
        ).values.mean(dim=1)
        symptom_weights = self._class_symptom_weights(prototypes)
        class_scores = torch.stack(
            [
                (
                    symptom_scores[:, self.bank.class_ids == class_id]
                    * symptom_weights[self.bank.class_ids == class_id]
                ).sum(dim=1)
                for class_id in range(self.bank.num_classes)
            ],
            dim=1,
        )
        class_probabilities = torch.softmax(
            class_scores / self.temperature,
            dim=1,
        )
        conditional = torch.zeros_like(symptom_scores)
        for class_id in range(self.bank.num_classes):
            mask = self.bank.class_ids == class_id
            conditional[:, mask] = torch.softmax(
                symptom_scores[:, mask] / self.temperature,
                dim=1,
            )
        joint = conditional * class_probabilities[:, self.bank.class_ids]
        fuzzy_embedding = F.normalize(
            joint @ prototypes,
            dim=-1,
        )
        return LocalSymptomOutput(
            symptom_prototypes=prototypes,
            token_similarities=token_similarities,
            symptom_scores=symptom_scores,
            symptom_weights=symptom_weights,
            class_scores=class_scores,
            class_probabilities=class_probabilities,
            conditional_symptom_probabilities=conditional,
            joint_probabilities=joint,
            fuzzy_symptom_embedding=fuzzy_embedding,
        )
