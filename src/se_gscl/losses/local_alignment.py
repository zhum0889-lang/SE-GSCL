"""Weakly supervised local symptom alignment."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from se_gscl.models.local_symptom import LocalSymptomOutput


def local_symptom_alignment_loss(
    output: LocalSymptomOutput,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if output.class_scores.ndim != 2:
        raise ValueError("class_scores must have shape [B,K].")
    if labels.shape != (output.class_scores.shape[0],):
        raise ValueError("labels must have shape [B].")
    return F.cross_entropy(output.class_scores / temperature, labels.long())


def physics_guided_local_alignment_loss(
    output: LocalSymptomOutput,
    labels: torch.Tensor,
    symptom_targets: torch.Tensor,
    symptom_target_weights: torch.Tensor,
    anchor_prototypes: torch.Tensor,
    *,
    temperature: float = 0.1,
    physics_weight: float = 1.0,
    anchor_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint class, physical-attribute, and semantic-anchor supervision."""

    if symptom_targets.shape != output.symptom_scores.shape:
        raise ValueError("symptom_targets must match symptom_scores.")
    if symptom_target_weights.shape != symptom_targets.shape:
        raise ValueError("symptom_target_weights must match targets.")
    if anchor_prototypes.shape != output.symptom_prototypes.shape:
        raise ValueError("anchor_prototypes must match current prototypes.")
    class_loss = local_symptom_alignment_loss(
        output,
        labels,
        temperature=temperature,
    )
    elementwise = F.binary_cross_entropy_with_logits(
        output.symptom_scores / temperature,
        symptom_targets,
        reduction="none",
    )
    denominator = symptom_target_weights.sum().clamp_min(1.0)
    physics_loss = (elementwise * symptom_target_weights).sum() / denominator
    anchor_loss = (
        1.0
        - F.cosine_similarity(
            output.symptom_prototypes,
            anchor_prototypes,
            dim=-1,
        )
    ).mean()
    total = (
        class_loss
        + float(physics_weight) * physics_loss
        + float(anchor_weight) * anchor_loss
    )
    return total, {
        "classification": class_loss,
        "physics": physics_loss,
        "prototype_anchor": anchor_loss,
    }
