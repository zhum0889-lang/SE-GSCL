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
    ranking_weight: float = 0.0,
    ranking_temperature: float = 0.2,
    symptom_class_ids: torch.Tensor | None = None,
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
    ranking_loss = torch.zeros((), device=class_loss.device)
    if ranking_weight > 0:
        if symptom_class_ids is None:
            raise ValueError(
                "symptom_class_ids is required when ranking_weight > 0."
            )
        ranking_loss = within_class_symptom_distribution_loss(
            output.symptom_scores,
            symptom_targets,
            symptom_target_weights,
            labels,
            symptom_class_ids,
            temperature=ranking_temperature,
        )
    total = (
        class_loss
        + float(physics_weight) * physics_loss
        + float(anchor_weight) * anchor_loss
        + float(ranking_weight) * ranking_loss
    )
    return total, {
        "classification": class_loss,
        "physics": physics_loss,
        "prototype_anchor": anchor_loss,
        "within_class_distribution": ranking_loss,
    }


def within_class_symptom_distribution_loss(
    symptom_scores: torch.Tensor,
    symptom_targets: torch.Tensor,
    symptom_target_weights: torch.Tensor,
    labels: torch.Tensor,
    symptom_class_ids: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Align relative symptom strengths within the ground-truth fault class."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if symptom_scores.shape != symptom_targets.shape:
        raise ValueError("symptom_scores and targets must have equal shape.")
    if symptom_target_weights.shape != symptom_targets.shape:
        raise ValueError("symptom_target_weights must match targets.")
    if symptom_class_ids.shape != (symptom_scores.shape[1],):
        raise ValueError("symptom_class_ids must have shape [R].")
    losses: list[torch.Tensor] = []
    for class_id in torch.unique(labels).tolist():
        sample_mask = labels == int(class_id)
        symptom_mask = symptom_class_ids == int(class_id)
        predicted = symptom_scores[sample_mask][:, symptom_mask]
        targets = symptom_targets[sample_mask][:, symptom_mask]
        weights = symptom_target_weights[sample_mask][:, symptom_mask]
        if predicted.shape[1] <= 1:
            continue
        valid = weights > 0
        target_logits = targets / temperature
        target_logits = target_logits.masked_fill(~valid, -1e4)
        target_distribution = torch.softmax(target_logits, dim=1)
        predicted_log_distribution = torch.log_softmax(
            predicted / temperature,
            dim=1,
        )
        sample_valid = valid.any(dim=1)
        if sample_valid.any():
            losses.append(
                F.kl_div(
                    predicted_log_distribution[sample_valid],
                    target_distribution[sample_valid],
                    reduction="batchmean",
                )
            )
    if not losses:
        return symptom_scores.sum() * 0.0
    return torch.stack(losses).mean()
