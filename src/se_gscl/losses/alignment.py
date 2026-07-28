"""P1 global semantic alignment and cross-condition constraints."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def global_prototype_alignment_loss(
    fault_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if fault_embeddings.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("fault_embeddings and prototypes must be rank-2 tensors.")
    if fault_embeddings.shape[1] != prototypes.shape[1]:
        raise ValueError("fault_embeddings and prototypes must share semantic dimension.")
    normalized_embeddings = F.normalize(fault_embeddings, dim=-1)
    normalized_prototypes = F.normalize(prototypes, dim=-1)
    logits = normalized_embeddings @ normalized_prototypes.T / temperature
    return F.cross_entropy(logits, labels.long()), logits


def cross_condition_supervised_contrastive_loss(
    fault_embeddings: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrast same-label samples from different domains.

    Anchors without a same-label/different-domain positive are excluded from
    the outer mean, matching the normalized objective in the method section.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if fault_embeddings.ndim != 2:
        raise ValueError("fault_embeddings must have shape [B,d].")
    batch_size = int(fault_embeddings.shape[0])
    if batch_size < 2:
        return fault_embeddings.sum() * 0.0

    embeddings = F.normalize(fault_embeddings, dim=-1)
    similarities = embeddings @ embeddings.T / temperature
    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = (
        labels.view(-1, 1).eq(labels.view(1, -1))
        & domains.view(-1, 1).ne(domains.view(1, -1))
        & ~eye
    )
    valid = positive_mask.sum(dim=1) > 0
    if not torch.any(valid):
        return fault_embeddings.sum() * 0.0

    denominator_logits = similarities.masked_fill(eye, float("-inf"))
    log_denominator = torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    log_probabilities = similarities - log_denominator
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    anchor_losses = -(
        log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / positive_counts
    )
    return anchor_losses[valid].mean()


def cross_covariance_loss(
    fault_embeddings: torch.Tensor,
    condition_embeddings: torch.Tensor,
) -> torch.Tensor:
    if fault_embeddings.shape != condition_embeddings.shape:
        raise ValueError("fault and condition embeddings must have matching shape.")
    if fault_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [B,d].")
    batch_size = max(1, int(fault_embeddings.shape[0]))
    centered_fault = fault_embeddings - fault_embeddings.mean(dim=0, keepdim=True)
    centered_condition = condition_embeddings - condition_embeddings.mean(
        dim=0, keepdim=True
    )
    cross_covariance = centered_fault.T @ centered_condition / batch_size
    return cross_covariance.square().sum()
