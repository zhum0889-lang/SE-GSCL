"""Hierarchical fuzzy identity mixtures over multiple text descriptions."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def hierarchical_fuzzy_identity(
    signal_embeddings: torch.Tensor,
    class_probabilities: torch.Tensor,
    description_prototypes: torch.Tensor,
    description_class_ids: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve class uncertainty and within-class text-facet uncertainty.

    Class probabilities retain the global diagnostic posterior. Within each
    class, cosine similarities distribute that class mass over its identity,
    mechanism, signature, condition, and disambiguation descriptions. This
    prevents classes with more descriptions from receiving a larger prior.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if signal_embeddings.ndim != 2 or description_prototypes.ndim != 2:
        raise ValueError("Embeddings and prototypes must be rank-2 tensors.")
    if signal_embeddings.shape[1] != description_prototypes.shape[1]:
        raise ValueError("Signal and description semantic dimensions must match.")
    if class_probabilities.ndim != 2:
        raise ValueError("class_probabilities must have shape [B,K].")
    if class_probabilities.shape[0] != signal_embeddings.shape[0]:
        raise ValueError("Signal and class-probability batch sizes must match.")
    if description_class_ids.shape != (description_prototypes.shape[0],):
        raise ValueError("description_class_ids must have shape [M].")
    num_classes = int(class_probabilities.shape[1])
    expected = list(range(num_classes))
    observed = sorted(set(int(value) for value in description_class_ids.tolist()))
    if observed != expected:
        raise ValueError(
            "Every class needs at least one description prototype; "
            f"got class ids {observed}."
        )

    signal = F.normalize(signal_embeddings, dim=-1)
    descriptions = F.normalize(description_prototypes, dim=-1)
    similarities = signal @ descriptions.T
    joint = torch.zeros_like(similarities)
    normalized_classes = class_probabilities / class_probabilities.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-8)
    for class_id in range(num_classes):
        mask = description_class_ids == class_id
        conditional = torch.softmax(similarities[:, mask] / temperature, dim=1)
        joint[:, mask] = conditional * normalized_classes[:, class_id : class_id + 1]
    fuzzy_embedding = F.normalize(joint @ descriptions, dim=-1)
    return joint, fuzzy_embedding
