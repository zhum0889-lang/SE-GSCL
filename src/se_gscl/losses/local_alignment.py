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
