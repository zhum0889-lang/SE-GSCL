"""Update-time preservation of old sample-to-prototype relations."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def snapshot_probabilities(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return F.softmax(logits.detach() / temperature, dim=-1)


def global_relation_snapshot_loss(
    current_logits: torch.Tensor,
    snapshot_probs: torch.Tensor,
    replay_mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL-match current class relations to frozen pre-update snapshots."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if current_logits.ndim != 2:
        raise ValueError("current_logits must have shape [B,K].")
    if replay_mask is not None:
        if replay_mask.shape != (current_logits.shape[0],):
            raise ValueError("replay_mask must have shape [B].")
        if snapshot_probs.shape == current_logits.shape:
            targets = snapshot_probs[replay_mask]
        else:
            targets = snapshot_probs
        current = current_logits[replay_mask]
    else:
        current = current_logits
        targets = snapshot_probs
    if current.shape != targets.shape:
        raise ValueError("Selected current logits and snapshot probabilities must match.")
    if current.shape[0] == 0:
        return current_logits.sum() * 0.0
    targets = targets.detach().clamp_min(1e-8)
    targets = targets / targets.sum(dim=-1, keepdim=True)
    log_probs = F.log_softmax(current / temperature, dim=-1)
    return F.kl_div(log_probs, targets, reduction="batchmean") * temperature**2
