"""Frozen-snapshot relation preservation for the legacy P1 baseline.

This module keeps the LLM/text prototype bank frozen and only updates the
specialist signal encoder. Targets are old-model relation snapshots over fault
semantic prototypes; they are not treated as a source of new semantic truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from fdllm_repro.models import ConvLSTMSignalEncoder, TrainHistory, prototype_alignment_loss


@dataclass(frozen=True)
class SemanticDistillBatch:
    x: np.ndarray
    y: np.ndarray
    replay_mask: np.ndarray
    replay_snapshot_probs: np.ndarray


def build_semantic_distill_batch(
    new_ds: dict[str, object],
    replay_ds: dict[str, object],
    replay_snapshot_probs: np.ndarray,
) -> SemanticDistillBatch:
    """Combine new-domain data with replay data and mark replay rows.

    `replay_snapshot_probs` must be aligned with `replay_ds`; those probabilities
    are produced by the previous frozen model before adaptation and act as
    historical relation targets during the continual update.
    """

    x_new = np.asarray(new_ds["x"], dtype=np.float32)
    y_new = np.asarray(new_ds["y"], dtype=np.int64)
    x_replay = np.asarray(replay_ds["x"], dtype=np.float32)
    y_replay = np.asarray(replay_ds["y"], dtype=np.int64)
    snapshot = np.asarray(replay_snapshot_probs, dtype=np.float32)
    if len(x_replay) != len(snapshot):
        raise ValueError("Replay samples and snapshot probabilities must have the same length.")

    x = np.concatenate([x_new, x_replay], axis=0)
    y = np.concatenate([y_new, y_replay], axis=0)
    replay_mask = np.concatenate(
        [np.zeros(len(x_new), dtype=bool), np.ones(len(x_replay), dtype=bool)],
        axis=0,
    )
    snapshot_full = np.zeros((len(x), snapshot.shape[1]), dtype=np.float32)
    snapshot_full[replay_mask] = snapshot
    return SemanticDistillBatch(
        x=x,
        y=y,
        replay_mask=replay_mask,
        replay_snapshot_probs=snapshot_full,
    )


def train_alignment_with_fse_distillation(
    encoder: ConvLSTMSignalEncoder,
    batch: SemanticDistillBatch,
    text_embeddings: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    alignment_temperature: float = 0.07,
    distill_temperature: float = 0.5,
    distill_weight: float = 0.5,
) -> list[TrainHistory]:
    """Train with prototype alignment plus replay FSE semantic distillation.

    Loss:

    L = L_align(new + replay labels) + lambda * KL(FSE_t0 || FSE_current)

    The KL term is computed only on replay rows, so new-domain plasticity is not
    directly constrained by old-domain soft labels.
    """

    encoder.to(device)
    text_bank = torch.tensor(text_embeddings, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(batch.x, dtype=torch.float32)
    y_tensor = torch.tensor(batch.y, dtype=torch.long)
    replay_mask_tensor = torch.tensor(batch.replay_mask, dtype=torch.bool)
    snapshot_tensor = torch.tensor(batch.replay_snapshot_probs, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor, replay_mask_tensor, snapshot_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    opt = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=1e-4)

    history: list[TrainHistory] = []
    for epoch in range(1, epochs + 1):
        encoder.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb, replay_mask, snapshot_probs in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            replay_mask = replay_mask.to(device)
            snapshot_probs = snapshot_probs.to(device)

            data_emb = encoder(xb)
            align_loss = prototype_alignment_loss(data_emb, text_bank, yb, alignment_temperature)
            loss = align_loss

            if replay_mask.any() and distill_weight > 0:
                replay_logits = data_emb[replay_mask] @ text_bank.T / distill_temperature
                log_student = F.log_softmax(replay_logits, dim=1)
                snapshot = snapshot_probs[replay_mask].clamp_min(1e-8)
                snapshot = snapshot / snapshot.sum(dim=1, keepdim=True)
                distill_loss = F.kl_div(log_student, snapshot, reduction="batchmean")
                loss = loss + distill_weight * distill_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                logits = data_emb @ text_bank.T
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
                total_loss += float(loss.item()) * int(yb.numel())

        history.append(TrainHistory(epoch, total_loss / total, correct / total))
    return history


def mean_kl_divergence(reference_probs: np.ndarray, current_probs: np.ndarray) -> float:
    """Mean KL(reference || current) for relation-snapshot drift auditing."""

    reference = np.asarray(reference_probs, dtype=np.float64)
    current = np.asarray(current_probs, dtype=np.float64)
    reference = reference / np.maximum(reference.sum(axis=1, keepdims=True), 1e-12)
    current = current / np.maximum(current.sum(axis=1, keepdims=True), 1e-12)
    kl = reference * (
        np.log(np.maximum(reference, 1e-12))
        - np.log(np.maximum(current, 1e-12))
    )
    return float(kl.sum(axis=1).mean())
