"""P1 global-semantic training for the lightweight signal specialist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from se_gscl.losses import (
    cross_condition_supervised_contrastive_loss,
    cross_covariance_loss,
    global_prototype_alignment_loss,
    global_relation_snapshot_loss,
)


class PrototypeSource(Protocol):
    def __call__(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class P1LossWeights:
    classification: float = 0.0
    global_alignment: float = 1.0
    cross_condition: float = 0.0
    decorrelation: float = 0.0
    global_relation: float = 0.0


class GlobalSemanticTrainer:
    def __init__(
        self,
        model: nn.Module,
        prototype_source: PrototypeSource | torch.Tensor,
        weights: P1LossWeights,
        *,
        device: str | torch.device = "cpu",
        temperature: float = 0.07,
        snapshot_temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.prototype_source = prototype_source
        self.weights = weights
        self.device = torch.device(device)
        self.temperature = float(temperature)
        self.snapshot_temperature = float(snapshot_temperature)
        self.model.to(self.device)

    def prototypes(self) -> torch.Tensor:
        source = self.prototype_source
        values = source() if callable(source) else source
        return values.to(self.device)

    def train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        self.model.train()
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "global_alignment": 0.0,
            "cross_condition": 0.0,
            "decorrelation": 0.0,
            "global_relation": 0.0,
        }
        steps = 0
        for batch in loader:
            x = batch["x"].to(self.device)
            labels = batch["label"].long().to(self.device)
            domains = batch["domain"].long().to(self.device)
            optimizer.zero_grad(set_to_none=True)
            output = self.model(x)
            alignment, logits = global_prototype_alignment_loss(
                output.fault_embedding,
                self.prototypes(),
                labels,
                temperature=self.temperature,
            )
            classification = logits.sum() * 0.0
            relation_logits = logits
            if output.class_logits is not None:
                classification = F.cross_entropy(output.class_logits, labels)
                relation_logits = output.class_logits
            elif self.weights.classification > 0.0:
                raise RuntimeError(
                    "Classification loss requires a specialist classifier head."
                )
            cross_condition = cross_condition_supervised_contrastive_loss(
                output.fault_embedding,
                labels,
                domains,
                temperature=self.temperature,
            )
            decorrelation = cross_covariance_loss(
                output.fault_embedding,
                output.condition_embedding,
            )
            relation = logits.sum() * 0.0
            if "snapshot_probs" in batch:
                replay_mask = batch.get("is_replay")
                if replay_mask is not None:
                    replay_mask = replay_mask.bool().to(self.device)
                relation = global_relation_snapshot_loss(
                    relation_logits,
                    batch["snapshot_probs"].to(self.device),
                    replay_mask,
                    temperature=self.snapshot_temperature,
                )
            loss = (
                self.weights.classification * classification
                + self.weights.global_alignment * alignment
                + self.weights.cross_condition * cross_condition
                + self.weights.decorrelation * decorrelation
                + self.weights.global_relation * relation
            )
            loss.backward()
            optimizer.step()
            values = {
                "loss": loss,
                "classification": classification,
                "global_alignment": alignment,
                "cross_condition": cross_condition,
                "decorrelation": decorrelation,
                "global_relation": relation,
            }
            for key, value in values.items():
                totals[key] += float(value.detach().cpu())
            steps += 1
        if steps == 0:
            raise ValueError("Training loader produced no batches.")
        return {key: value / steps for key, value in totals.items()}
