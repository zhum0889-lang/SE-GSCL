"""Frozen old-model class-relation distributions for replay samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class GlobalRelationSnapshot:
    sample_ids: torch.Tensor
    probabilities: torch.Tensor
    version: str

    def __post_init__(self) -> None:
        if self.sample_ids.ndim != 1:
            raise ValueError("sample_ids must have shape [N].")
        if self.probabilities.ndim != 2:
            raise ValueError("probabilities must have shape [N,K].")
        if self.probabilities.shape[0] != self.sample_ids.shape[0]:
            raise ValueError("sample_ids and probabilities must have equal length.")

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            sample_ids=self.sample_ids.long().cpu().numpy(),
            probabilities=self.probabilities.float().cpu().numpy(),
            version=np.asarray(self.version),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "GlobalRelationSnapshot":
        values = np.load(path)
        return cls(
            sample_ids=torch.from_numpy(values["sample_ids"]).long(),
            probabilities=torch.from_numpy(values["probabilities"]).float(),
            version=str(values["version"].item()),
        )

    def lookup(self, sample_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        table = {
            int(sample_id): row
            for sample_id, row in zip(self.sample_ids.tolist(), self.probabilities)
        }
        mask = torch.tensor(
            [int(sample_id) in table for sample_id in sample_ids.tolist()],
            dtype=torch.bool,
            device=sample_ids.device,
        )
        if not torch.any(mask):
            empty = torch.empty(
                (0, self.probabilities.shape[1]),
                dtype=torch.float32,
                device=sample_ids.device,
            )
            return empty, mask
        rows = torch.stack(
            [table[int(sample_id)] for sample_id in sample_ids[mask].tolist()]
        ).to(sample_ids.device)
        return rows, mask
