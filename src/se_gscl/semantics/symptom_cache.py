"""Serializable frozen text embeddings for local fault symptoms."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SymptomEmbeddingCache:
    embeddings: torch.Tensor
    class_ids: torch.Tensor
    class_names: tuple[str, ...]
    symptom_ids: tuple[str, ...]
    symptom_names: tuple[str, ...]
    texts: tuple[str, ...]
    model_id: str
    ontology: str
    version: str
    pooling: str = "masked_mean"

    def __post_init__(self) -> None:
        count = int(self.embeddings.shape[0])
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [R,d].")
        if self.class_ids.shape != (count,):
            raise ValueError("class_ids must have shape [R].")
        if not (
            len(self.symptom_ids)
            == len(self.symptom_names)
            == len(self.texts)
            == count
        ):
            raise ValueError("Symptom metadata and embeddings must have equal length.")
        if len(set(self.symptom_ids)) != count:
            raise ValueError("symptom_ids must be unique.")
        expected = list(range(len(self.class_names)))
        observed = sorted(set(int(value) for value in self.class_ids.tolist()))
        if observed != expected:
            raise ValueError(
                f"Every class must have at least one symptom; got class ids {observed}."
            )

    @property
    def hidden_size(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def num_symptoms(self) -> int:
        return int(self.embeddings.shape[0])

    def save(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "symptom_embeddings.npz",
            embeddings=self.embeddings.float().cpu().numpy(),
            class_ids=self.class_ids.long().cpu().numpy(),
        )
        metadata = {
            "model_id": self.model_id,
            "ontology": self.ontology,
            "version": self.version,
            "pooling": self.pooling,
            "hidden_size": self.hidden_size,
            "class_names": list(self.class_names),
            "symptom_ids": list(self.symptom_ids),
            "symptom_names": list(self.symptom_names),
            "texts": list(self.texts),
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return root

    @classmethod
    def load(cls, cache_dir: str | Path) -> "SymptomEmbeddingCache":
        root = Path(cache_dir)
        arrays = np.load(root / "symptom_embeddings.npz")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            embeddings=torch.from_numpy(arrays["embeddings"]).float(),
            class_ids=torch.from_numpy(arrays["class_ids"]).long(),
            class_names=tuple(metadata["class_names"]),
            symptom_ids=tuple(metadata["symptom_ids"]),
            symptom_names=tuple(metadata["symptom_names"]),
            texts=tuple(metadata["texts"]),
            model_id=str(metadata["model_id"]),
            ontology=str(metadata["ontology"]),
            version=str(metadata["version"]),
            pooling=str(metadata.get("pooling", "masked_mean")),
        )
