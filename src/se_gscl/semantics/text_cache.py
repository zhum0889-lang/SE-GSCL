"""Serializable frozen text embeddings and ontology metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TextEmbeddingCache:
    embeddings: torch.Tensor
    class_ids: torch.Tensor
    class_names: tuple[str, ...]
    texts: tuple[str, ...]
    model_id: str
    ontology: str
    version: str
    pooling: str = "masked_mean"

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [N,d].")
        if self.class_ids.shape != (self.embeddings.shape[0],):
            raise ValueError("class_ids must have shape [N].")
        if len(self.texts) != self.embeddings.shape[0]:
            raise ValueError("texts and embeddings must have equal length.")
        expected = list(range(len(self.class_names)))
        observed = sorted(set(int(value) for value in self.class_ids.tolist()))
        if observed != expected:
            raise ValueError(
                f"class ids must be contiguous and represented; got {observed}."
            )

    @property
    def hidden_size(self) -> int:
        return int(self.embeddings.shape[1])

    def class_means(self) -> torch.Tensor:
        rows = [
            self.embeddings[self.class_ids == class_id].mean(dim=0)
            for class_id in range(len(self.class_names))
        ]
        return torch.stack(rows, dim=0)

    def save(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "text_embeddings.npz",
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
            "texts": list(self.texts),
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return root

    @classmethod
    def load(cls, cache_dir: str | Path) -> "TextEmbeddingCache":
        root = Path(cache_dir)
        arrays = np.load(root / "text_embeddings.npz")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            embeddings=torch.from_numpy(arrays["embeddings"]).float(),
            class_ids=torch.from_numpy(arrays["class_ids"]).long(),
            class_names=tuple(metadata["class_names"]),
            texts=tuple(metadata["texts"]),
            model_id=str(metadata["model_id"]),
            ontology=str(metadata["ontology"]),
            version=str(metadata["version"]),
            pooling=str(metadata.get("pooling", "masked_mean")),
        )


def read_ontology(path: str | Path) -> dict[str, Any]:
    ontology = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = sorted(ontology["classes"], key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in classes]
    if ids != list(range(len(classes))):
        raise ValueError("Ontology class ids must be contiguous from zero.")
    if any(not row.get("descriptions") for row in classes):
        raise ValueError("Each ontology class needs at least one description.")
    ontology["classes"] = classes
    return ontology
