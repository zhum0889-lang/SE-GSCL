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
    description_ids: tuple[str, ...] = ()
    description_types: tuple[str, ...] = ()
    class_summaries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [N,d].")
        if self.class_ids.shape != (self.embeddings.shape[0],):
            raise ValueError("class_ids must have shape [N].")
        if len(self.texts) != self.embeddings.shape[0]:
            raise ValueError("texts and embeddings must have equal length.")
        if self.description_ids and len(self.description_ids) != len(self.texts):
            raise ValueError("description_ids and texts must have equal length.")
        if self.description_types and len(self.description_types) != len(self.texts):
            raise ValueError("description_types and texts must have equal length.")
        if self.class_summaries and len(self.class_summaries) != len(
            self.class_names
        ):
            raise ValueError(
                "class_summaries and class_names must have equal length."
            )
        if self.description_ids and len(set(self.description_ids)) != len(
            self.description_ids
        ):
            raise ValueError("description_ids must be unique.")
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
            "description_ids": list(self.description_ids),
            "description_types": list(self.description_types),
            "class_summaries": list(self.class_summaries),
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
            description_ids=tuple(metadata.get("description_ids", ())),
            description_types=tuple(metadata.get("description_types", ())),
            class_summaries=tuple(metadata.get("class_summaries", ())),
        )


def read_ontology(path: str | Path) -> dict[str, Any]:
    ontology = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = sorted(ontology["classes"], key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in classes]
    if ids != list(range(len(classes))):
        raise ValueError("Ontology class ids must be contiguous from zero.")
    if any(not row.get("descriptions") for row in classes):
        raise ValueError("Each ontology class needs at least one description.")
    description_ids: set[str] = set()
    required_facets = set(ontology.get("required_description_types", ()))
    for row in classes:
        normalized = []
        observed_facets: set[str] = set()
        for index, description in enumerate(row["descriptions"]):
            if isinstance(description, str):
                item = {
                    "id": f"class_{int(row['id'])}_description_{index}",
                    "type": "general",
                    "text": description,
                }
            elif isinstance(description, dict):
                item = {
                    "id": str(description.get("id", "")).strip(),
                    "type": str(description.get("type", "")).strip(),
                    "text": str(description.get("text", "")).strip(),
                }
                if not all(item.values()):
                    raise ValueError(
                        "Structured descriptions require non-empty id, type, "
                        "and text fields."
                    )
            else:
                raise TypeError("Descriptions must be strings or objects.")
            if item["id"] in description_ids:
                raise ValueError(f"Duplicate description id: {item['id']}.")
            description_ids.add(item["id"])
            observed_facets.add(item["type"])
            normalized.append(item)
        missing = required_facets - observed_facets
        if missing:
            raise ValueError(
                f"Class {row['name']} is missing semantic facets: "
                f"{sorted(missing)}."
            )
        summary = str(row.get("prompt_summary", "")).strip()
        if ontology.get("require_prompt_summaries") and not summary:
            raise ValueError(f"Class {row['name']} needs a prompt_summary.")
        row["descriptions"] = normalized
        row["prompt_summary"] = summary
    ontology["classes"] = classes
    return ontology
