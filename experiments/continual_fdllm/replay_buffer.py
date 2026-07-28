"""Metadata-only replay buffer for continual FD-LLM P0."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ReplayRecord:
    sample_id: int
    domain_id: int
    load: int
    true_label: int
    label_name: str
    file_id: str
    window_index: int
    predicted_label: int
    is_correct: bool
    fse_entropy: float
    top1_top2_margin: float
    replay_priority: float
    confusion_type: str
    snapshot_probs: list[float] | None = None
    selection_reason: str = ""
    prototype_version: str = ""


class ReplayBuffer:
    """Persistent, capacity-bounded replay metadata.

    Raw windows are addressed by stable ``sample_id``/file pointers. Updating a
    memory only combines its previous contents with current-domain candidates;
    callers must not rebuild it from all historical training data.
    """

    def __init__(
        self,
        records: Iterable[ReplayRecord | dict] | None = None,
        capacity: int | None = None,
        version: int = 0,
    ):
        self.capacity = None if capacity is None else max(0, int(capacity))
        self.version = int(version)
        self.records: list[ReplayRecord] = []
        self._index: dict[int, int] = {}
        if records:
            self.add_batch(records)

    def add_sample(self, record: ReplayRecord | dict) -> None:
        if isinstance(record, ReplayRecord):
            row = record
        else:
            payload = dict(record)
            if "snapshot_probs" not in payload and "teacher_probs" in payload:
                payload["snapshot_probs"] = payload.pop("teacher_probs")
            row = ReplayRecord(**payload)
        existing = self._index.get(int(row.sample_id))
        if existing is None:
            self._index[int(row.sample_id)] = len(self.records)
            self.records.append(row)
        else:
            self.records[existing] = row
        self._enforce_capacity()

    def add_batch(self, records: Iterable[ReplayRecord | dict]) -> None:
        for record in records:
            self.add_sample(record)

    def sample_random(self, n: int, seed: int = 42) -> list[ReplayRecord]:
        rng = random.Random(seed)
        rows = list(self.records)
        rng.shuffle(rows)
        return rows[: max(0, min(n, len(rows)))]

    def sample_by_priority(self, n: int, seed: int = 42) -> list[ReplayRecord]:
        rng = random.Random(seed)
        rows = sorted(
            self.records,
            key=lambda row: (float(row.replay_priority), rng.random()),
            reverse=True,
        )
        return rows[: max(0, min(n, len(rows)))]

    def replace(
        self,
        records: Iterable[ReplayRecord | dict],
        *,
        capacity: int | None = None,
        version: int | None = None,
    ) -> None:
        if capacity is not None:
            self.capacity = max(0, int(capacity))
        self.records = []
        self._index = {}
        for record in records:
            self.add_sample(record)
        if version is not None:
            self.version = int(version)
        self._enforce_capacity()

    def sample_ids(self) -> list[int]:
        return [int(row.sample_id) for row in self.records]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "capacity": self.capacity,
            "records": [asdict(row) for row in self.records],
            "summary": self.summarize(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            payload.get("records", []),
            capacity=payload.get("capacity"),
            version=int(payload.get("version", 0)),
        )

    def summarize(self) -> dict[str, object]:
        total = len(self.records)
        if total == 0:
            return {"total": 0}
        by_load: dict[str, int] = {}
        by_label: dict[str, int] = {}
        by_confusion: dict[str, int] = {}
        for row in self.records:
            by_load[str(row.load)] = by_load.get(str(row.load), 0) + 1
            by_label[row.label_name] = by_label.get(row.label_name, 0) + 1
            by_confusion[row.confusion_type] = by_confusion.get(row.confusion_type, 0) + 1
        return {
            "total": total,
            "capacity": self.capacity,
            "version": self.version,
            "correct": sum(1 for row in self.records if row.is_correct),
            "incorrect": sum(1 for row in self.records if not row.is_correct),
            "mean_fse_entropy": sum(row.fse_entropy for row in self.records) / total,
            "mean_top1_top2_margin": sum(row.top1_top2_margin for row in self.records) / total,
            "mean_replay_priority": sum(row.replay_priority for row in self.records) / total,
            "by_load": by_load,
            "by_label": by_label,
            "by_confusion": by_confusion,
        }

    def to_dataframe(self) -> list[dict[str, object]]:
        """Return CSV-friendly rows without requiring pandas."""

        return [asdict(row) for row in self.records]

    def _enforce_capacity(self) -> None:
        if self.capacity is None or len(self.records) <= self.capacity:
            self._reindex()
            return
        self.records = sorted(
            self.records,
            key=lambda row: (
                float(row.replay_priority),
                float(row.top1_top2_margin),
                -int(row.sample_id),
            ),
            reverse=True,
        )[: self.capacity]
        self._reindex()

    def _reindex(self) -> None:
        self._index = {int(row.sample_id): idx for idx, row in enumerate(self.records)}
