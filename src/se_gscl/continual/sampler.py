"""Batch sampling that guarantees cross-condition positives when available."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class ClassDomainBatchSampler(Sampler[list[int]]):
    """Construct batches from the same class under at least two conditions."""

    def __init__(
        self,
        labels: np.ndarray,
        domains: np.ndarray,
        batch_size: int,
        *,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.domains = np.asarray(domains, dtype=np.int64)
        if self.labels.shape != self.domains.shape or self.labels.ndim != 1:
            raise ValueError("labels and domains must be matching one-dimensional arrays.")
        if batch_size < 2:
            raise ValueError("batch_size must be at least two.")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.groups: dict[tuple[int, int], np.ndarray] = {}
        domains_by_class: dict[int, list[int]] = defaultdict(list)
        for label in sorted(np.unique(self.labels).tolist()):
            for domain in sorted(np.unique(self.domains[self.labels == label]).tolist()):
                indices = np.flatnonzero(
                    (self.labels == int(label)) & (self.domains == int(domain))
                )
                if len(indices):
                    self.groups[(int(label), int(domain))] = indices
                    domains_by_class[int(label)].append(int(domain))
        self.domains_by_class = {
            label: tuple(values)
            for label, values in domains_by_class.items()
            if len(values) >= 2
        }
        if not self.domains_by_class:
            raise ValueError(
                "No class occurs in at least two domains; cross-condition positives "
                "cannot be constructed."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        viable_labels = np.asarray(sorted(self.domains_by_class), dtype=np.int64)
        all_indices = np.arange(len(self.labels), dtype=np.int64)
        for _ in range(len(self)):
            selected: list[int] = []
            used: set[int] = set()
            label_order = rng.permutation(viable_labels)
            label_cursor = 0
            while len(selected) + 2 <= self.batch_size:
                label = int(label_order[label_cursor % len(label_order)])
                label_cursor += 1
                domain_pair = rng.choice(
                    np.asarray(self.domains_by_class[label], dtype=np.int64),
                    size=2,
                    replace=False,
                )
                pair: list[int] = []
                for domain in domain_pair:
                    candidates = self.groups[(label, int(domain))]
                    unused = candidates[
                        np.asarray([int(index) not in used for index in candidates])
                    ]
                    source = unused if len(unused) else candidates
                    pair.append(int(rng.choice(source)))
                for index in pair:
                    if index not in used:
                        selected.append(index)
                        used.add(index)
                if label_cursor >= len(label_order) and len(selected) >= 2:
                    break

            fill_pool = all_indices[
                np.asarray([int(index) not in used for index in all_indices])
            ]
            rng.shuffle(fill_pool)
            selected.extend(
                int(index)
                for index in fill_pool[: self.batch_size - len(selected)]
            )
            if len(selected) < self.batch_size and self.drop_last:
                continue
            rng.shuffle(selected)
            yield selected
