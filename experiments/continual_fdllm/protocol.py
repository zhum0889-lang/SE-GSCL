"""Leakage-resistant continual-learning protocol utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

import numpy as np

from experiments.continual_fdllm.replay_buffer import ReplayBuffer, ReplayRecord


def update_persistent_memory(
    previous: ReplayBuffer,
    current_candidates: Iterable[ReplayRecord],
    capacity: int,
    episode: int,
    seed: int,
    strategy: str,
) -> ReplayBuffer:
    """Update memory from ``previous memory + current domain`` only."""

    candidates_by_id = {int(row.sample_id): row for row in previous.records}
    for row in current_candidates:
        candidates_by_id[int(row.sample_id)] = row
    candidates = list(candidates_by_id.values())

    if strategy == "random_replay":
        rng = np.random.default_rng(seed + episode * 1009)
        order = rng.permutation(len(candidates)).tolist()
        selected = [replace(candidates[i], selection_reason="random_reservoir") for i in order[:capacity]]
    elif strategy == "balanced_semantic_replay":
        selected = _balanced_semantic_selection(candidates, capacity, seed + episode * 1009)
    else:
        raise ValueError(f"Unsupported persistent-memory strategy: {strategy}")

    memory = ReplayBuffer(capacity=capacity, version=episode + 1)
    memory.replace(selected, capacity=capacity, version=episode + 1)
    return memory


def compute_sequential_metrics(
    strategy: str,
    domain_order: list[int],
    matrix: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Compute ACC, learning accuracy, and forgetting from a seen-only matrix."""

    final_seen = matrix[-1, : len(domain_order)]
    diagonal = np.asarray([matrix[i, i] for i in range(len(domain_order))], dtype=np.float32)
    forgetting_terms: list[float] = []
    for domain_idx in range(len(domain_order) - 1):
        history = matrix[domain_idx:, domain_idx]
        best = float(np.nanmax(history))
        forgetting_terms.append(max(0.0, best - float(matrix[-1, domain_idx])))
    return {
        "seed": int(seed),
        "strategy": strategy,
        "domain_order": "->".join(str(v) for v in domain_order),
        "ACC": float(np.nanmean(final_seen)),
        "LA": float(np.nanmean(diagonal)),
        "FM": float(np.mean(forgetting_terms)) if forgetting_terms else 0.0,
        "final_domain_accuracies_json": json.dumps(
            {str(domain_order[i]): float(final_seen[i]) for i in range(len(domain_order))}
        ),
    }


def assert_seen_only_matrix(matrix: np.ndarray) -> None:
    """Ensure future-domain cells were not evaluated during earlier episodes."""

    for row in range(matrix.shape[0]):
        if np.isfinite(matrix[row, row + 1 :]).any():
            raise AssertionError(f"Accuracy matrix row {row} contains future-domain evaluations.")


def _balanced_semantic_selection(
    candidates: list[ReplayRecord],
    capacity: int,
    seed: int,
) -> list[ReplayRecord]:
    if capacity <= 0 or not candidates:
        return []
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int], list[ReplayRecord]] = defaultdict(list)
    for row in candidates:
        groups[(int(row.domain_id), int(row.true_label))].append(row)

    selected: list[ReplayRecord] = []
    group_keys = sorted(groups)
    base_quota = max(1, capacity // max(1, len(group_keys)))
    for key in group_keys:
        rows = groups[key]
        quota = min(base_quota, len(rows))
        selected.extend(_select_group(rows, quota, rng))

    if len(selected) < capacity:
        used = {int(row.sample_id) for row in selected}
        remaining = [row for row in candidates if int(row.sample_id) not in used]
        remaining.sort(
            key=lambda row: (
                float(row.replay_priority),
                -float(row.fse_entropy),
                float(row.top1_top2_margin),
            ),
            reverse=True,
        )
        selected.extend(replace(row, selection_reason="capacity_fill") for row in remaining[: capacity - len(selected)])
    return selected[:capacity]


def _select_group(
    rows: list[ReplayRecord],
    quota: int,
    rng: np.random.Generator,
) -> list[ReplayRecord]:
    if quota <= 0:
        return []
    n_stable = int(round(quota * 0.4))
    n_hard = int(round(quota * 0.4))
    n_diverse = max(0, quota - n_stable - n_hard)

    stable = sorted(
        [row for row in rows if row.is_correct],
        key=lambda row: (
            float(row.top1_top2_margin),
            -float(row.fse_entropy),
            -int(row.sample_id),
        ),
        reverse=True,
    )
    hard = sorted(
        rows,
        key=lambda row: (float(row.replay_priority), float(row.fse_entropy), -int(row.sample_id)),
        reverse=True,
    )

    chosen: list[ReplayRecord] = []
    used: set[int] = set()
    _extend_unique(chosen, used, stable, n_stable, "stable_anchor")
    _extend_unique(chosen, used, hard, n_hard, "hard_boundary")

    diverse_pool = [row for row in rows if int(row.sample_id) not in used]
    if diverse_pool:
        order = rng.permutation(len(diverse_pool)).tolist()
        shuffled = [diverse_pool[i] for i in order]
        _extend_unique(chosen, used, shuffled, n_diverse, "diverse")

    if len(chosen) < quota:
        fallback = sorted(rows, key=lambda row: (float(row.replay_priority), -int(row.sample_id)), reverse=True)
        _extend_unique(chosen, used, fallback, quota - len(chosen), "group_fill")
    return chosen[:quota]


def _extend_unique(
    output: list[ReplayRecord],
    used: set[int],
    candidates: Iterable[ReplayRecord],
    n: int,
    reason: str,
) -> None:
    if n <= 0:
        return
    added = 0
    for row in candidates:
        sample_id = int(row.sample_id)
        if sample_id in used:
            continue
        output.append(replace(row, selection_reason=reason))
        used.add(sample_id)
        added += 1
        if added >= n:
            break
