"""Metrics for square task-incremental accuracy matrices."""

from __future__ import annotations

import numpy as np


def summarize_accuracy_matrix(matrix: np.ndarray) -> dict[str, object]:
    """Summarize a stage-by-domain matrix with one newly seen domain per row."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("accuracy matrix must be square [num_stages,num_domains].")
    if not np.all(np.isfinite(values)):
        raise ValueError("accuracy matrix must contain finite values.")
    num_domains = values.shape[0]
    final = values[-1]
    learned = np.diag(values)
    forgetting = np.asarray(
        [
            float(np.max(values[domain_index:, domain_index]) - final[domain_index])
            for domain_index in range(num_domains)
        ]
    )
    backward_transfer = (
        np.asarray(
            [
                float(final[index] - learned[index])
                for index in range(num_domains - 1)
            ]
        )
        if num_domains > 1
        else np.zeros(1, dtype=np.float64)
    )
    stage_seen_averages = np.asarray(
        [
            float(np.mean(values[stage_index, : stage_index + 1]))
            for stage_index in range(num_domains)
        ]
    )
    old_learned = learned[:-1]
    old_final = final[:-1]
    retention = (
        np.divide(
            old_final,
            old_learned,
            out=np.zeros_like(old_final),
            where=np.abs(old_learned) > 1e-12,
        )
        if num_domains > 1
        else np.ones(1, dtype=np.float64)
    )
    return {
        "final_average_accuracy": float(np.mean(final)),
        "average_incremental_accuracy": float(np.mean(stage_seen_averages)),
        "average_forgetting": float(np.mean(forgetting[:-1]))
        if num_domains > 1
        else 0.0,
        "maximum_forgetting": float(np.max(forgetting[:-1]))
        if num_domains > 1
        else 0.0,
        "average_backward_transfer": float(np.mean(backward_transfer)),
        "average_old_domain_retention": float(np.mean(retention)),
        "final_by_domain": final.tolist(),
        "learned_by_domain": learned.tolist(),
        "forgetting_by_domain": forgetting.tolist(),
        "backward_transfer_by_old_domain": backward_transfer.tolist(),
        "retention_by_old_domain": retention.tolist(),
        "stage_seen_average": stage_seen_averages.tolist(),
    }
