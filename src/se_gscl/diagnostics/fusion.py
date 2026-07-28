"""Validation-calibrated reliability gating for global/local diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def fuse_probabilities(
    global_probabilities: np.ndarray,
    local_probabilities: np.ndarray,
    local_weights: float | np.ndarray,
) -> np.ndarray:
    global_values = np.clip(
        np.asarray(global_probabilities, dtype=np.float64),
        1e-12,
        1.0,
    )
    local_values = np.clip(
        np.asarray(local_probabilities, dtype=np.float64),
        1e-12,
        1.0,
    )
    if global_values.shape != local_values.shape:
        raise ValueError("Global and local probability shapes must match.")
    weights = np.asarray(local_weights, dtype=np.float64)
    if weights.ndim == 0:
        weights = np.full(len(global_values), float(weights))
    if weights.shape != (len(global_values),):
        raise ValueError("local_weights must be scalar or have shape [N].")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("local_weights must be in [0,1].")
    fused_log = (
        (1.0 - weights[:, None]) * np.log(global_values)
        + weights[:, None] * np.log(local_values)
    )
    fused = np.exp(fused_log - fused_log.max(axis=1, keepdims=True))
    return fused / fused.sum(axis=1, keepdims=True)


def branch_reliability(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-12,
        1.0,
    )
    values = values / values.sum(axis=1, keepdims=True)
    ordered = np.sort(values, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -(values * np.log(values)).sum(axis=1)
    normalized_entropy = entropy / np.log(values.shape[1])
    return margin * (1.0 - normalized_entropy)


@dataclass(frozen=True)
class ReliabilityGate:
    agreement_weight: float
    override_weight: float
    reliability_threshold: float
    validation_balanced_accuracy: float
    validation_accuracy: float

    def local_weights(
        self,
        global_probabilities: np.ndarray,
        local_probabilities: np.ndarray,
    ) -> np.ndarray:
        global_values = np.asarray(global_probabilities)
        local_values = np.asarray(local_probabilities)
        agreement = (
            global_values.argmax(axis=1) == local_values.argmax(axis=1)
        )
        global_reliability = branch_reliability(global_values)
        local_reliability = branch_reliability(local_values)
        override = (
            ~agreement
            & (
                local_reliability
                > global_reliability + self.reliability_threshold
            )
        )
        weights = np.zeros(len(global_values), dtype=np.float64)
        weights[agreement] = self.agreement_weight
        weights[override] = self.override_weight
        return weights

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_reliability_gate(
    global_probabilities: np.ndarray,
    local_probabilities: np.ndarray,
    labels: np.ndarray,
) -> ReliabilityGate:
    """Grid-search a conservative gate on the initial-domain validation set."""

    targets = np.asarray(labels, dtype=np.int64)
    num_classes = int(global_probabilities.shape[1])
    best: tuple[tuple[float, float, float], ReliabilityGate] | None = None
    for agreement_weight in (0.0, 0.25, 0.5):
        for override_weight in (0.5, 0.75, 1.0):
            for threshold in (
                -1.0,
                -0.10,
                0.0,
                0.05,
                0.10,
                0.20,
                1.0,
            ):
                candidate = ReliabilityGate(
                    agreement_weight=agreement_weight,
                    override_weight=override_weight,
                    reliability_threshold=threshold,
                    validation_balanced_accuracy=0.0,
                    validation_accuracy=0.0,
                )
                weights = candidate.local_weights(
                    global_probabilities,
                    local_probabilities,
                )
                fused = fuse_probabilities(
                    global_probabilities,
                    local_probabilities,
                    weights,
                )
                predictions = fused.argmax(axis=1)
                recalls = [
                    float(
                        np.mean(
                            predictions[targets == class_id] == class_id
                        )
                    )
                    for class_id in range(num_classes)
                    if np.any(targets == class_id)
                ]
                balanced_accuracy = float(np.mean(recalls))
                accuracy = float(np.mean(predictions == targets))
                calibrated = ReliabilityGate(
                    agreement_weight=agreement_weight,
                    override_weight=override_weight,
                    reliability_threshold=threshold,
                    validation_balanced_accuracy=balanced_accuracy,
                    validation_accuracy=accuracy,
                )
                key = (
                    balanced_accuracy,
                    accuracy,
                    -float(np.mean(weights)),
                )
                if best is None or key > best[0]:
                    best = (key, calibrated)
    if best is None:
        raise ValueError("Could not fit reliability gate.")
    return best[1]
