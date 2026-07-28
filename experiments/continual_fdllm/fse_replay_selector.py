"""FSE-guided replay selection utilities."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from experiments.continual_fdllm.replay_buffer import ReplayBuffer, ReplayRecord


def compute_fse_entropy(probs: np.ndarray, normalize: bool = True) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    if normalize and probs.shape[1] > 1:
        entropy = entropy / math.log(probs.shape[1])
    return entropy.astype(np.float32)


def compute_top_margin(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    sorted_probs = np.sort(probs, axis=1)
    if sorted_probs.shape[1] == 1:
        return sorted_probs[:, -1].astype(np.float32)
    return (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)


def detect_fault_size_confusion(true_name: str, pred_name: str) -> bool:
    true_size = _fault_size(true_name)
    pred_size = _fault_size(pred_name)
    if true_size == "Normal" or pred_size == "Normal":
        return False
    same_location = _fault_location(true_name) == _fault_location(pred_name)
    same_bearing = _bearing_side(true_name) == _bearing_side(pred_name)
    return same_location and same_bearing and true_size != pred_size


def detect_confusion_class(true_name: str, pred_name: str) -> bool:
    if true_name == pred_name:
        return False
    if _fault_location(true_name) == "Ball" and _fault_location(pred_name) == "Ball":
        return True
    if _bearing_side(true_name) != _bearing_side(pred_name) and _fault_location(true_name) == _fault_location(pred_name):
        return True
    return True


def compute_replay_priority(
    fse_entropy: np.ndarray,
    top1_top2_margin: np.ndarray,
    error_flag: np.ndarray,
    confusion_class_flag: np.ndarray,
    fault_size_confusion_flag: np.ndarray,
) -> np.ndarray:
    entropy = np.asarray(fse_entropy, dtype=np.float32)
    margin = np.asarray(top1_top2_margin, dtype=np.float32)
    return (
        0.35 * entropy
        + 0.25 * (1.0 - margin)
        + 0.20 * np.asarray(error_flag, dtype=np.float32)
        + 0.10 * np.asarray(confusion_class_flag, dtype=np.float32)
        + 0.10 * np.asarray(fault_size_confusion_flag, dtype=np.float32)
    ).astype(np.float32)


def select_replay_samples(
    dataset: dict[str, object],
    probs: np.ndarray,
    preds: np.ndarray,
    n: int,
    seed: int = 42,
    prototype_version: str = "",
) -> tuple[ReplayBuffer, list[ReplayRecord]]:
    """Build a replay buffer and return top-priority samples."""

    del seed  # deterministic priority ordering is used for P0 reproducibility.
    y = np.asarray(dataset["y"], dtype=np.int64)
    sample_ids = np.asarray(dataset["sample_id"], dtype=np.int64)
    loads = np.asarray(dataset["load"], dtype=np.int64)
    domains = np.asarray(dataset["domain_id"], dtype=np.int64)
    window_indices = np.asarray(dataset["window_index"], dtype=np.int64)
    file_ids = list(dataset["file_id"])  # type: ignore[arg-type]
    class_names = list(dataset["class_names"])  # type: ignore[arg-type]

    entropy = compute_fse_entropy(probs, normalize=True)
    margin = compute_top_margin(probs)
    error_flag = preds != y
    confusion_flags = []
    size_flags = []
    confusion_types = []
    for true, pred in zip(y, preds):
        true_name = class_names[int(true)]
        pred_name = class_names[int(pred)]
        size_flag = detect_fault_size_confusion(true_name, pred_name)
        confusion_flag = detect_confusion_class(true_name, pred_name)
        size_flags.append(size_flag)
        confusion_flags.append(confusion_flag)
        confusion_types.append(_confusion_type(true_name, pred_name, size_flag))

    priority = compute_replay_priority(
        entropy,
        margin,
        error_flag,
        np.asarray(confusion_flags, dtype=bool),
        np.asarray(size_flags, dtype=bool),
    )

    rows: list[ReplayRecord] = []
    for i in range(len(y)):
        rows.append(
            ReplayRecord(
                sample_id=int(sample_ids[i]),
                domain_id=int(domains[i]),
                load=int(loads[i]),
                true_label=int(y[i]),
                label_name=class_names[int(y[i])],
                file_id=str(file_ids[i]),
                window_index=int(window_indices[i]),
                predicted_label=int(preds[i]),
                is_correct=bool(preds[i] == y[i]),
                fse_entropy=float(entropy[i]),
                top1_top2_margin=float(margin[i]),
                replay_priority=float(priority[i]),
                confusion_type=confusion_types[i],
                snapshot_probs=np.asarray(probs[i], dtype=np.float32).tolist(),
                selection_reason="fse_priority",
                prototype_version=prototype_version,
            )
        )
    buffer = ReplayBuffer(rows)
    selected = buffer.sample_by_priority(n)
    return buffer, selected


def export_selected_samples(samples: list[ReplayRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ReplayRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in samples:
            writer.writerow(row.__dict__)


def _fault_location(name: str) -> str:
    if name == "Normal":
        return "Normal"
    if "InnerRace" in name:
        return "InnerRace"
    if "Ball" in name:
        return "Ball"
    if "OuterRace" in name:
        return "OuterRace"
    return "Unknown"


def _fault_size(name: str) -> str:
    if name == "Normal":
        return "Normal"
    for token in ["007", "014", "021"]:
        if token in name:
            return f"0.{token}"
    return "Unknown"


def _bearing_side(name: str) -> str:
    if name.startswith("FanEnd"):
        return "FanEnd"
    if name.startswith("DriveEnd"):
        return "DriveEnd"
    return "Normal"


def _confusion_type(true_name: str, pred_name: str, size_flag: bool) -> str:
    if true_name == pred_name:
        return "correct"
    if size_flag:
        return "fault_size_confusion"
    if _bearing_side(true_name) != _bearing_side(pred_name):
        return "fan_drive_confusion"
    if _fault_location(true_name) == _fault_location(pred_name):
        return "within_fault_location_confusion"
    return "other_confusion"
