"""Metrics for cross-load continual FD-LLM experiments."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import f1_score

from experiments.continual_fdllm.fse_replay_selector import compute_fse_entropy


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def compute_domain_accuracy(y_true: np.ndarray, y_pred: np.ndarray, domain_id: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for domain in sorted(np.unique(domain_id).tolist()):
        mask = np.asarray(domain_id) == domain
        out[str(int(domain))] = compute_accuracy(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])
    return out


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 3) -> float:
    if len(y_true) == 0:
        return 0.0
    topk = np.argsort(np.asarray(probs), axis=1)[:, -k:]
    return float(np.mean([int(label) in row for label, row in zip(y_true, topk)]))


def compute_fse_entropy_mean(probs: np.ndarray) -> float:
    if len(probs) == 0:
        return 0.0
    return float(compute_fse_entropy(probs, normalize=True).mean())


def compute_forgetting(t0_old_accuracy: float, adapted_old_accuracy: float) -> float:
    return float(t0_old_accuracy - adapted_old_accuracy)


def compute_old_domain_retention(t0_old_accuracy: float, adapted_old_accuracy: float) -> float:
    if t0_old_accuracy <= 1e-12:
        return 0.0
    return float(adapted_old_accuracy / t0_old_accuracy)


def compute_fault_location_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> float:
    true_loc = np.asarray([_fault_location(class_names[int(v)]) for v in y_true])
    pred_loc = np.asarray([_fault_location(class_names[int(v)]) for v in y_pred])
    return compute_accuracy(true_loc, pred_loc)


def compute_fault_size_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> float:
    true_size = np.asarray([_fault_size(class_names[int(v)]) for v in y_true])
    pred_size = np.asarray([_fault_size(class_names[int(v)]) for v in y_pred])
    return compute_accuracy(true_size, pred_size)


def compute_confusion_pair_count(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for true, pred in zip(y_true, y_pred):
        true_name = class_names[int(true)]
        pred_name = class_names[int(pred)]
        if true_name == pred_name:
            continue
        true_size = _fault_size(true_name)
        pred_size = _fault_size(pred_name)
        if {true_size, pred_size} == {"0.014", "0.021"}:
            counts["0.014_vs_0.021"] += 1
        if true_name.startswith("FanEnd_Ball") and pred_name.startswith("FanEnd_Ball"):
            counts["FanEnd_Ball_internal"] += 1
        if true_name.startswith("DriveEnd_Ball") and pred_name.startswith("DriveEnd_Ball"):
            counts["DriveEnd_Ball_internal"] += 1
        if _bearing_side(true_name) != _bearing_side(pred_name):
            counts["FanEnd_vs_DriveEnd"] += 1
    for key in ["0.014_vs_0.021", "FanEnd_Ball_internal", "DriveEnd_Ball_internal", "FanEnd_vs_DriveEnd"]:
        counts.setdefault(key, 0)
    return dict(counts)


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

