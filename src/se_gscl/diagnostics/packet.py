"""Auditable semantic evidence packet for prompt-enhanced diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SemanticDiagnosticPacket:
    sample_id: int
    domain_id: int
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    normalized_entropy: float
    top1_top2_margin: float
    global_local_agreement: bool
    top_candidates: tuple[dict[str, Any], ...]
    top_symptoms: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    return clipped / clipped.sum()


def build_semantic_diagnostic_packet(
    *,
    sample_id: int,
    domain_id: int,
    class_names: Sequence[str],
    symptom_names: Sequence[str],
    symptom_class_ids: Sequence[int],
    global_probabilities: np.ndarray,
    local_probabilities: np.ndarray,
    symptom_joint_probabilities: np.ndarray,
    local_weight: float = 0.3,
    top_k: int = 3,
    top_symptoms: int = 4,
) -> SemanticDiagnosticPacket:
    if not 0.0 <= local_weight <= 1.0:
        raise ValueError("local_weight must be in [0,1].")
    if top_k <= 0 or top_symptoms <= 0:
        raise ValueError("top_k and top_symptoms must be positive.")
    global_probs = _normalize(global_probabilities)
    local_probs = _normalize(local_probabilities)
    if global_probs.shape != local_probs.shape:
        raise ValueError("Global and local probabilities must have equal shape.")
    if global_probs.shape != (len(class_names),):
        raise ValueError("Class probability length must match class_names.")
    joint = _normalize(symptom_joint_probabilities)
    if joint.shape != (len(symptom_names),):
        raise ValueError("Symptom probability length must match symptom_names.")
    if len(symptom_class_ids) != len(symptom_names):
        raise ValueError("symptom_class_ids must match symptom_names.")

    fused_log = (
        (1.0 - local_weight) * np.log(global_probs)
        + local_weight * np.log(local_probs)
    )
    fused = np.exp(fused_log - fused_log.max())
    fused = _normalize(fused)
    order = np.argsort(-fused)
    prediction = int(order[0])
    margin = float(fused[order[0]] - fused[order[1]]) if len(order) > 1 else 1.0
    entropy = float(-(fused * np.log(fused)).sum())
    normalized_entropy = entropy / math.log(len(fused)) if len(fused) > 1 else 0.0

    candidate_rows = tuple(
        {
            "class_id": int(index),
            "class_name": str(class_names[int(index)]),
            "probability": float(fused[int(index)]),
            "global_probability": float(global_probs[int(index)]),
            "local_probability": float(local_probs[int(index)]),
        }
        for index in order[: min(top_k, len(order))]
    )
    symptom_order = np.argsort(-joint)
    symptom_rows = tuple(
        {
            "symptom_id": int(index),
            "symptom_name": str(symptom_names[int(index)]),
            "class_id": int(symptom_class_ids[int(index)]),
            "probability": float(joint[int(index)]),
        }
        for index in symptom_order[: min(top_symptoms, len(symptom_order))]
    )
    return SemanticDiagnosticPacket(
        sample_id=int(sample_id),
        domain_id=int(domain_id),
        predicted_class_id=prediction,
        predicted_class_name=str(class_names[prediction]),
        confidence=float(fused[prediction]),
        normalized_entropy=float(normalized_entropy),
        top1_top2_margin=margin,
        global_local_agreement=bool(
            int(global_probs.argmax()) == int(local_probs.argmax())
        ),
        top_candidates=candidate_rows,
        top_symptoms=symptom_rows,
    )
