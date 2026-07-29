"""Auditable prompt construction and evaluation for frozen-LLM diagnosis."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Sequence


ALLOWED_MAINTENANCE_ACTIONS = (
    "Continue monitoring under the current condition.",
    "Verify the diagnosis with spectrum and envelope-spectrum inspection.",
    "Inspect the bearing and schedule maintenance.",
    "Stop the equipment for immediate bearing inspection.",
)

_SYSTEM_PROMPT = """\
You are an industrial rolling-bearing diagnosis module. An upstream signal
model has converted a vibration sample into fault probabilities and physically
grounded symptom evidence. Use only the supplied evidence. Do not invent
frequencies, amplitudes, defect severity, or operating conditions.

Return exactly one JSON object without Markdown. The schema is:
{
  "diagnosis": "<one supplied candidate class>",
  "confidence_level": "high|medium|low",
  "evidence": ["<zero to three supplied symptom names>"],
  "explanation": "<concise evidence-linked explanation>",
  "uncertainty_acknowledged": true,
  "maintenance_action": "<one allowed action>"
}
"""


def _probability(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def build_diagnostic_messages(
    packet: dict[str, Any],
) -> list[dict[str, str]]:
    """Build a label-leakage-free diagnostic conversation."""

    candidates = list(packet["top_candidates"])
    symptoms = list(packet["top_symptoms"])
    if not candidates:
        raise ValueError("At least one fault candidate is required.")
    candidate_names = [str(row["class_name"]) for row in candidates]
    symptom_names = [str(row["symptom_name"]) for row in symptoms]
    candidate_lines = [
        (
            f"- {row['class_name']}: fused={_probability(row['probability'])}, "
            f"global={_probability(row['global_probability'])}, "
            f"local={_probability(row['local_probability'])}"
        )
        for row in candidates
    ]
    symptom_lines = [
        f"- {row['symptom_name']}: {_probability(row['probability'])}"
        for row in symptoms
    ]
    user_prompt = "\n".join(
        [
            f"Sample: {int(packet['sample_id'])}",
            f"Condition ID: D{int(packet['domain_id'])}",
            (
                "Global/local branch agreement: "
                f"{bool(packet['global_local_agreement'])}"
            ),
            (
                "Normalized predictive entropy: "
                f"{float(packet['normalized_entropy']):.4f}"
            ),
            (
                "Top-1/Top-2 probability margin: "
                f"{float(packet['top1_top2_margin']):.4f}"
            ),
            "Fault candidates:",
            *candidate_lines,
            "Available symptom evidence:",
            *(symptom_lines or ["- No reliable symptom evidence supplied."]),
            "Allowed diagnosis labels: " + json.dumps(candidate_names),
            "Allowed evidence strings: " + json.dumps(symptom_names),
            (
                "Allowed maintenance actions: "
                + json.dumps(ALLOWED_MAINTENANCE_ACTIONS)
            ),
            (
                "Select the most defensible diagnosis. Set "
                "uncertainty_acknowledged=true when entropy is high, the "
                "margin is small, or the branches disagree."
            ),
        ]
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_diagnostic_json(text: str) -> dict[str, Any] | None:
    """Extract the first complete JSON object from a generated response."""

    value = str(text).strip()
    if value.startswith("```"):
        value = value.removeprefix("```json").removeprefix("```")
        value = value.removesuffix("```").strip()
    start = value.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(value[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def select_evaluation_rows(
    rows: Sequence[dict[str, Any]],
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Select uncertainty-stratified samples from every condition."""

    if max_samples is None or max_samples <= 0 or max_samples >= len(rows):
        return list(rows)
    by_domain: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(int(row["domain_id"]), []).append(row)
    selected: list[dict[str, Any]] = []
    domains = sorted(by_domain)
    base, remainder = divmod(max_samples, len(domains))
    for domain_index, domain in enumerate(domains):
        budget = base + int(domain_index < remainder)
        ordered = sorted(
            by_domain[domain],
            key=lambda row: (
                float(row["normalized_entropy"]),
                int(row["sample_id"]),
            ),
        )
        if budget <= 0:
            continue
        if budget >= len(ordered):
            selected.extend(ordered)
            continue
        indices = {
            int(round(index * (len(ordered) - 1) / max(1, budget - 1)))
            for index in range(budget)
        }
        selected.extend(ordered[index] for index in sorted(indices))
    return sorted(
        selected,
        key=lambda row: (int(row["domain_id"]), int(row["sample_id"])),
    )


def _mean(values: Iterable[bool]) -> float:
    rows = list(values)
    return float(sum(rows) / len(rows)) if rows else 0.0


def evaluate_llm_outputs(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Measure diagnostic preservation and evidence faithfulness."""

    parsed_count = 0
    schema_valid: list[bool] = []
    label_valid: list[bool] = []
    upstream_agreement: list[bool] = []
    llm_correct: list[bool] = []
    upstream_correct: list[bool] = []
    evidence_grounded: list[bool] = []
    sample_evidence_valid: list[bool] = []
    maintenance_valid: list[bool] = []
    uncertainty_required: list[bool] = []
    uncertainty_respected: list[bool] = []
    for record in records:
        packet = record["packet"]
        parsed = record.get("parsed_output")
        upstream_correct.append(
            str(packet["predicted_class_name"])
            == str(packet["ground_truth_class_name"])
        )
        requires_uncertainty = (
            float(packet["normalized_entropy"]) >= 0.60
            or float(packet["top1_top2_margin"]) <= 0.15
            or not bool(packet["global_local_agreement"])
        )
        uncertainty_required.append(requires_uncertainty)
        if not isinstance(parsed, dict):
            schema_valid.append(False)
            label_valid.append(False)
            upstream_agreement.append(False)
            llm_correct.append(False)
            sample_evidence_valid.append(False)
            maintenance_valid.append(False)
            if requires_uncertainty:
                uncertainty_respected.append(False)
            continue
        parsed_count += 1
        required = {
            "diagnosis",
            "confidence_level",
            "evidence",
            "explanation",
            "uncertainty_acknowledged",
            "maintenance_action",
        }
        valid_schema = (
            required.issubset(parsed)
            and isinstance(parsed.get("evidence"), list)
            and isinstance(parsed.get("explanation"), str)
            and isinstance(parsed.get("uncertainty_acknowledged"), bool)
            and parsed.get("confidence_level") in {"high", "medium", "low"}
        )
        schema_valid.append(valid_schema)
        candidate_names = {
            str(row["class_name"]) for row in packet["top_candidates"]
        }
        diagnosis = str(parsed.get("diagnosis", ""))
        valid_label = diagnosis in candidate_names
        label_valid.append(valid_label)
        upstream_agreement.append(
            valid_label
            and diagnosis == str(packet["predicted_class_name"])
        )
        llm_correct.append(
            valid_label
            and diagnosis == str(packet["ground_truth_class_name"])
        )
        supplied_symptoms = {
            str(row["symptom_name"]) for row in packet["top_symptoms"]
        }
        evidence = parsed.get("evidence", [])
        valid_evidence_rows = [
            isinstance(value, str) and value in supplied_symptoms
            for value in evidence
        ] if isinstance(evidence, list) else []
        evidence_grounded.extend(valid_evidence_rows)
        sample_evidence_valid.append(
            bool(valid_evidence_rows) and all(valid_evidence_rows)
        )
        maintenance_valid.append(
            parsed.get("maintenance_action")
            in ALLOWED_MAINTENANCE_ACTIONS
        )
        if requires_uncertainty:
            uncertainty_respected.append(
                parsed.get("uncertainty_acknowledged") is True
            )
    total = len(records)
    return {
        "samples": total,
        "json_parse_rate": parsed_count / max(1, total),
        "schema_valid_rate": _mean(schema_valid),
        "candidate_label_valid_rate": _mean(label_valid),
        "upstream_diagnosis_accuracy": _mean(upstream_correct),
        "llm_diagnosis_accuracy": _mean(llm_correct),
        "llm_upstream_agreement_rate": _mean(upstream_agreement),
        "evidence_item_grounded_rate": _mean(evidence_grounded),
        "sample_evidence_valid_rate": _mean(sample_evidence_valid),
        "maintenance_action_valid_rate": _mean(maintenance_valid),
        "uncertain_samples": int(sum(uncertainty_required)),
        "uncertainty_acknowledgement_rate": _mean(uncertainty_respected),
        "evaluation_note": (
            "Ground-truth labels are used only after generation for metrics."
        ),
    }
