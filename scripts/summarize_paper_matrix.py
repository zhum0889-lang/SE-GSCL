"""Aggregate paper-level P1/P2/P3 experiment matrices over random seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs" / "experiments" / "paper_matrix.json"),
    )
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results" / "paper_matrix"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "paper_matrix" / "summary"),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "ok":
        raise ValueError(f"Non-successful report: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    identity_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        identity = tuple(row[key] for key in identity_keys)
        grouped.setdefault(identity, []).append(row)
    result: list[dict[str, Any]] = []
    for identity, samples in sorted(grouped.items()):
        output = dict(zip(identity_keys, identity))
        output["num_seeds"] = len(samples)
        numeric_keys = sorted(
            {
                key
                for sample in samples
                for key, value in sample.items()
                if key not in identity_keys
                and key != "seed"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )
        for key in numeric_keys:
            values = [
                float(sample[key])
                for sample in samples
                if isinstance(sample.get(key), (int, float))
                and not isinstance(sample.get(key), bool)
            ]
            if not values:
                continue
            output[f"{key}_mean"] = fmean(values)
            output[f"{key}_std"] = pstdev(values)
            output[f"{key}_min"] = min(values)
            output[f"{key}_max"] = max(values)
        result.append(output)
    return result


def _minimum_recall(metrics_by_domain: dict[str, Any]) -> float:
    recalls = [
        float(value)
        for metrics in metrics_by_domain.values()
        for value in metrics.get("per_class_accuracy", {}).values()
    ]
    return min(recalls) if recalls else 0.0


def collect_p1(
    root: Path,
    matrix: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    jobs = {str(row["id"]): row for row in matrix["p1_jobs"]}
    for dataset in matrix["datasets"]:
        for seed in matrix["seeds"]:
            for job_id, job in jobs.items():
                path = (
                    root
                    / "p1"
                    / dataset
                    / f"seed_{seed}"
                    / job_id
                    / "p1_report.json"
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue
                report = _read_json(path)
                acc = report["sequence_summary"]["accuracy"]
                bal = report["sequence_summary"]["balanced_accuracy"]
                rows.append(
                    {
                        "stage": "p1",
                        "dataset": dataset,
                        "job": job_id,
                        "label": job.get("label", job_id),
                        "group": job.get("group", ""),
                        "seed": int(seed),
                        "final_accuracy": acc["final_average_accuracy"],
                        "incremental_accuracy": acc[
                            "average_incremental_accuracy"
                        ],
                        "final_balanced_accuracy": bal[
                            "final_average_accuracy"
                        ],
                        "incremental_balanced_accuracy": bal[
                            "average_incremental_accuracy"
                        ],
                        "average_forgetting": bal["average_forgetting"],
                        "maximum_forgetting": bal["maximum_forgetting"],
                        "backward_transfer": bal[
                            "average_backward_transfer"
                        ],
                        "old_domain_retention": bal[
                            "average_old_domain_retention"
                        ],
                        "minimum_final_class_recall": _minimum_recall(
                            report["final_stage_metrics"]
                        ),
                    }
                )
    return rows, missing


def _mean_domain_metric(
    domain_metrics: dict[str, Any],
    branch: str,
    metric: str,
) -> float:
    return fmean(
        float(value[branch][metric]) for value in domain_metrics.values()
    )


def collect_p2(
    root: Path,
    matrix: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    jobs = {str(row["id"]): row for row in matrix["p2_jobs"]}
    for dataset in matrix["datasets"]:
        for seed in matrix["seeds"]:
            for job_id, job in jobs.items():
                path = (
                    root
                    / "p2"
                    / dataset
                    / f"seed_{seed}"
                    / job_id
                    / "p2_report.json"
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue
                report = _read_json(path)
                domain_metrics = report["domain_metrics"]
                row = {
                    "stage": "p2",
                    "dataset": dataset,
                    "job": job_id,
                    "group": job.get("group", ""),
                    "seed": int(seed),
                }
                for branch in ("global", "local", "fused"):
                    row[f"{branch}_accuracy"] = _mean_domain_metric(
                        domain_metrics,
                        branch,
                        "accuracy",
                    )
                    row[f"{branch}_balanced_accuracy"] = (
                        _mean_domain_metric(
                            domain_metrics,
                            branch,
                            "balanced_accuracy",
                        )
                    )
                row["mean_local_weight"] = fmean(
                    float(value["fusion_diagnostics"]["mean_local_weight"])
                    for value in domain_metrics.values()
                )
                row["local_override_rate"] = fmean(
                    float(value["fusion_diagnostics"]["local_override_rate"])
                    for value in domain_metrics.values()
                )
                rows.append(row)
    return rows, missing


def _metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def collect_p3(
    root: Path,
    matrix: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    jobs = {str(row["id"]): row for row in matrix["p3_jobs"]}
    report_names = {
        "structured_text": "p3_report.json",
        "continuous_prompt": "p31_report.json",
        "continuous_explanation": "p32_report.json",
    }
    for dataset in matrix["datasets"]:
        for seed in matrix["seeds"]:
            for job_id, job in jobs.items():
                job_type = str(job["type"])
                path = (
                    root
                    / "p3"
                    / dataset
                    / f"seed_{seed}"
                    / job_id
                    / report_names[job_type]
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue
                report = _read_json(path)
                row: dict[str, Any] = {
                    "stage": "p3",
                    "dataset": dataset,
                    "job": job_id,
                    "group": job.get("group", ""),
                    "type": job_type,
                    "seed": int(seed),
                }
                if job_type == "continuous_prompt":
                    generated = report["continuous_prompt_metrics"]
                    row["raw_accuracy"] = generated["accuracy"]
                    row["raw_balanced_accuracy"] = generated[
                        "balanced_accuracy"
                    ]
                    row["valid_label_rate"] = generated[
                        "valid_label_rate"
                    ]
                    row["upstream_accuracy"] = report[
                        "upstream_fused_baseline"
                    ]["accuracy"]
                    paired = report.get(
                        "qwen_upstream_paired_comparison",
                        {},
                    )
                    row["upstream_agreement"] = paired.get(
                        "prediction_agreement_rate"
                    )
                else:
                    raw = report["raw_metrics"]
                    controlled = report["controlled_metrics"]
                    for metric_name in (
                        "llm_diagnosis_accuracy",
                        "json_parse_rate",
                        "schema_valid_rate",
                        "sample_evidence_valid_rate",
                        "maintenance_policy_consistency_rate",
                    ):
                        raw_value = _metric(raw, metric_name)
                        controlled_value = _metric(
                            controlled,
                            metric_name,
                        )
                        if raw_value is not None:
                            row[f"raw_{metric_name}"] = raw_value
                        if controlled_value is not None:
                            row[f"controlled_{metric_name}"] = (
                                controlled_value
                            )
                    row["semantic_control_repair_rate"] = (
                        controlled.get("semantic_control_repair_rate", 0.0)
                    )
                    if "diagnosis_preservation" in report:
                        preservation = report["diagnosis_preservation"]
                        row["diagnosis_preservation_rate"] = preservation[
                            "diagnosis_preservation_rate"
                        ]
                        row["valid_output_preservation_rate"] = (
                            preservation.get(
                                "valid_output_diagnosis_preservation_rate",
                                preservation["diagnosis_preservation_rate"],
                            )
                        )
                rows.append(row)
    return rows, missing


def _relative_paths(
    paths: Iterable[str],
    root: Path,
) -> list[str]:
    values: list[str] = []
    for value in paths:
        path = Path(value)
        try:
            values.append(str(path.relative_to(root)))
        except ValueError:
            values.append(str(path))
    return values


def main() -> int:
    args = parse_args()
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    p1_rows, p1_missing = collect_p1(root, matrix)
    p2_rows, p2_missing = collect_p2(root, matrix)
    p3_rows, p3_missing = collect_p3(root, matrix)
    p1_summary = _aggregate(
        p1_rows,
        identity_keys=("stage", "dataset", "job", "label", "group"),
    )
    p2_summary = _aggregate(
        p2_rows,
        identity_keys=("stage", "dataset", "job", "group"),
    )
    p3_summary = _aggregate(
        p3_rows,
        identity_keys=("stage", "dataset", "job", "group", "type"),
    )
    _write_csv(output / "p1_by_seed.csv", p1_rows)
    _write_csv(output / "p1_summary.csv", p1_summary)
    _write_csv(output / "p2_by_seed.csv", p2_rows)
    _write_csv(output / "p2_summary.csv", p2_summary)
    _write_csv(output / "p3_by_seed.csv", p3_rows)
    _write_csv(output / "p3_summary.csv", p3_summary)

    missing = p1_missing + p2_missing + p3_missing
    summary = {
        "status": "ok" if not missing else "incomplete",
        "matrix": str(Path(args.matrix).resolve()),
        "statistics": {
            "seeds": matrix["seeds"],
            "mean": "arithmetic mean",
            "std": "population standard deviation (ddof=0)",
        },
        "completed_reports": {
            "p1": len(p1_rows),
            "p2": len(p2_rows),
            "p3": len(p3_rows),
        },
        "expected_reports": {
            "p1": (
                len(matrix["datasets"])
                * len(matrix["seeds"])
                * len(matrix["p1_jobs"])
            ),
            "p2": (
                len(matrix["datasets"])
                * len(matrix["seeds"])
                * len(matrix["p2_jobs"])
            ),
            "p3": (
                len(matrix["datasets"])
                * len(matrix["seeds"])
                * len(matrix["p3_jobs"])
            ),
        },
        "missing_reports": _relative_paths(missing, root),
        "tables": {
            "p1": "p1_summary.csv",
            "p2": "p2_summary.csv",
            "p3": "p3_summary.csv",
        },
    }
    (output / "paper_matrix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
