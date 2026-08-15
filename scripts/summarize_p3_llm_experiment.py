"""Summarize and visualize multi-seed LLM diagnosis ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


DEFAULT_JOBS = (
    "continuous_no_fuzzy_identity",
    "continuous_identity_only",
    "continuous_no_condition",
    "continuous_full",
    "continuous_full_lora",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jobs", default=",".join(DEFAULT_JOBS))
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float]) -> tuple[float, float]:
    return fmean(values), pstdev(values)


def _load_reports(root: Path, jobs: list[str]) -> list[tuple[str, int, dict[str, Any]]]:
    reports = []
    for seed_dir in sorted(root.glob("seed_*"), key=lambda path: int(path.name.split("_")[-1])):
        seed = int(seed_dir.name.split("_")[-1])
        for job in jobs:
            path = seed_dir / job / "p31_report.json"
            if not path.is_file():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "ok":
                raise ValueError(f"Non-successful report: {path}")
            reports.append((job, seed, report))
    return reports


def _by_seed_rows(
    reports: list[tuple[str, int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    domain_rows = []
    for job, seed, report in reports:
        llm = report["continuous_prompt_metrics"]
        baselines = report["upstream_semantic_baselines"]
        global_identity = baselines["global_fault_identity"]
        fused = baselines["hierarchical_fusion"]
        paired = report.get("llm_upstream_paired_comparison")
        if paired is None:
            paired = report["qwen_upstream_paired_comparison"]
        coverage = global_identity["candidate_coverage"]
        rows.append(
            {
                "job": job,
                "seed": seed,
                "llm_tuning": report.get("llm_tuning", "frozen"),
                "context_mode": report["input_contract"].get("context_mode", "unknown"),
                "samples": llm["samples"],
                "global_identity_accuracy": global_identity["accuracy"],
                "global_identity_balanced_accuracy": global_identity["balanced_accuracy"],
                "fused_accuracy": fused["accuracy"],
                "fused_balanced_accuracy": fused["balanced_accuracy"],
                "llm_accuracy": llm["accuracy"],
                "llm_balanced_accuracy": llm["balanced_accuracy"],
                "llm_minus_fused_accuracy": paired["qwen_minus_upstream_accuracy"],
                "llm_minus_fused_balanced_accuracy": (
                    llm["balanced_accuracy"] - fused["balanced_accuracy"]
                ),
                "correction_rate": paired["correction_rate"],
                "corruption_rate": paired["corruption_rate"],
                "net_correction_rate": paired["net_correction_rate"],
                "agreement_rate": paired["prediction_agreement_rate"],
                "mcnemar_exact_p_value": paired["mcnemar_exact_p_value"],
                "identity_top1_coverage": coverage.get("top_1"),
                "identity_top2_coverage": coverage.get("top_2"),
                "identity_top3_coverage": coverage.get("top_3"),
            }
        )
        for domain, metrics in llm["domain_metrics"].items():
            domain_rows.append(
                {
                    "job": job,
                    "seed": seed,
                    "domain": int(domain),
                    "llm_accuracy": metrics["accuracy"],
                    "llm_balanced_accuracy": metrics.get("balanced_accuracy"),
                    "fused_accuracy": fused["domain_metrics"][domain]["accuracy"],
                    "fused_balanced_accuracy": fused["domain_metrics"][domain].get(
                        "balanced_accuracy"
                    ),
                    "global_identity_accuracy": global_identity["domain_metrics"][domain][
                        "accuracy"
                    ],
                }
            )
    return rows, domain_rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["job"]), []).append(row)
    output = []
    identity = {"job", "seed", "llm_tuning", "context_mode"}
    for job, samples in grouped.items():
        result: dict[str, Any] = {
            "job": job,
            "num_seeds": len(samples),
            "llm_tuning": samples[0]["llm_tuning"],
            "context_mode": samples[0]["context_mode"],
        }
        numeric_keys = [
            key
            for key, value in samples[0].items()
            if key not in identity
            and key != "mcnemar_exact_p_value"
            and isinstance(value, (int, float))
        ]
        for key in numeric_keys:
            values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
            if not values:
                continue
            mean, std = _stats(values)
            result[f"{key}_mean"] = mean
            result[f"{key}_std"] = std
        p_values = [
            float(sample["mcnemar_exact_p_value"])
            for sample in samples
            if sample.get("mcnemar_exact_p_value") is not None
        ]
        result["mcnemar_significant_seeds"] = sum(
            value < 0.05 for value in p_values
        )
        result["mcnemar_available_seeds"] = len(p_values)
        result["mcnemar_p_value_min"] = min(p_values) if p_values else None
        result["mcnemar_p_value_max"] = max(p_values) if p_values else None
        output.append(result)
    return output


def _markdown(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# 18-domain LLM diagnosis analysis",
        "",
        "| Setting | LLM tuning | Context | LLM BAcc | Fused BAcc | BAcc delta | Correction | Corruption | Significant seeds | Max p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {job} | {llm_tuning} | {context_mode} | {llm:.4f} +/- {llm_std:.4f} | "
            "{fused:.4f} | {delta:+.4f} | {correction:.4f} | {corruption:.4f} | "
            "{significant}/{available} | {p} |".format(
                job=row["job"],
                llm_tuning=row["llm_tuning"],
                context_mode=row["context_mode"],
                llm=row["llm_balanced_accuracy_mean"],
                llm_std=row["llm_balanced_accuracy_std"],
                fused=row["fused_balanced_accuracy_mean"],
                delta=row["llm_minus_fused_balanced_accuracy_mean"],
                correction=row["correction_rate_mean"],
                corruption=row["corruption_rate_mean"],
                significant=row["mcnemar_significant_seeds"],
                available=row["mcnemar_available_seeds"],
                p=(
                    "n/a"
                    if row["mcnemar_p_value_max"] is None
                    else f'{row["mcnemar_p_value_max"]:.4g}'
                ),
            )
        )
    lines.extend(
        [
            "",
            "A positive LLM claim requires correction rate > corruption rate, a positive paired accuracy delta, and consistent gains across seeds/domains.",
            "Ground-truth labels are used only after generation for evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def _save_figure(
    figure: Any,
    output_dir: Path,
    stem: str,
    formats: list[str],
) -> list[str]:
    paths = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    return paths


def _preferred_job(reports: list[tuple[str, int, dict[str, Any]]]) -> str:
    available = {job for job, _, _ in reports}
    for job in ("continuous_full_lora", "continuous_full"):
        if job in available:
            return job
    return reports[-1][0]


def _mean_std(values: list[float]) -> tuple[float, float]:
    return fmean(values), pstdev(values)


def _plot(
    summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    domain_rows: list[dict[str, Any]],
    reports: list[tuple[str, int, dict[str, Any]]],
    output_dir: Path,
    formats: list[str],
) -> dict[str, list[str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Visualization dependencies are missing. Install requirements.txt."
        ) from exc
    manifest: dict[str, list[str]] = {}
    labels = [row["job"].replace("continuous_", "") for row in summary]
    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(
        x - width,
        [100 * row["global_identity_balanced_accuracy_mean"] for row in summary],
        width,
        label="Small model identity",
        color="#7A8291",
    )
    axes[0].bar(
        x,
        [100 * row["fused_balanced_accuracy_mean"] for row in summary],
        width,
        label="Hierarchical fusion",
        color="#2A9D8F",
    )
    axes[0].bar(
        x + width,
        [100 * row["llm_balanced_accuracy_mean"] for row in summary],
        width,
        yerr=[100 * row["llm_balanced_accuracy_std"] for row in summary],
        label="LLM diagnosis",
        color="#D55E00",
        capsize=4,
    )
    axes[0].set_title("Diagnostic balanced accuracy")
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x - width / 2,
        [100 * row["correction_rate_mean"] for row in summary],
        width,
        label="LLM correction rate",
        color="#3A86FF",
    )
    axes[1].bar(
        x + width / 2,
        [100 * row["corruption_rate_mean"] for row in summary],
        width,
        label="LLM corruption rate",
        color="#E76F51",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Paired LLM contribution")
    axes[1].set_ylabel("Samples (%)")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("18-domain continuous-semantic LLM evaluation", fontweight="bold")
    fig.tight_layout()
    manifest["overall_classification"] = _save_figure(
        fig,
        output_dir,
        "p3_llm_accuracy_and_correction",
        formats,
    )
    plt.close(fig)

    selected_job = _preferred_job(reports)
    selected_domains = [
        row for row in domain_rows if row["job"] == selected_job
    ]
    domains = sorted({int(row["domain"]) for row in selected_domains})
    if domains:
        fig, axis = plt.subplots(figsize=(12.2, 4.8))
        methods = (
            ("global_identity_accuracy", "Small-model fault identity", "#7A8291", "o"),
            ("fused_accuracy", "Hierarchical semantic fusion", "#2A9D8F", "s"),
            ("llm_accuracy", "Continuous-prompt LLM", "#D55E00", "D"),
        )
        x = np.arange(len(domains))
        for key, label, color, marker in methods:
            means = []
            stds = []
            for domain in domains:
                values = [
                    100.0 * float(row[key])
                    for row in selected_domains
                    if int(row["domain"]) == domain
                ]
                mean, std = _mean_std(values)
                means.append(mean)
                stds.append(std)
            axis.plot(
                x,
                means,
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=5.5,
                label=label,
            )
            axis.fill_between(
                x,
                np.asarray(means) - np.asarray(stds),
                np.asarray(means) + np.asarray(stds),
                color=color,
                alpha=0.12,
            )
        axis.set_xticks(x, [f"D{index + 1}" for index in range(len(domains))])
        axis.set_xlabel("Operating-condition domain")
        axis.set_ylabel("Accuracy (%)")
        axis.set_title(
            f"Domain-wise classification across three seeds ({selected_job})",
            fontweight="bold",
        )
        axis.set_ylim(0.0, 102.0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, ncol=3, loc="lower center")
        axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        manifest["domain_accuracy"] = _save_figure(
            fig,
            output_dir,
            "p3_llm_domain_accuracy",
            formats,
        )
        plt.close(fig)

    selected_reports = [
        report for job, _, report in reports if job == selected_job
    ]
    if selected_reports:
        llm_metrics = [
            report["continuous_prompt_metrics"] for report in selected_reports
        ]
        class_names = list(llm_metrics[0].get("per_class_recall", {}))
        confusion_rows = [
            np.asarray(metrics.get("confusion_matrix", []), dtype=float)
            for metrics in llm_metrics
        ]
        has_confusion = (
            class_names
            and confusion_rows
            and all(matrix.shape == confusion_rows[0].shape for matrix in confusion_rows)
            and confusion_rows[0].shape == (len(class_names), len(class_names))
        )
        if class_names:
            fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
            x = np.arange(len(class_names))
            width = 0.24
            branches = (
                (
                    "global_fault_identity",
                    "Small-model identity",
                    "#7A8291",
                ),
                (
                    "hierarchical_fusion",
                    "Hierarchical fusion",
                    "#2A9D8F",
                ),
                (None, "Continuous-prompt LLM", "#D55E00"),
            )
            for offset, (branch, label, color) in enumerate(branches):
                means = []
                stds = []
                for class_name in class_names:
                    if branch is None:
                        values = [
                            100.0 * float(metrics["per_class_recall"][class_name])
                            for metrics in llm_metrics
                        ]
                    else:
                        values = [
                            100.0
                            * float(
                                report["upstream_semantic_baselines"][branch][
                                    "per_class_recall"
                                ][class_name]
                            )
                            for report in selected_reports
                        ]
                    mean, std = _mean_std(values)
                    means.append(mean)
                    stds.append(std)
                axes[0].bar(
                    x + (offset - 1) * width,
                    means,
                    width,
                    yerr=stds,
                    color=color,
                    label=label,
                    capsize=3,
                )
            axes[0].set_xticks(x, class_names, rotation=18, ha="right")
            axes[0].set_ylabel("Recall (%)")
            axes[0].set_title("Class-wise recall")
            axes[0].set_ylim(0.0, 105.0)
            axes[0].grid(axis="y", alpha=0.25)
            axes[0].legend(frameon=False, fontsize=8)
            axes[0].spines[["top", "right"]].set_visible(False)

            if has_confusion:
                confusion = np.sum(confusion_rows, axis=0)
                row_totals = confusion.sum(axis=1, keepdims=True)
                normalized = np.divide(
                    confusion,
                    row_totals,
                    out=np.zeros_like(confusion),
                    where=row_totals > 0,
                )
                image = axes[1].imshow(
                    normalized * 100.0,
                    vmin=0.0,
                    vmax=100.0,
                    cmap="Blues",
                    aspect="auto",
                )
                for row_index in range(len(class_names)):
                    for column_index in range(len(class_names)):
                        value = normalized[row_index, column_index] * 100.0
                        axes[1].text(
                            column_index,
                            row_index,
                            f"{value:.1f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if value >= 55.0 else "black",
                        )
                axes[1].set_xticks(x, class_names, rotation=18, ha="right")
                axes[1].set_yticks(x, class_names)
                axes[1].set_xlabel("Predicted class")
                axes[1].set_ylabel("True class")
                axes[1].set_title("Aggregated normalized confusion matrix")
                fig.colorbar(image, ax=axes[1], label="Recall within true class (%)")
            else:
                axes[1].axis("off")
                axes[1].text(
                    0.5,
                    0.5,
                    "Confusion matrices are unavailable",
                    ha="center",
                    va="center",
                )
            fig.suptitle(
                f"Class-level classification analysis ({selected_job}, n={len(selected_reports)})",
                fontweight="bold",
            )
            fig.tight_layout()
            manifest["class_recall_confusion"] = _save_figure(
                fig,
                output_dir,
                "p3_llm_class_recall_confusion",
                formats,
            )
            plt.close(fig)

    seed_rows = [row for row in rows if row["job"] == selected_job]
    if seed_rows:
        seed_rows.sort(key=lambda row: int(row["seed"]))
        fig, axis = plt.subplots(figsize=(7.8, 4.4))
        seed_labels = [str(row["seed"]) for row in seed_rows]
        x = np.arange(len(seed_rows))
        width = 0.25
        axis.bar(
            x - width,
            [100.0 * float(row["global_identity_balanced_accuracy"]) for row in seed_rows],
            width,
            label="Small-model identity",
            color="#7A8291",
        )
        axis.bar(
            x,
            [100.0 * float(row["fused_balanced_accuracy"]) for row in seed_rows],
            width,
            label="Hierarchical fusion",
            color="#2A9D8F",
        )
        axis.bar(
            x + width,
            [100.0 * float(row["llm_balanced_accuracy"]) for row in seed_rows],
            width,
            label="Continuous-prompt LLM",
            color="#D55E00",
        )
        axis.set_xticks(x, seed_labels)
        axis.set_xlabel("Random seed")
        axis.set_ylabel("Balanced accuracy (%)")
        axis.set_title(
            f"Classification stability across seeds ({selected_job})",
            fontweight="bold",
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        manifest["seed_stability"] = _save_figure(
            fig,
            output_dir,
            "p3_llm_seed_stability",
            formats,
        )
        plt.close(fig)
    return manifest


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [value.strip() for value in args.jobs.split(",") if value.strip()]
    reports = _load_reports(root, jobs)
    if not reports:
        raise FileNotFoundError(f"No P3.1 reports found under {root}")
    rows, domain_rows = _by_seed_rows(reports)
    summary = _aggregate(rows)
    _write_csv(output_dir / "p3_llm_by_seed.csv", rows)
    _write_csv(output_dir / "p3_llm_by_domain.csv", domain_rows)
    _write_csv(output_dir / "p3_llm_summary.csv", summary)
    (output_dir / "p3_llm_analysis.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    payload = {
        "status": "ok",
        "reports": len(reports),
        "jobs": jobs,
        "summary": summary,
        "interpretation_rule": (
            "LLM gain is supported only when correction exceeds corruption, "
            "paired delta is positive, and gains are stable across seeds/domains."
        ),
    }
    (output_dir / "p3_llm_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figures = _plot(
        summary,
        rows,
        domain_rows,
        reports,
        output_dir,
        [value.strip() for value in args.formats.split(",") if value.strip()],
    )
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(figures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    payload["figures"] = figures
    (output_dir / "p3_llm_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
