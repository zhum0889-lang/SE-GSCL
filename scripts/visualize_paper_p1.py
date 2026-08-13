"""Visualize multi-seed P1 continual-learning paper results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import fmean, pstdev
import tempfile
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "se_gscl_matplotlib"),
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as error:  # pragma: no cover - environment guard
    raise SystemExit(
        "Visualization dependencies are missing. Run: "
        "pip install 'matplotlib>=3.7' 'numpy>=1.23'"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
CORE_JOBS = ("finetune", "lwf_relation", "experience_replay", "se_gscl_full")
STRATEGY_DIRECTORY_ALIASES = {
    "sequential": "finetune",
    "lwf": "lwf_relation",
    "balanced_replay": "experience_replay",
    "full": "se_gscl_full",
}
ABLATION_JOBS = (
    "se_gscl_full",
    "wo_cross_condition",
    "wo_relation",
    "wo_decorrelation",
)
SHORT_LABELS = {
    "finetune": "Fine-tuning",
    "lwf_relation": "LwF",
    "experience_replay": "Replay",
    "se_gscl_full": "SE-GSCL",
    "wo_cross_condition": "w/o cross-condition",
    "wo_relation": "w/o relation",
    "wo_decorrelation": "w/o decorrelation",
}
COLORS = {
    "finetune": "#6B7280",
    "lwf_relation": "#2A9D8F",
    "experience_replay": "#457B9D",
    "se_gscl_full": "#D55E00",
    "wo_cross_condition": "#CC79A7",
    "wo_relation": "#E9C46A",
    "wo_decorrelation": "#56B4E9",
}
MARKERS = ("o", "s", "^", "D", "P", "X", "v")


def _domain_labels(
    reports: dict[str, list[dict[str, Any]]],
    domains: list[int],
    style: str = "auto",
) -> list[str]:
    dataset = next(iter(reports.values()))[0].get("dataset", "")
    if style == "short" or (
        style == "auto"
        and dataset in {"multidomain8_disjoint18", "multidomain16_disjoint18"}
    ):
        return [f"D{index + 1}" for index in range(len(domains))]
    if dataset in {"multidomain8_atomic", "multidomain16_atomic"}:
        bearings = ("6204", "N204/NJ204", "30204")
        environments = ("H", "M1", "M2", "M3", "U1", "U2", "U3", "L")
        speed_groups = ("slow", "fast")
        return [
            f"{bearings[domain // 16]}\n{environments[(domain % 16) // 2]} | {speed_groups[domain % 2]}"
            for domain in domains
        ]
    if dataset in {"multidomain8", "multidomain16"}:
        bearings = ("6204", "N204/NJ204", "30204")
        environments = ("A", "B", "C")
        speed_groups = ("slow", "fast")
        return [
            f"{bearings[domain // 6]}\n{environments[(domain % 6) // 2]} | {speed_groups[domain % 2]}"
            for domain in domains
        ]
    if dataset in {"multidomain8_disjoint18", "multidomain16_disjoint18"}:
        bearings = ("6204", "N204/NJ204", "30204")
        environments = ("H+L", "U1-U3", "M1-M3")
        speed_groups = ("slow", "fast")
        return [
            f"{bearings[domain // 6]}\n{environments[(domain % 6) // 2]} | {speed_groups[domain % 2]}"
            for domain in domains
        ]
    if dataset == "hustbearing":
        return [f"{domain} Hz" for domain in domains]
    return [f"Domain {domain}" for domain in domains]


def _full_domain_labels(dataset: str, domains: list[int]) -> list[str]:
    proxy_reports = {"dataset": [{"dataset": dataset}]}
    return _domain_labels(proxy_reports, domains, style="full")


def _write_domain_mapping(
    path: Path,
    dataset: str,
    domains: list[int],
    display_labels: list[str],
) -> None:
    full_labels = _full_domain_labels(dataset, domains)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stream_stage",
                "display_label",
                "internal_domain_id",
                "full_condition",
            ),
        )
        writer.writeheader()
        for stage, (domain, display, full) in enumerate(
            zip(domains, display_labels, full_labels, strict=True),
            start=1,
        ):
            writer.writerow(
                {
                    "stream_stage": stage,
                    "display_label": display,
                    "internal_domain_id": domain,
                    "full_condition": full.replace("\n", " | "),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset-level P1 root containing seed_*/job/p1_report.json.",
    )
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs" / "experiments" / "paper_matrix.json"),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--jobs", help="Optional comma-separated job IDs.")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--domain-label-style",
        choices=("auto", "short", "full"),
        default="auto",
        help=(
            "Domain tick labels. auto uses D1-D18 for the main disjoint18 "
            "protocol and full condition names for smaller datasets."
        ),
    )
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_reports(
    root: Path,
    requested_jobs: set[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("seed_*/*/p1_report.json")):
        job = STRATEGY_DIRECTORY_ALIASES.get(path.parent.name, path.parent.name)
        if requested_jobs is not None and job not in requested_jobs:
            continue
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        if report.get("status") != "ok":
            continue
        report["_path"] = str(path.resolve())
        report["_seed"] = _infer_seed(path, report)
        reports.setdefault(job, []).append(report)
    if not reports:
        raise FileNotFoundError(
            f"No successful P1 reports found under {root}/seed_*/*/p1_report.json."
        )
    for job_reports in reports.values():
        job_reports.sort(key=lambda report: int(report["_seed"]))
    return reports


def _infer_seed(path: Path, report: dict[str, Any]) -> int:
    path_seed = int(path.parents[1].name.removeprefix("seed_"))
    reported_seed = report.get("training_config", {}).get("seed")
    if reported_seed is not None and int(reported_seed) != path_seed:
        raise ValueError(
            f"Seed mismatch for {path}: directory={path_seed}, "
            f"report={reported_seed}."
        )
    return path_seed


def _validate_contract(reports: dict[str, list[dict[str, Any]]]) -> tuple[list[int], list[str]]:
    all_reports = [report for values in reports.values() for report in values]
    domains = [int(value) for value in all_reports[0]["domains"]]
    class_names = [str(value) for value in all_reports[0]["class_names"]]
    for report in all_reports[1:]:
        if [int(value) for value in report["domains"]] != domains:
            raise ValueError("P1 reports use inconsistent domain orders.")
        if [str(value) for value in report["class_names"]] != class_names:
            raise ValueError("P1 reports use inconsistent class ontologies.")
    return domains, class_names


def _job_order(
    reports: dict[str, list[dict[str, Any]]],
    matrix_path: Path,
) -> list[str]:
    configured: list[str] = []
    if matrix_path.is_file():
        matrix = json.loads(matrix_path.read_text(encoding="utf-8-sig"))
        configured = [str(row["id"]) for row in matrix.get("p1_jobs", [])]
    ordered = [job for job in configured if job in reports]
    ordered.extend(sorted(set(reports) - set(ordered)))
    return ordered


def _summary_value(report: dict[str, Any], key: str) -> float:
    summary = report["sequence_summary"]["balanced_accuracy"]
    if key in summary:
        return float(summary[key])
    if key == "maximum_forgetting":
        values = [float(value) for value in summary["forgetting_by_domain"]]
        return max(values[:-1], default=0.0)
    if key == "average_old_domain_retention":
        learned = np.asarray(summary["learned_by_domain"][:-1], dtype=float)
        final = np.asarray(summary["final_by_domain"][:-1], dtype=float)
        if len(learned) == 0:
            return 1.0
        retention = np.divide(
            final,
            learned,
            out=np.zeros_like(final),
            where=np.abs(learned) > 1e-12,
        )
        return float(np.mean(retention))
    raise KeyError(f"Missing continual metric {key!r} in {report.get('_path', 'report')}.")


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    materialized = [float(value) for value in values]
    return fmean(materialized), pstdev(materialized)


def _write_summary_csv(
    path: Path,
    reports: dict[str, list[dict[str, Any]]],
    jobs: list[str],
) -> None:
    rows: list[dict[str, object]] = []
    metrics = {
        "final_balanced_accuracy": "final_average_accuracy",
        "incremental_balanced_accuracy": "average_incremental_accuracy",
        "average_forgetting": "average_forgetting",
        "maximum_forgetting": "maximum_forgetting",
        "backward_transfer": "average_backward_transfer",
        "old_domain_retention": "average_old_domain_retention",
    }
    for job in jobs:
        row: dict[str, object] = {
            "job": job,
            "label": SHORT_LABELS.get(job, job),
            "num_seeds": len(reports[job]),
            "seeds": ",".join(str(value["_seed"]) for value in reports[job]),
        }
        for output_key, report_key in metrics.items():
            mean, std = _mean_std(
                _summary_value(report, report_key) for report in reports[job]
            )
            row[f"{output_key}_mean"] = mean
            row[f"{output_key}_std"] = std
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decorate_axis(axis: plt.Axes, *, zero_line: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    if zero_line:
        axis.axhline(0.0, color="#333333", linewidth=0.7)


def _plot_metric_comparison(
    reports: dict[str, list[dict[str, Any]]],
    jobs: list[str],
    *,
    title: str,
) -> plt.Figure:
    panels = (
        ("final_average_accuracy", "Final average accuracy"),
        ("average_incremental_accuracy", "Average incremental accuracy"),
        ("average_forgetting", "Average forgetting"),
        ("average_backward_transfer", "Backward transfer"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(13.4, 3.4))
    x = np.arange(len(jobs))
    labels = [SHORT_LABELS.get(job, job) for job in jobs]
    for axis, (metric, panel_title) in zip(axes, panels):
        means: list[float] = []
        stds: list[float] = []
        for job in jobs:
            mean, std = _mean_std(
                100.0 * _summary_value(report, metric)
                for report in reports[job]
            )
            means.append(mean)
            stds.append(std)
        bars = axis.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=[COLORS.get(job, "#777777") for job in jobs],
            edgecolor="black",
            linewidth=0.45,
            error_kw={"linewidth": 0.8},
        )
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(panel_title)
        axis.set_ylabel("%")
        _decorate_axis(axis, zero_line=metric == "average_backward_transfer")
        if metric in {"final_average_accuracy", "average_incremental_accuracy"}:
            lower = max(0.0, min(means) - max(8.0, max(stds, default=0.0) * 2.0))
            axis.set_ylim(lower, 102.0)
        y_min, y_max = axis.get_ylim()
        label_offset = max((y_max - y_min) * 0.025, 1e-4)
        for bar, mean in zip(bars, means):
            offset = label_offset if mean >= 0 else -label_offset
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + offset,
                f"{mean:.1f}",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=7,
            )
    figure.suptitle(title, fontsize=12, fontweight="bold")
    figure.tight_layout()
    return figure


def _plot_stage_trajectory(
    reports: dict[str, list[dict[str, Any]]],
    jobs: list[str],
    domains: list[int],
    domain_labels: list[str],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(domains))
    all_means: list[float] = []
    for index, job in enumerate(jobs):
        values = np.asarray(
            [
                report["sequence_summary"]["balanced_accuracy"][
                    "stage_seen_average"
                ]
                for report in reports[job]
            ],
            dtype=float,
        ) * 100.0
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        all_means.extend(mean.tolist())
        color = COLORS.get(job, "#777777")
        axis.plot(
            x,
            mean,
            marker=MARKERS[index % len(MARKERS)],
            markersize=4.5,
            linewidth=2.0,
            color=color,
            label=f"{SHORT_LABELS.get(job, job)} (n={len(values)})",
        )
        axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.13)
    axis.set_xticks(x, domain_labels)
    axis.set_xlabel("Continual training stage")
    axis.set_ylabel("Seen-condition balanced accuracy (%)")
    axis.set_title("Performance evolution along the operating-condition stream")
    _decorate_axis(axis)
    lower = max(0.0, np.floor((min(all_means) - 5.0) / 10.0) * 10.0)
    axis.set_ylim(lower, 102.0)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    figure.tight_layout()
    return figure


def _balanced_matrix(report: dict[str, Any], domains: list[int]) -> np.ndarray:
    return np.asarray(
        [
            [
                stage["domain_metrics"][str(domain)]["balanced_accuracy"]
                for domain in domains
            ]
            for stage in report["stage_metrics"]
        ],
        dtype=float,
    ) * 100.0


def _annotated_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    domains: list[int],
    domain_labels: list[str],
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    mask_future: bool,
) -> Any:
    mask = (
        np.triu(np.ones_like(matrix, dtype=bool), k=1)
        if mask_future
        else np.zeros_like(matrix, dtype=bool)
    )
    image = axis.imshow(
        np.ma.array(matrix, mask=mask),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axis.set_xticks(np.arange(len(domains)), domain_labels, rotation=20)
    axis.set_yticks(np.arange(len(domains)), domain_labels)
    axis.set_xlabel("Evaluation condition")
    axis.set_ylabel("Training completed through condition")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if mask[row, column]:
                continue
            value = float(matrix[row, column])
            color = "white" if (cmap == "YlGnBu" and value >= 65.0) else "black"
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=6.3,
            )
    return image


def _plot_accuracy_matrix(
    reports: list[dict[str, Any]],
    domains: list[int],
    domain_labels: list[str],
    job: str,
) -> plt.Figure:
    matrices = np.asarray(
        [_balanced_matrix(report, domains) for report in reports],
        dtype=float,
    )
    mean = np.mean(matrices, axis=0)
    std = np.std(matrices, axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    mean_image = _annotated_heatmap(
        axes[0],
        mean,
        domains,
        domain_labels,
        title="Mean balanced accuracy",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=100.0,
        mask_future=True,
    )
    std_image = _annotated_heatmap(
        axes[1],
        std,
        domains,
        domain_labels,
        title="Across-seed standard deviation",
        cmap="OrRd",
        vmin=0.0,
        vmax=max(5.0, float(np.nanmax(std)) * 1.1),
        mask_future=True,
    )
    figure.colorbar(mean_image, ax=axes[0], fraction=0.046, pad=0.04, label="Accuracy (%)")
    figure.colorbar(std_image, ax=axes[1], fraction=0.046, pad=0.04, label="Std. dev. (pp)")
    figure.suptitle(
        f"{SHORT_LABELS.get(job, job)} continual-learning matrix (n={len(reports)})",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def _plot_domain_profiles(
    reports: dict[str, list[dict[str, Any]]],
    jobs: list[str],
    domains: list[int],
    domain_labels: list[str],
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    x = np.arange(len(domains))
    for index, job in enumerate(jobs):
        final_values = np.asarray(
            [
                report["sequence_summary"]["balanced_accuracy"]["final_by_domain"]
                for report in reports[job]
            ],
            dtype=float,
        ) * 100.0
        forgetting_values = np.asarray(
            [
                report["sequence_summary"]["balanced_accuracy"][
                    "forgetting_by_domain"
                ]
                for report in reports[job]
            ],
            dtype=float,
        ) * 100.0
        color = COLORS.get(job, "#777777")
        marker = MARKERS[index % len(MARKERS)]
        final_mean = np.mean(final_values, axis=0)
        final_std = np.std(final_values, axis=0)
        forgetting_mean = np.mean(forgetting_values, axis=0)
        forgetting_std = np.std(forgetting_values, axis=0)
        label = SHORT_LABELS.get(job, job)
        axes[0].errorbar(
            x,
            final_mean,
            yerr=final_std,
            marker=marker,
            linewidth=1.7,
            capsize=2,
            color=color,
            label=label,
        )
        axes[1].errorbar(
            x,
            forgetting_mean,
            yerr=forgetting_std,
            marker=marker,
            linewidth=1.7,
            capsize=2,
            color=color,
            label=label,
        )
    axes[0].set_xticks(x, domain_labels)
    axes[1].set_xticks(x, domain_labels)
    axes[0].set_title("Final performance by condition")
    axes[1].set_title("Forgetting by previously learned condition")
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[1].set_ylabel("Forgetting (percentage points)")
    for axis in axes:
        axis.set_xlabel("Operating condition")
        _decorate_axis(axis, zero_line=axis is axes[1])
    axes[0].legend(frameon=False, ncol=2)
    figure.suptitle("Condition-wise retention analysis", fontsize=12, fontweight="bold")
    figure.tight_layout()
    return figure


def _plot_class_recall(
    reports: list[dict[str, Any]],
    domains: list[int],
    domain_labels: list[str],
    class_names: list[str],
    job: str,
) -> plt.Figure:
    matrices: list[np.ndarray] = []
    for report in reports:
        domain_metrics = report["final_stage_metrics"]
        matrices.append(
            np.asarray(
                [
                    [
                        domain_metrics[str(domain)]["per_class_accuracy"][str(label)]
                        for label in range(len(class_names))
                    ]
                    for domain in domains
                ],
                dtype=float,
            )
        )
    mean = np.mean(np.asarray(matrices), axis=0) * 100.0
    figure, axis = plt.subplots(figsize=(10.6, 5.2))
    image = axis.imshow(mean, cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")
    axis.set_xticks(np.arange(len(class_names)), class_names, rotation=28, ha="right")
    axis.set_yticks(np.arange(len(domains)), domain_labels)
    axis.set_xlabel("Fault class")
    axis.set_ylabel("Evaluation condition")
    axis.set_title(
        f"Final class recall of {SHORT_LABELS.get(job, job)} (mean over {len(reports)} seeds)"
    )
    for row in range(mean.shape[0]):
        for column in range(mean.shape[1]):
            value = float(mean[row, column])
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value >= 65.0 else "black",
                fontsize=6.5,
            )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Recall (%)")
    figure.tight_layout()
    return figure


def _save(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[str]:
    paths: list[str] = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def main() -> int:
    args = parse_args()
    _style()
    root = Path(args.root)
    requested_jobs = (
        {value.strip() for value in args.jobs.split(",") if value.strip()}
        if args.jobs
        else None
    )
    reports = _load_reports(root, requested_jobs)
    domains, class_names = _validate_contract(reports)
    domain_labels = _domain_labels(reports, domains, style=args.domain_label_style)
    jobs = _job_order(reports, Path(args.matrix))
    output_dir = Path(args.output_dir) if args.output_dir else root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = next(iter(reports.values()))[0].get("dataset", "")
    _write_domain_mapping(
        output_dir / "domain_mapping.csv",
        dataset,
        domains,
        domain_labels,
    )
    formats = tuple(
        value.strip().lower() for value in args.formats.split(",") if value.strip()
    )
    if not formats:
        raise ValueError("At least one figure format is required.")
    for path in output_dir.iterdir():
        if path.suffix.lower().lstrip(".") not in formats:
            continue
        if any(path.stem.startswith(f"fig{index:02d}_") for index in range(1, 7)):
            path.unlink()

    summary_csv = output_dir / "p1_visual_summary.csv"
    _write_summary_csv(summary_csv, reports, jobs)
    artifacts: dict[str, list[str]] = {}

    core_jobs = [job for job in CORE_JOBS if job in reports]
    if core_jobs:
        artifacts["fig01_core_method_comparison"] = _save(
            _plot_metric_comparison(
                reports,
                core_jobs,
                title="Continual-learning method comparison",
            ),
            output_dir,
            "fig01_core_method_comparison",
            formats,
            args.dpi,
        )
        artifacts["fig02_stage_trajectory"] = _save(
            _plot_stage_trajectory(reports, core_jobs, domains, domain_labels),
            output_dir,
            "fig02_stage_trajectory",
            formats,
            args.dpi,
        )
        artifacts["fig04_condition_retention"] = _save(
            _plot_domain_profiles(reports, core_jobs, domains, domain_labels),
            output_dir,
            "fig04_condition_retention",
            formats,
            args.dpi,
        )

    ablation_jobs = [job for job in ABLATION_JOBS if job in reports]
    if len(ablation_jobs) > 1:
        artifacts["fig05_ablation_comparison"] = _save(
            _plot_metric_comparison(
                reports,
                ablation_jobs,
                title="Ablation analysis of continual semantic alignment",
            ),
            output_dir,
            "fig05_ablation_comparison",
            formats,
            args.dpi,
        )

    focus_job = "se_gscl_full" if "se_gscl_full" in reports else jobs[0]
    artifacts["fig03_continual_accuracy_matrix"] = _save(
        _plot_accuracy_matrix(reports[focus_job], domains, domain_labels, focus_job),
        output_dir,
        "fig03_continual_accuracy_matrix",
        formats,
        args.dpi,
    )
    artifacts["fig06_final_class_recall"] = _save(
        _plot_class_recall(
            reports[focus_job],
            domains,
            domain_labels,
            class_names,
            focus_job,
        ),
        output_dir,
        "fig06_final_class_recall",
        formats,
        args.dpi,
    )

    manifest = {
        "status": "ok",
        "source_root": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "domains": domains,
        "class_names": class_names,
        "jobs": {
            job: {
                "label": SHORT_LABELS.get(job, job),
                "seeds": [int(report["_seed"]) for report in reports[job]],
                "num_seeds": len(reports[job]),
            }
            for job in jobs
        },
        "focus_job": focus_job,
        "summary_csv": str(summary_csv.resolve()),
        "artifacts": artifacts,
        "note": "Error bars and shaded bands show population standard deviation across seeds.",
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
