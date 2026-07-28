"""Create publication-ready P1 continual-learning diagnostic figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "se_gscl_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


STRATEGY_ORDER = ("sequential", "balanced_replay", "full")
STRATEGY_LABELS = {
    "sequential": "Sequential",
    "balanced_replay": "Balanced replay",
    "full": "SE-GSCL",
}
STRATEGY_COLORS = {
    "sequential": "#7A7A7A",
    "balanced_replay": "#0072B2",
    "full": "#D55E00",
}
CLASS_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
DOMAIN_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--class-names", default="")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
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


def _save(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[str]:
    written: list[str] = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(str(path.resolve()))
    plt.close(figure)
    return written


def _load_inputs(
    root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    comparison_path = root / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(f"Missing comparison JSON: {comparison_path}")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    reports: dict[str, dict[str, Any]] = {}
    for report_path in sorted(root.glob("*/p1_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports[str(report["strategy"])] = report
    return comparison, reports


def _ordered_strategies(comparison: dict[str, Any]) -> list[str]:
    present = comparison["strategies"]
    return [strategy for strategy in STRATEGY_ORDER if strategy in present]


def _infer_domains(
    comparison: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> list[int]:
    if reports:
        return [int(value) for value in next(iter(reports.values()))["domains"]]
    first = next(iter(comparison["strategies"].values()))
    count = len(first["balanced_accuracy"]["final_by_domain"])
    return list(range(count))


def _infer_class_names(
    comparison: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    override: str,
) -> list[str]:
    if override.strip():
        return [value.strip() for value in override.split(",") if value.strip()]
    if reports:
        return [str(value) for value in next(iter(reports.values()))["class_names"]]
    first = next(iter(comparison["strategies"].values()))
    domain_metrics = next(iter(first["final_metrics_by_domain"].values()))
    count = len(domain_metrics["per_class_accuracy"])
    return [f"Class {index}" for index in range(count)]


def _decorate_axis(axis: plt.Axes, *, percent: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    if percent:
        axis.set_ylim(0.0, 105.0)


def _plot_strategy_summary(
    comparison: dict[str, Any],
    strategies: list[str],
) -> plt.Figure:
    metrics = (
        ("final_average_accuracy", "Final balanced accuracy", True),
        ("average_incremental_accuracy", "Average incremental accuracy", True),
        ("average_forgetting", "Average forgetting", False),
        ("average_backward_transfer", "Backward transfer", False),
    )
    figure, axes = plt.subplots(1, 4, figsize=(12.4, 3.0))
    x = np.arange(len(strategies))
    for axis, (key, title, accuracy_scale) in zip(axes, metrics):
        values = np.asarray(
            [
                comparison["strategies"][strategy]["balanced_accuracy"][key]
                for strategy in strategies
            ],
            dtype=float,
        ) * 100.0
        bars = axis.bar(
            x,
            values,
            width=0.62,
            color=[STRATEGY_COLORS[strategy] for strategy in strategies],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(title)
        axis.set_xticks(x, [STRATEGY_LABELS[value] for value in strategies], rotation=22)
        axis.set_ylabel("%")
        _decorate_axis(axis, percent=accuracy_scale)
        if not accuracy_scale:
            if key == "average_forgetting" and np.all(values >= 0.0):
                axis.set_ylim(-1.0, max(5.0, float(np.max(values)) * 1.35))
            else:
                span = max(5.0, float(np.max(np.abs(values))) * 1.35)
                axis.set_ylim(-span, span)
        for bar, value in zip(bars, values):
            offset = 1.0 if value >= 0 else -1.0
            vertical = "bottom" if value >= 0 else "top"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.1f}",
                ha="center",
                va=vertical,
                fontsize=7.5,
            )
    figure.suptitle("Continual-learning strategy comparison", fontweight="bold")
    figure.tight_layout()
    return figure


def _plot_stage_accuracy(
    comparison: dict[str, Any],
    strategies: list[str],
    domains: list[int],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(domains))
    for strategy in strategies:
        values = (
            np.asarray(
                comparison["strategies"][strategy]["balanced_accuracy"][
                    "stage_seen_average"
                ],
                dtype=float,
            )
            * 100.0
        )
        axis.plot(
            x,
            values,
            marker="o",
            markersize=5,
            linewidth=2,
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
        )
    axis.set_xticks(x, [f"After D{domain}" for domain in domains])
    axis.set_xlabel("Continual training stage")
    axis.set_ylabel("Seen-domain balanced accuracy (%)")
    axis.set_title("Accuracy evolution over the condition stream", fontweight="bold")
    _decorate_axis(axis, percent=True)
    all_values = np.concatenate(
        [
            np.asarray(
                comparison["strategies"][strategy]["balanced_accuracy"][
                    "stage_seen_average"
                ],
                dtype=float,
            )
            * 100.0
            for strategy in strategies
        ]
    )
    lower = max(0.0, float(np.floor((np.min(all_values) - 5.0) / 10.0) * 10.0))
    axis.set_ylim(lower, 102.0)
    axis.legend(frameon=False, ncol=len(strategies), loc="lower right")
    figure.tight_layout()
    return figure


def _plot_final_domains(
    comparison: dict[str, Any],
    strategies: list[str],
    domains: list[int],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    x = np.arange(len(domains))
    width = 0.78 / len(strategies)
    for index, strategy in enumerate(strategies):
        values = (
            np.asarray(
                comparison["strategies"][strategy]["balanced_accuracy"][
                    "final_by_domain"
                ],
                dtype=float,
            )
            * 100.0
        )
        axis.bar(
            x + (index - (len(strategies) - 1) / 2) * width,
            values,
            width=width,
            color=STRATEGY_COLORS[strategy],
            edgecolor="black",
            linewidth=0.45,
            label=STRATEGY_LABELS[strategy],
        )
    axis.set_xticks(x, [f"D{domain}" for domain in domains])
    axis.set_xlabel("Evaluation condition")
    axis.set_ylabel("Final balanced accuracy (%)")
    axis.set_title("Final retention and adaptation by condition", fontweight="bold")
    _decorate_axis(axis, percent=True)
    axis.legend(frameon=False, ncol=len(strategies), loc="lower right")
    figure.tight_layout()
    return figure


def _annotated_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    vmin: float = 0.0,
    vmax: float = 100.0,
    mask: np.ndarray | None = None,
) -> Any:
    values = np.ma.array(matrix, mask=mask)
    image = axis.imshow(values, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    axis.set_xticks(np.arange(len(xlabels)), xlabels, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(ylabels)), ylabels)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if mask is not None and bool(mask[row, column]):
                continue
            value = float(matrix[row, column])
            color = "white" if value > 66 else "black"
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    return image


def _plot_per_class_recall(
    comparison: dict[str, Any],
    strategies: list[str],
    domains: list[int],
    class_names: list[str],
) -> plt.Figure:
    figure, axes = plt.subplots(
        1,
        len(strategies),
        figsize=(4.0 * len(strategies), 3.5),
        squeeze=False,
    )
    image = None
    for axis, strategy in zip(axes[0], strategies):
        domain_metrics = comparison["strategies"][strategy][
            "final_metrics_by_domain"
        ]
        matrix = np.asarray(
            [
                [
                    domain_metrics[str(domain)]["per_class_accuracy"][str(label)]
                    for label in range(len(class_names))
                ]
                for domain in domains
            ],
            dtype=float,
        ) * 100.0
        image = _annotated_heatmap(
            axis,
            matrix,
            xlabels=class_names,
            ylabels=[f"D{domain}" for domain in domains],
            title=STRATEGY_LABELS[strategy],
        )
        axis.set_xlabel("Fault class")
        if axis is axes[0, 0]:
            axis.set_ylabel("Condition")
    figure.suptitle("Final class recall by condition", fontweight="bold")
    figure.subplots_adjust(left=0.07, right=0.90, bottom=0.20, top=0.80, wspace=0.28)
    if image is not None:
        color_axis = figure.add_axes([0.925, 0.22, 0.012, 0.56])
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.set_label("Recall (%)")
    return figure


def _plot_accuracy_matrices(
    reports: dict[str, dict[str, Any]],
    strategies: list[str],
    domains: list[int],
) -> plt.Figure | None:
    available = [strategy for strategy in strategies if strategy in reports]
    if not available:
        return None
    figure, axes = plt.subplots(
        1,
        len(available),
        figsize=(4.0 * len(available), 3.6),
        squeeze=False,
    )
    image = None
    mask = np.triu(np.ones((len(domains), len(domains)), dtype=bool), k=1)
    for axis, strategy in zip(axes[0], available):
        rows = reports[strategy]["stage_metrics"]
        matrix = np.asarray(
            [
                [
                    row["domain_metrics"][str(domain)]["balanced_accuracy"]
                    for domain in domains
                ]
                for row in rows
            ],
            dtype=float,
        ) * 100.0
        image = _annotated_heatmap(
            axis,
            matrix,
            xlabels=[f"Eval D{domain}" for domain in domains],
            ylabels=[f"After D{domain}" for domain in domains],
            title=STRATEGY_LABELS[strategy],
            mask=mask,
        )
        axis.set_xlabel("Evaluation condition")
        if axis is axes[0, 0]:
            axis.set_ylabel("Training stage")
    figure.suptitle("Seen-domain balanced-accuracy matrices", fontweight="bold")
    figure.subplots_adjust(left=0.07, right=0.90, bottom=0.21, top=0.80, wspace=0.30)
    if image is not None:
        color_axis = figure.add_axes([0.925, 0.22, 0.012, 0.56])
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.set_label("Balanced accuracy (%)")
    return figure


def _plot_full_losses(report: dict[str, Any]) -> plt.Figure | None:
    history = report.get("history", [])
    if not history:
        return None
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    keys = (
        ("loss", "Total", "#000000"),
        ("global_alignment", "Global alignment", "#0072B2"),
        ("cross_condition", "Cross-condition", "#D55E00"),
        ("global_relation", "Relation preservation", "#009E73"),
    )
    x = np.arange(len(history))
    for key, label, color in keys:
        values = np.asarray([float(row.get(key, 0.0)) for row in history])
        if key != "loss" and np.allclose(values, 0.0):
            continue
        axis.plot(x, values, linewidth=1.7, color=color, label=label)
    boundaries: list[int] = []
    previous = history[0].get("stage_index")
    for index, row in enumerate(history[1:], start=1):
        if row.get("stage_index") != previous:
            boundaries.append(index)
            previous = row.get("stage_index")
    for boundary in boundaries:
        axis.axvline(boundary - 0.5, color="#999999", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Optimization epoch across continual stages")
    axis.set_ylabel("Raw loss value")
    axis.set_title("SE-GSCL optimization trajectory", fontweight="bold")
    _decorate_axis(axis)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    return figure


def _load_stage_outputs(strategy_dir: Path) -> list[tuple[Path, dict[str, np.ndarray]]]:
    outputs: list[tuple[Path, dict[str, np.ndarray]]] = []
    for path in strategy_dir.glob("stage_outputs_after_domain_*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        outputs.append((path, values))
    outputs.sort(key=lambda item: int(item[1]["trained_domain"][0]))
    return outputs


def _embedding_projection(
    first: dict[str, np.ndarray],
    last: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    first_embeddings = np.asarray(first["embeddings"], dtype=float)
    last_embeddings = np.asarray(last["embeddings"], dtype=float)
    prototypes = np.asarray(last["prototypes"], dtype=float)
    combined = np.concatenate([first_embeddings, last_embeddings, prototypes])
    if len(combined) < 12:
        projected = PCA(n_components=2, random_state=seed).fit_transform(combined)
    else:
        perplexity = min(30.0, max(5.0, (len(combined) - 1) / 4.0))
        projected = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(combined)
    first_end = len(first_embeddings)
    last_end = first_end + len(last_embeddings)
    return projected[:first_end], projected[first_end:last_end], projected[last_end:]


def _plot_embedding_evolution(
    outputs: list[tuple[Path, dict[str, np.ndarray]]],
    class_names: list[str],
    seed: int,
) -> plt.Figure | None:
    if len(outputs) < 2:
        return None
    first = outputs[0][1]
    last = outputs[-1][1]
    first_xy, last_xy, prototype_xy = _embedding_projection(first, last, seed)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    panels = (
        (axes[0], first, first_xy, f"After D{int(first['trained_domain'][0])}"),
        (axes[1], last, last_xy, f"After D{int(last['trained_domain'][0])}"),
    )
    for axis, values, xy, title in panels:
        labels = values["labels"].astype(int)
        sample_domains = values["domains"].astype(int)
        for label, class_name in enumerate(class_names):
            for domain in sorted(np.unique(sample_domains).tolist()):
                mask = (labels == label) & (sample_domains == domain)
                if not np.any(mask):
                    continue
                axis.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    s=17,
                    alpha=0.68,
                    color=CLASS_COLORS[label % len(CLASS_COLORS)],
                    marker=DOMAIN_MARKERS[domain % len(DOMAIN_MARKERS)],
                    linewidths=0.2,
                    edgecolors="white",
                )
        for label, point in enumerate(prototype_xy):
            axis.scatter(
                point[0],
                point[1],
                marker="*",
                s=145,
                color=CLASS_COLORS[label % len(CLASS_COLORS)],
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_COLORS[index % len(CLASS_COLORS)],
            markeredgecolor="none",
            label=name,
        )
        for index, name in enumerate(class_names)
    ]
    domain_handles = [
        Line2D(
            [0],
            [0],
            marker=DOMAIN_MARKERS[int(domain) % len(DOMAIN_MARKERS)],
            color="#555555",
            linestyle="none",
            label=f"D{int(domain)}",
        )
        for domain in sorted(np.unique(last["domains"]).tolist())
    ]
    prototype_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="black",
        markerfacecolor="white",
        linestyle="none",
        markersize=10,
        label="Text prototype",
    )
    figure.legend(
        handles=class_handles + domain_handles + [prototype_handle],
        loc="lower center",
        ncol=min(6, len(class_handles) + len(domain_handles) + 1),
        frameon=False,
    )
    figure.suptitle(
        "Joint t-SNE of specialist embeddings and frozen text prototypes",
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.22, top=0.84, wspace=0.08)
    return figure


def _plot_final_confusions(
    output: dict[str, np.ndarray],
    class_names: list[str],
) -> plt.Figure:
    from sklearn.metrics import confusion_matrix

    domains = sorted(np.unique(output["domains"]).astype(int).tolist())
    figure, axes = plt.subplots(
        1,
        len(domains),
        figsize=(3.3 * len(domains), 3.2),
        squeeze=False,
    )
    image = None
    for axis, domain in zip(axes[0], domains):
        mask = output["domains"].astype(int) == domain
        matrix = confusion_matrix(
            output["labels"][mask],
            output["predictions"][mask],
            labels=np.arange(len(class_names)),
            normalize="true",
        ) * 100.0
        image = _annotated_heatmap(
            axis,
            matrix,
            xlabels=class_names,
            ylabels=class_names,
            title=f"D{domain}",
        )
        axis.set_xlabel("Predicted")
        if axis is axes[0, 0]:
            axis.set_ylabel("True")
    figure.suptitle("SE-GSCL final normalized confusion matrices", fontweight="bold")
    figure.subplots_adjust(left=0.06, right=0.90, bottom=0.22, top=0.80, wspace=0.34)
    if image is not None:
        color_axis = figure.add_axes([0.925, 0.22, 0.012, 0.56])
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.set_label("Samples (%)")
    return figure


def _plot_confidence_diagnostics(output: dict[str, np.ndarray]) -> plt.Figure:
    probabilities = np.asarray(output["probabilities"], dtype=float)
    confidence = probabilities.max(axis=1)
    correct = output["predictions"] == output["labels"]
    edges = np.linspace(0.0, 1.0, 11)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_accuracy = np.full(10, np.nan)
    bin_confidence = np.full(10, np.nan)
    bin_share = np.zeros(10)
    for index in range(10):
        if index == 9:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(mask):
            bin_accuracy[index] = np.mean(correct[mask]) * 100.0
            bin_confidence[index] = np.mean(confidence[mask]) * 100.0
            bin_share[index] = np.mean(mask) * 100.0

    order = np.argsort(-confidence)
    ranked_correct = correct[order].astype(float)
    coverage = np.arange(1, len(order) + 1) / len(order) * 100.0
    selective_accuracy = np.cumsum(ranked_correct) / np.arange(1, len(order) + 1) * 100.0

    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.7))
    left = axes[0]
    right_scale = left.twinx()
    left.bar(
        centers,
        bin_share,
        width=0.085,
        color="#BDBDBD",
        edgecolor="black",
        linewidth=0.4,
        label="Sample share",
    )
    right_scale.plot(
        centers,
        bin_accuracy,
        marker="o",
        color="#D55E00",
        linewidth=1.8,
        label="Accuracy",
    )
    right_scale.plot(
        centers,
        bin_confidence,
        marker="s",
        color="#0072B2",
        linewidth=1.5,
        label="Mean confidence",
    )
    left.set_xlabel("Specialist confidence bin")
    left.set_ylabel("Samples (%)")
    right_scale.set_ylabel("Accuracy / confidence (%)")
    left.set_title("Confidence reliability")
    left.spines["top"].set_visible(False)
    right_scale.spines["top"].set_visible(False)
    left.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right_scale.get_legend_handles_labels()
    left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        loc="upper left",
    )

    axes[1].plot(
        coverage,
        selective_accuracy,
        color="#009E73",
        linewidth=2,
    )
    axes[1].axhline(
        np.mean(correct) * 100.0,
        color="#777777",
        linestyle="--",
        linewidth=1,
        label="All-sample accuracy",
    )
    axes[1].set_xlabel("Coverage retained by confidence (%)")
    axes[1].set_ylabel("Selective accuracy (%)")
    axes[1].set_title("Accuracy-coverage trade-off")
    _decorate_axis(axes[1], percent=True)
    axes[1].set_xlim(0, 100)
    axes[1].legend(frameon=False)
    figure.suptitle(
        "Specialist confidence diagnostics for future LLM routing",
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def main() -> int:
    args = parse_args()
    _style()
    root = Path(args.root)
    comparison, reports = _load_inputs(root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(
        value.strip().lower() for value in args.formats.split(",") if value.strip()
    )
    strategies = _ordered_strategies(comparison)
    domains = _infer_domains(comparison, reports)
    class_names = _infer_class_names(
        comparison,
        reports,
        args.class_names,
    )
    artifacts: dict[str, list[str]] = {}
    known_stems = [f"fig{index:02d}" for index in range(1, 10)]
    for path in output_dir.iterdir():
        if path.suffix.lower().lstrip(".") not in formats:
            continue
        if any(path.stem.startswith(stem) for stem in known_stems):
            path.unlink()

    figures = {
        "fig01_strategy_summary": _plot_strategy_summary(comparison, strategies),
        "fig02_stage_seen_accuracy": _plot_stage_accuracy(
            comparison,
            strategies,
            domains,
        ),
        "fig03_final_domain_accuracy": _plot_final_domains(
            comparison,
            strategies,
            domains,
        ),
        "fig04_final_class_recall": _plot_per_class_recall(
            comparison,
            strategies,
            domains,
            class_names,
        ),
    }
    matrix_figure = _plot_accuracy_matrices(reports, strategies, domains)
    if matrix_figure is not None:
        figures["fig05_accuracy_matrices"] = matrix_figure
    if "full" in reports:
        loss_figure = _plot_full_losses(reports["full"])
        if loss_figure is not None:
            figures["fig06_full_loss_curves"] = loss_figure

    full_dir = root / "full"
    stage_outputs = _load_stage_outputs(full_dir) if full_dir.is_dir() else []
    if stage_outputs:
        figures["fig07_final_confusion_matrices"] = _plot_final_confusions(
            stage_outputs[-1][1],
            class_names,
        )
        figures["fig08_specialist_confidence"] = _plot_confidence_diagnostics(
            stage_outputs[-1][1]
        )
        embedding_figure = _plot_embedding_evolution(
            stage_outputs,
            class_names,
            args.seed,
        )
        if embedding_figure is not None:
            figures["fig09_embedding_evolution"] = embedding_figure

    for stem, figure in figures.items():
        artifacts[stem] = _save(
            figure,
            output_dir,
            stem,
            formats,
            args.dpi,
        )

    manifest = {
        "status": "ok",
        "source_root": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "diagnostic_scope": (
            "P1 lightweight specialist with frozen text-prototype classification; "
            "not generative LLM diagnosis."
        ),
        "strategies": strategies,
        "domains": domains,
        "class_names": class_names,
        "sample_level_outputs_available": bool(stage_outputs),
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
