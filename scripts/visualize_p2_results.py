"""Create publication-ready figures for the P2 local symptom probe."""

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
import numpy as np  # noqa: E402


BRANCHES = ("global", "local", "fused")
BRANCH_LABELS = {
    "global": "Global identity",
    "local": "Local symptoms",
    "fused": "Hierarchical fusion",
}
BRANCH_COLORS = {
    "global": "#0072B2",
    "local": "#009E73",
    "fused": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
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


def _confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for target, prediction in zip(labels, predictions):
        matrix[int(target), int(prediction)] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums > 0,
    )


def _decorate_metric_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_ylim(0.0, 115.0)


def _plot_domain_metrics(
    report: dict[str, Any],
    domains: list[int],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(domains))
    width = 0.23
    for offset, branch in enumerate(BRANCHES):
        values = np.asarray(
            [
                report["domain_metrics"][str(domain)][branch][
                    "balanced_accuracy"
                ]
                for domain in domains
            ],
            dtype=float,
        ) * 100.0
        positions = x + (offset - 1) * width
        bars = axis.bar(
            positions,
            values,
            width=width,
            color=BRANCH_COLORS[branch],
            edgecolor="black",
            linewidth=0.45,
            label=BRANCH_LABELS[branch],
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    axis.set_xticks(x, [f"Condition D{domain}" for domain in domains])
    axis.set_ylabel("Balanced accuracy (%)")
    axis.set_title(
        "Global identity, local symptoms, and hierarchical fusion",
        fontweight="bold",
    )
    axis.legend(ncol=3, loc="lower center", frameon=False)
    _decorate_metric_axis(axis)
    figure.tight_layout()
    return figure


def _draw_confusion(
    axis: plt.Axes,
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    image = axis.imshow(matrix * 100.0, vmin=0.0, vmax=100.0, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title, fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column] * 100.0
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value >= 55.0 else "black",
                fontsize=8,
            )
    return image


def _plot_confusions(
    arrays: dict[str, np.ndarray],
    domains: list[int],
    class_names: list[str],
) -> plt.Figure:
    columns = min(3, len(domains) + 1)
    rows = int(np.ceil((len(domains) + 1) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.6 * rows),
        squeeze=False,
    )
    labels = arrays["labels"]
    predictions = arrays["fused_probabilities"].argmax(axis=1)
    panels: list[tuple[np.ndarray, str]] = [
        (
            _confusion_matrix(labels, predictions, len(class_names)),
            "All conditions",
        )
    ]
    for domain in domains:
        mask = arrays["domains"] == domain
        panels.append(
            (
                _confusion_matrix(
                    labels[mask],
                    predictions[mask],
                    len(class_names),
                ),
                f"Condition D{domain}",
            )
        )
    image = None
    for axis, (matrix, title) in zip(axes.flat, panels):
        image = _draw_confusion(axis, matrix, class_names, title)
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.03,
            label="Recall-normalized samples (%)",
        )
    figure.suptitle(
        "Hierarchical-fusion confusion matrices",
        fontweight="bold",
        y=1.01,
    )
    figure.subplots_adjust(wspace=0.35, hspace=0.45)
    return figure


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(values * np.log(values)).sum(axis=1)
    if values.shape[1] <= 1:
        return np.zeros(len(values), dtype=np.float64)
    return entropy / np.log(values.shape[1])


def _plot_uncertainty(
    arrays: dict[str, np.ndarray],
    domains: list[int],
) -> plt.Figure:
    global_predictions = arrays["global_probabilities"].argmax(axis=1)
    local_predictions = arrays["local_probabilities"].argmax(axis=1)
    fused_predictions = arrays["fused_probabilities"].argmax(axis=1)
    correct = fused_predictions == arrays["labels"]
    entropy = _normalized_entropy(arrays["fused_probabilities"])
    confidence = arrays["fused_probabilities"].max(axis=1)

    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))
    axes[0].boxplot(
        [confidence[correct] * 100.0, confidence[~correct] * 100.0],
        patch_artist=True,
        boxprops={"facecolor": "#56B4E9", "alpha": 0.8},
        medianprops={"color": "black"},
    )
    axes[0].set_xticks(
        (1, 2),
        (
            f"Correct\n(n={int(correct.sum())})",
            f"Incorrect\n(n={int((~correct).sum())})",
        ),
    )
    axes[0].set_ylabel("Fused confidence (%)")
    axes[0].set_title("Confidence separation", fontweight="bold")

    axes[1].boxplot(
        [entropy[correct], entropy[~correct]],
        patch_artist=True,
        boxprops={"facecolor": "#E69F00", "alpha": 0.8},
        medianprops={"color": "black"},
    )
    axes[1].set_xticks(
        (1, 2),
        (
            f"Correct\n(n={int(correct.sum())})",
            f"Incorrect\n(n={int((~correct).sum())})",
        ),
    )
    axes[1].set_ylabel("Normalized entropy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Predictive uncertainty", fontweight="bold")

    agreement = np.asarray(
        [
            np.mean(
                global_predictions[arrays["domains"] == domain]
                == local_predictions[arrays["domains"] == domain]
            )
            for domain in domains
        ],
        dtype=float,
    ) * 100.0
    bars = axes[2].bar(
        range(len(domains)),
        agreement,
        color="#009E73",
        edgecolor="black",
        linewidth=0.45,
    )
    axes[2].set_xticks(range(len(domains)), [f"D{value}" for value in domains])
    axes[2].set_ylabel("Agreement (%)")
    axes[2].set_ylim(0.0, 105.0)
    axes[2].set_title("Global-local agreement", fontweight="bold")
    for bar, value in zip(bars, agreement):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.tight_layout()
    return figure


def _plot_symptom_evidence(
    arrays: dict[str, np.ndarray],
    class_names: list[str],
    symptom_names: list[str],
) -> plt.Figure:
    labels = arrays["labels"]
    joint = arrays["symptom_joint_probabilities"]
    evidence = np.stack(
        [
            joint[labels == class_id].mean(axis=0)
            for class_id in range(len(class_names))
        ],
        axis=0,
    )
    row_sums = evidence.sum(axis=1, keepdims=True)
    evidence = np.divide(
        evidence,
        row_sums,
        out=np.zeros_like(evidence),
        where=row_sums > 0,
    )
    width = max(8.5, 0.66 * len(symptom_names))
    figure, axis = plt.subplots(figsize=(width, 3.7))
    image = axis.imshow(evidence * 100.0, vmin=0.0, cmap="YlGnBu", aspect="auto")
    axis.set_xticks(
        range(len(symptom_names)),
        symptom_names,
        rotation=42,
        ha="right",
    )
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Local symptom prototype")
    axis.set_ylabel("True fault class")
    axis.set_title(
        "Average local symptom evidence by true class",
        fontweight="bold",
    )
    figure.colorbar(
        image,
        ax=axis,
        fraction=0.025,
        pad=0.02,
        label="Within-class evidence (%)",
    )
    figure.tight_layout()
    return figure


def _plot_physical_grounding(
    arrays: dict[str, np.ndarray],
    class_names: list[str],
    symptom_names: list[str],
) -> plt.Figure:
    labels = arrays["labels"]
    targets = arrays["physical_symptom_targets"]
    predicted = arrays["symptom_probabilities"]
    target_matrix = np.stack(
        [
            targets[labels == class_id].mean(axis=0)
            for class_id in range(len(class_names))
        ],
        axis=0,
    )
    predicted_matrix = np.stack(
        [
            predicted[labels == class_id].mean(axis=0)
            for class_id in range(len(class_names))
        ],
        axis=0,
    )
    width = max(11.0, 0.75 * len(symptom_names))
    figure, axes = plt.subplots(1, 2, figsize=(width, 3.9), sharey=True)
    image = None
    for axis, matrix, title in (
        (axes[0], target_matrix, "Physics-derived soft targets"),
        (axes[1], predicted_matrix, "Learned symptom probabilities"),
    ):
        image = axis.imshow(
            matrix * 100.0,
            vmin=0.0,
            vmax=100.0,
            cmap="YlGnBu",
            aspect="auto",
        )
        axis.set_xticks(
            range(len(symptom_names)),
            symptom_names,
            rotation=45,
            ha="right",
        )
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Local symptom prototype")
        axis.set_title(title, fontweight="bold")
    axes[0].set_ylabel("True fault class")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
            label="Mean activation (%)",
        )
    figure.suptitle(
        "Physical grounding of local symptom semantics",
        fontweight="bold",
        y=1.02,
    )
    figure.subplots_adjust(wspace=0.15, bottom=0.33)
    return figure


def _plot_p22_training_and_gate(
    report: dict[str, Any],
    domains: list[int],
    arrays: dict[str, np.ndarray],
) -> plt.Figure:
    history = report["history"]
    epochs = np.asarray([row["epoch"] for row in history])
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))

    for key, label, color in (
        ("classification", "Classification", "#0072B2"),
        ("physics", "Physics BCE", "#009E73"),
        ("within_class_distribution", "Within-class KL", "#D55E00"),
    ):
        values = np.asarray([row.get(key, np.nan) for row in history])
        axes[0].plot(
            epochs,
            values,
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("P2.2 loss components", fontweight="bold")
    axes[0].legend(frameon=False)

    validation_loss = np.asarray(
        [row.get("validation_loss", np.nan) for row in history]
    )
    validation_accuracy = np.asarray(
        [row.get("validation_balanced_accuracy", np.nan) for row in history]
    ) * 100.0
    axes[1].plot(
        epochs,
        validation_loss,
        marker="o",
        color="#CC79A7",
        linewidth=1.8,
        label="Validation loss",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Validation model selection", fontweight="bold")
    accuracy_axis = axes[1].twinx()
    accuracy_axis.plot(
        epochs,
        validation_accuracy,
        marker="s",
        color="#E69F00",
        linewidth=1.5,
        label="Balanced accuracy",
    )
    accuracy_axis.set_ylabel("Balanced accuracy (%)")
    best_epoch = report["model_selection"]["best_epoch"]
    if best_epoch is not None:
        axes[1].axvline(
            best_epoch,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )

    mean_weights = [
        report["domain_metrics"][str(domain)]["fusion_diagnostics"][
            "mean_local_weight"
        ]
        * 100.0
        for domain in domains
    ]
    activation_rates = []
    for domain in domains:
        mask = arrays["domains"] == domain
        activation_rates.append(
            float(np.mean(arrays["fusion_local_weights"][mask] > 0.0))
            * 100.0
        )
    x = np.arange(len(domains))
    width = 0.36
    mean_bars = axes[2].bar(
        x - width / 2,
        mean_weights,
        width,
        color="#56B4E9",
        edgecolor="black",
        linewidth=0.4,
        label="Mean local weight",
    )
    activation_bars = axes[2].bar(
        x + width / 2,
        activation_rates,
        width,
        color="#D55E00",
        edgecolor="black",
        linewidth=0.4,
        label="Local activation rate",
    )
    axes[2].set_xticks(x, [f"D{domain}" for domain in domains])
    gate_maximum = max([*mean_weights, *activation_rates, 1.0])
    axes[2].set_ylim(0.0, max(5.0, gate_maximum * 1.35))
    axes[2].set_ylabel("%")
    axes[2].set_title("Reliability-gated fusion", fontweight="bold")
    axes[2].legend(frameon=False)
    for bars in (mean_bars, activation_bars):
        for bar in bars:
            value = float(bar.get_height())
            if value > 0.0:
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.04 * gate_maximum,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Semantic preservation and validation-calibrated fusion",
        fontweight="bold",
        y=1.02,
    )
    figure.tight_layout()
    return figure


def main() -> int:
    args = parse_args()
    _style()
    root = Path(args.root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((root / "p2_report.json").read_text(encoding="utf-8"))
    loaded = np.load(root / "p2_outputs.npz")
    arrays = {key: loaded[key] for key in loaded.files}
    domains = [int(value) for value in report["domains"]]
    class_names = [str(value) for value in report["class_names"]]
    symptom_names = [str(value) for value in report["symptom_names"]]
    formats = tuple(
        value.strip().lower()
        for value in args.formats.split(",")
        if value.strip()
    )
    figures: dict[str, list[str]] = {}
    figures["domain_metrics"] = _save(
        _plot_domain_metrics(report, domains),
        output_dir,
        "p2_global_local_fusion_by_domain",
        formats,
        args.dpi,
    )
    figures["confusion_matrices"] = _save(
        _plot_confusions(arrays, domains, class_names),
        output_dir,
        "p2_fused_confusion_matrices",
        formats,
        args.dpi,
    )
    figures["uncertainty"] = _save(
        _plot_uncertainty(arrays, domains),
        output_dir,
        "p2_uncertainty_and_agreement",
        formats,
        args.dpi,
    )
    figures["symptom_evidence"] = _save(
        _plot_symptom_evidence(arrays, class_names, symptom_names),
        output_dir,
        "p2_symptom_evidence_heatmap",
        formats,
        args.dpi,
    )
    if {
        "physical_symptom_targets",
        "symptom_probabilities",
    }.issubset(arrays):
        figures["physical_grounding"] = _save(
            _plot_physical_grounding(
                arrays,
                class_names,
                symptom_names,
            ),
            output_dir,
            "p21_physical_target_vs_prediction",
            formats,
            args.dpi,
        )
    if report.get("semantic_guard"):
        figures["semantic_guard"] = _save(
            _plot_p22_training_and_gate(report, domains, arrays),
            output_dir,
            "p22_semantic_guard_and_reliability_gate",
            formats,
            args.dpi,
        )
    manifest = {
        "status": "ok",
        "source": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "figures": figures,
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
