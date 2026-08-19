"""Audit text anchors against initial-domain learned signal anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Seed directory containing se_gscl_full and wo_text_semantics.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def scalar(value: np.ndarray | float) -> float:
    return float(np.asarray(value).item())


def audit_job(job_dir: Path) -> dict[str, object]:
    report = json.loads((job_dir / "p1_report.json").read_text(encoding="utf-8"))
    final_domain = int(report["domains"][-1])
    arrays = np.load(job_dir / f"stage_outputs_after_domain_{final_domain}.npz")
    embeddings = normalize(np.asarray(arrays["embeddings"], dtype=np.float64))
    prototypes = normalize(np.asarray(arrays["prototypes"], dtype=np.float64))
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    domains = np.asarray(arrays["domains"], dtype=np.int64)
    class_names = [str(value) for value in arrays["class_names"].tolist()]

    prototype_cosine = prototypes @ prototypes.T
    off_diagonal = prototype_cosine[~np.eye(len(prototypes), dtype=bool)]
    similarities = embeddings @ prototypes.T
    correct = similarities[np.arange(len(labels)), labels]
    wrong = similarities.copy()
    wrong[np.arange(len(labels)), labels] = -np.inf
    margins = correct - np.max(wrong, axis=1)

    centroids = []
    compactness = []
    for class_id in range(len(class_names)):
        rows = embeddings[labels == class_id]
        centroid = normalize(rows.mean(axis=0, keepdims=True))[0]
        centroids.append(centroid)
        compactness.append(scalar(np.mean(rows @ centroid)))
    centroids_array = np.stack(centroids)
    centroid_cosine = centroids_array @ centroids_array.T
    centroid_off_diagonal = centroid_cosine[
        ~np.eye(len(centroids_array), dtype=bool)
    ]

    by_domain: dict[str, object] = {}
    for domain in report["domains"]:
        mask = domains == int(domain)
        by_domain[str(domain)] = {
            "samples": int(mask.sum()),
            "mean_correct_similarity": scalar(np.mean(correct[mask])),
            "mean_margin": scalar(np.mean(margins[mask])),
            "positive_margin_rate": scalar(np.mean(margins[mask] > 0.0)),
        }

    summary = report["sequence_summary"]["balanced_accuracy"]
    return {
        "prototype_source": report["prototype_source"],
        "samples": int(len(labels)),
        "class_names": class_names,
        "continual_metrics": {
            key: summary[key]
            for key in (
                "final_average_accuracy",
                "average_incremental_accuracy",
                "average_forgetting",
                "maximum_forgetting",
                "average_backward_transfer",
                "average_old_domain_retention",
            )
        },
        "prototype_geometry": {
            "cosine_matrix": prototype_cosine.tolist(),
            "mean_off_diagonal_cosine": scalar(np.mean(off_diagonal)),
            "maximum_off_diagonal_cosine": scalar(np.max(off_diagonal)),
            "minimum_off_diagonal_cosine": scalar(np.min(off_diagonal)),
        },
        "final_embedding_geometry": {
            "mean_correct_similarity": scalar(np.mean(correct)),
            "mean_classification_margin": scalar(np.mean(margins)),
            "positive_margin_rate": scalar(np.mean(margins > 0.0)),
            "mean_class_compactness": scalar(np.mean(compactness)),
            "maximum_centroid_cosine": scalar(np.max(centroid_off_diagonal)),
            "class_compactness": {
                name: value for name, value in zip(class_names, compactness)
            },
        },
        "by_domain": by_domain,
    }


def as_markdown(results: dict[str, dict[str, object]]) -> str:
    lines = [
        "# P1 semantic-anchor ablation audit",
        "",
        "| Variant | Final BA | Avg. forgetting | Prototype max cosine | "
        "Mean margin | Positive-margin rate | Class compactness |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        continual = result["continual_metrics"]
        prototype = result["prototype_geometry"]
        embedding = result["final_embedding_geometry"]
        lines.append(
            f"| {name} | {100 * continual['final_average_accuracy']:.2f}% | "
            f"{100 * continual['average_forgetting']:.2f}% | "
            f"{prototype['maximum_off_diagonal_cosine']:.4f} | "
            f"{embedding['mean_classification_margin']:.4f} | "
            f"{100 * embedding['positive_margin_rate']:.2f}% | "
            f"{embedding['mean_class_compactness']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Positive margin means that the sample is closer to its labelled "
            "prototype than to every competing prototype.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    jobs = {
        "SE-GSCL (text anchors)": root / "se_gscl_full",
        "Learned signal anchors": root / "wo_text_semantics",
    }
    results = {name: audit_job(path) for name, path in jobs.items()}
    output = Path(args.output) if args.output else root / "semantic_anchor_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(as_markdown(results), encoding="utf-8")
    print(markdown.read_text(encoding="utf-8"))
    print(f"JSON: {output.resolve()}")
    print(f"Markdown: {markdown.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
