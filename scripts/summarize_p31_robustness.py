"""Aggregate repeated P3.1.1 runs into a robustness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _stats(values: Iterable[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("Cannot summarize an empty metric series.")
    return {
        "mean": mean(numbers),
        "std": pstdev(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _nested_stats(
    reports: list[dict[str, Any]],
    section: str,
    metric: str,
) -> dict[str, dict[str, float]]:
    keys = sorted(
        {
            key
            for report in reports
            for key in report[section][metric]
        }
    )
    return {
        key: _stats(
            report[section][metric][key]
            for report in reports
        )
        for key in keys
    }


def _domain_accuracy_stats(
    reports: list[dict[str, Any]],
    section: str,
) -> dict[str, dict[str, float]]:
    domains = sorted(
        {
            domain
            for report in reports
            for domain in report[section]["domain_metrics"]
        },
        key=int,
    )
    return {
        domain: _stats(
            report[section]["domain_metrics"][domain]["accuracy"]
            for report in reports
        )
        for domain in domains
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# P3.1.2 continuous-prompt robustness summary",
        "",
        "| Seed | Qwen accuracy | Balanced accuracy | "
        "Auxiliary probe | Upstream | Agreement |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            "| {seed} | {qwen:.4f} | {balanced:.4f} | {auxiliary:.4f} | "
            "{upstream:.4f} | {agreement:.4f} |".format(**run)
        )
    aggregate = summary["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "- Qwen accuracy: "
            f"{aggregate['qwen_accuracy']['mean']:.4f} +/- "
            f"{aggregate['qwen_accuracy']['std']:.4f}",
            "- Balanced accuracy: "
            f"{aggregate['qwen_balanced_accuracy']['mean']:.4f} +/- "
            f"{aggregate['qwen_balanced_accuracy']['std']:.4f}",
            "- Auxiliary probe accuracy: "
            f"{aggregate['auxiliary_accuracy']['mean']:.4f} +/- "
            f"{aggregate['auxiliary_accuracy']['std']:.4f}",
            "- Qwen-upstream agreement: "
            f"{aggregate['qwen_upstream_agreement']['mean']:.4f} +/- "
            f"{aggregate['qwen_upstream_agreement']['std']:.4f}",
            "",
            "## Stability checks",
            "",
            f"- All generated labels valid: "
            f"{summary['stability_checks']['all_labels_valid']}",
            f"- No class has zero recall: "
            f"{summary['stability_checks']['no_class_collapse']}",
            f"- Minimum run accuracy: "
            f"{summary['stability_checks']['minimum_qwen_accuracy']:.4f}",
            f"- Minimum domain accuracy: "
            f"{summary['stability_checks']['minimum_domain_accuracy']:.4f}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    paths = [Path(value) for value in args.reports]
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    if len(reports) < 2:
        raise ValueError("Robustness aggregation requires at least two runs.")
    for path, report in zip(paths, reports):
        if report.get("status") != "ok":
            raise ValueError(f"Non-successful report: {path}")
        if "training_only_auxiliary_probe_metrics" not in report:
            raise ValueError(f"Report predates P3.1.1: {path}")
        if "qwen_upstream_paired_comparison" not in report:
            raise ValueError(
                f"Report lacks paired comparison; rerun current code: {path}"
            )

    runs = []
    for path, report in zip(paths, reports):
        generated = report["continuous_prompt_metrics"]
        auxiliary = report["training_only_auxiliary_probe_metrics"]
        upstream = report["upstream_fused_baseline"]
        paired = report["qwen_upstream_paired_comparison"]
        runs.append(
            {
                "seed": int(report["training"]["seed"]),
                "report": str(path.resolve()),
                "samples": int(generated["samples"]),
                "qwen": float(generated["accuracy"]),
                "balanced": float(generated["balanced_accuracy"]),
                "auxiliary": float(auxiliary["accuracy"]),
                "upstream": float(upstream["accuracy"]),
                "agreement": float(paired["prediction_agreement_rate"]),
                "qwen_only_correct": int(paired["qwen_only_correct"]),
                "upstream_only_correct": int(
                    paired["upstream_only_correct"]
                ),
            }
        )

    domain_stats = _domain_accuracy_stats(
        reports,
        "continuous_prompt_metrics",
    )
    summary = {
        "status": "ok",
        "stage": "P3.1.2 multi-seed continuous-prompt robustness",
        "num_runs": len(reports),
        "runs": runs,
        "aggregate": {
            "qwen_accuracy": _stats(run["qwen"] for run in runs),
            "qwen_balanced_accuracy": _stats(
                run["balanced"] for run in runs
            ),
            "auxiliary_accuracy": _stats(
                run["auxiliary"] for run in runs
            ),
            "upstream_accuracy": _stats(
                run["upstream"] for run in runs
            ),
            "qwen_upstream_agreement": _stats(
                run["agreement"] for run in runs
            ),
            "qwen_minus_upstream_accuracy": _stats(
                report["qwen_upstream_paired_comparison"][
                    "qwen_minus_upstream_accuracy"
                ]
                for report in reports
            ),
            "per_class_recall": _nested_stats(
                reports,
                "continuous_prompt_metrics",
                "per_class_recall",
            ),
            "domain_accuracy": domain_stats,
        },
        "stability_checks": {
            "all_labels_valid": all(
                report["continuous_prompt_metrics"]["valid_label_rate"] == 1.0
                for report in reports
            ),
            "no_class_collapse": all(
                min(
                    report["continuous_prompt_metrics"][
                        "per_class_recall"
                    ].values()
                )
                > 0.0
                for report in reports
            ),
            "minimum_qwen_accuracy": min(run["qwen"] for run in runs),
            "minimum_domain_accuracy": min(
                values["min"] for values in domain_stats.values()
            ),
        },
        "note": (
            "All runs train only on source-condition labels. Test labels are "
            "used only for post-generation evaluation."
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "p312_robustness_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "p312_robustness_summary.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
