"""Combine per-strategy P1 sequence reports into one feedback JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    reports: dict[str, dict[str, object]] = {}
    for report_path in sorted(root.glob("*/p1_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        strategy = str(report["strategy"])
        final_metrics = report["final_stage_metrics"]
        per_class_minimum = min(
            min(values["per_class_accuracy"].values())
            for values in final_metrics.values()
        )
        reports[strategy] = {
            "output_dir": str(report_path.parent.resolve()),
            "accuracy": report["sequence_summary"]["accuracy"],
            "balanced_accuracy": report["sequence_summary"]["balanced_accuracy"],
            "final_metrics_by_domain": final_metrics,
            "minimum_final_class_recall": float(per_class_minimum),
            "memory_samples": int(report["replay_samples"]),
        }
    if not reports:
        raise ValueError(f"No strategy reports were found under {root}.")

    deltas: dict[str, float] = {}
    if "full" in reports:
        full = reports["full"]["balanced_accuracy"]["final_average_accuracy"]
        for baseline in ("sequential", "balanced_replay"):
            if baseline in reports:
                baseline_value = reports[baseline]["balanced_accuracy"][
                    "final_average_accuracy"
                ]
                deltas[f"full_minus_{baseline}_balanced_accuracy"] = float(
                    full - baseline_value
                )
    comparison = {
        "status": "ok",
        "root": str(root.resolve()),
        "strategies": reports,
        "deltas": deltas,
        "feedback_request": [
            "Return this comparison JSON.",
            "If full is not best, also return full/p1_report.json.",
        ],
        "note": "Single-seed diagnostic comparison; not a paper result.",
    }
    output_path = root / "comparison.json"
    output_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
