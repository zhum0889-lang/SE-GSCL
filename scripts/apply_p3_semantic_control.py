"""Apply P3 semantic control to existing Qwen generations without rerunning."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.llm import apply_semantic_control, evaluate_llm_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-dir", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p3_dir = Path(args.p3_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else p3_dir / "semantic_control"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = p3_dir / "p3_predictions.jsonl"
    with source_path.open("r", encoding="utf-8") as handle:
        records: list[dict[str, Any]] = [
            json.loads(line) for line in handle if line.strip()
        ]
    controlled_records: list[dict[str, Any]] = []
    repair_counts: Counter[str] = Counter()
    repaired_samples = 0
    for record in records:
        controlled = apply_semantic_control(
            record["packet"],
            record.get("parsed_output"),
        )
        repairs = controlled["semantic_control_repairs"]
        if repairs:
            repaired_samples += 1
            repair_counts.update(str(value) for value in repairs)
        controlled_records.append(
            {
                **record,
                "controlled_output": controlled,
            }
        )
    controlled_for_metrics = [
        {
            **record,
            "parsed_output": record["controlled_output"],
        }
        for record in controlled_records
    ]
    summary = {
        "status": "ok",
        "stage": "P3.0.2 offline semantic consistency control",
        "source_predictions": str(source_path.resolve()),
        "samples": len(records),
        "raw_metrics": evaluate_llm_outputs(records),
        "controlled_metrics": evaluate_llm_outputs(controlled_for_metrics),
        "semantic_control": {
            "repair_rate": repaired_samples / max(1, len(records)),
            "repair_counts": dict(sorted(repair_counts.items())),
        },
        "note": "No Qwen inference was repeated.",
    }
    with (output_dir / "p3_controlled_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in controlled_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "p3_control_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
