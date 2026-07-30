#!/usr/bin/env python
"""Audit the HUSTbearing ten-condition continual-learning protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from se_gscl.data import (  # noqa: E402
    HUST_FIXED_SPEED_DOMAINS,
    audit_hust_protocol,
    write_hust_protocol_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--domains",
        default=",".join(str(value) for value in HUST_FIXED_SPEED_DOMAINS),
    )
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--step-size", type=int, default=1024)
    parser.add_argument("--max-windows-per-file", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "protocol_audit" / "hustbearing10",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domains = [
        int(value.strip())
        for value in args.domains.split(",")
        if value.strip()
    ]
    rows, summary = audit_hust_protocol(
        args.data_root,
        domains,
        window_size=args.window_size,
        step_size=args.step_size,
        max_windows_per_file=args.max_windows_per_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    csv_path, json_path = write_hust_protocol_audit(
        rows,
        summary,
        args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Detailed CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    return 0 if summary["protocol_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
