"""Create a record-level audit manifest before window segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fdllm_repro.datasets import load_hustbearing, load_records  # noqa: E402
from se_gscl.data import build_manifest_rows, manifest_summary, write_manifest_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=(
            "cwru4",
            "cwru10",
            "cwru19",
            "paderborn",
            "hustbearing",
            "multidomain8",
            "multidomain16",
        ),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--domains", help="Comma-separated domain IDs; omit to scan all.")
    parser.add_argument(
        "--channels",
        default="X,Y,Z",
        help="HUST channels only; comma-separated subset of X,Y,Z.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "data_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domains = None
    if args.domains:
        domains = [int(value.strip()) for value in args.domains.split(",") if value.strip()]
    if args.dataset == "hustbearing":
        channels = tuple(value.strip().upper() for value in args.channels.split(",") if value.strip())
        records = load_hustbearing(args.data_root, domains=domains, channels=channels)
    else:
        records = load_records(args.dataset, args.data_root, domains=domains)

    rows = build_manifest_rows(records)
    csv_path, json_path = write_manifest_bundle(rows, args.output_dir)
    print(json.dumps(manifest_summary(rows), ensure_ascii=False, indent=2))
    print(f"Manifest CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
