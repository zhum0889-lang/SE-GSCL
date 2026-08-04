"""Inspect extracted MultiDomainBearing MAT files before protocol construction.

The Mendeley subsets are collected independently and can use different MATLAB
layouts.  This read-only utility records file-name patterns, root variables,
and candidate numeric signal arrays so the experiment loader is built from the
actual archive schema rather than from filename assumptions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.io import loadmat, whosmat
except ModuleNotFoundError as exc:  # pragma: no cover - cloud dependency check.
    raise SystemExit("scipy is required; install requirements.txt first.") from exc


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="MultiDomainBearing root containing extracted/.",
    )
    parser.add_argument(
        "--sample-files",
        type=int,
        default=12,
        help="Number of representative MAT files to load deeply.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "data_audit" / "multidomain",
    )
    return parser.parse_args()


def mat_paths(data_root: Path) -> list[Path]:
    extracted = data_root / "extracted"
    root = extracted if extracted.is_dir() else data_root
    return sorted(path for path in root.rglob("*.mat") if path.is_file())


def subset_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except ValueError:
        return "unknown"


def filename_signature(path: Path) -> str:
    """Keep a compact comparable shape while preserving semantic tokens."""

    stem = path.stem.lower()
    stem = re.sub(r"\d+", "#", stem)
    stem = re.sub(r"[_\-]+", "_", stem)
    return stem


def describe_value(value: Any, depth: int = 0) -> dict[str, Any]:
    if depth >= 3:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, np.ndarray):
        result: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "size": int(value.size),
        }
        if value.dtype.names:
            result["fields"] = list(value.dtype.names)
        if np.issubdtype(value.dtype, np.number) and value.size:
            result["numeric"] = True
        return result
    if hasattr(value, "_fieldnames"):
        fields = list(getattr(value, "_fieldnames", []) or [])
        return {
            "type": type(value).__name__,
            "fields": {
                field: describe_value(getattr(value, field), depth + 1)
                for field in fields[:24]
            },
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "fields": {
                str(key): describe_value(item, depth + 1)
                for key, item in list(value.items())[:24]
            },
        }
    return {"type": type(value).__name__}


def numeric_candidates(value: Any, prefix: str = "", depth: int = 0) -> list[dict[str, Any]]:
    if depth >= 5:
        return []
    candidates: list[dict[str, Any]] = []
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for field in value.dtype.names:
                candidates.extend(
                    numeric_candidates(value[field], f"{prefix}.{field}".strip("."), depth + 1)
                )
        elif np.issubdtype(value.dtype, np.number) and value.size >= 1024:
            candidates.append(
                {
                    "path": prefix or "<root>",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "size": int(value.size),
                }
            )
        return candidates
    if hasattr(value, "_fieldnames"):
        for field in getattr(value, "_fieldnames", []) or []:
            candidates.extend(
                numeric_candidates(getattr(value, field), f"{prefix}.{field}".strip("."), depth + 1)
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            candidates.extend(numeric_candidates(item, f"{prefix}.{key}".strip("."), depth + 1))
    return candidates


def inspect_file(path: Path, root: Path, deep: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "file": path.relative_to(root).as_posix(),
        "subset": subset_name(path, root),
        "bytes": path.stat().st_size,
        "filename_signature": filename_signature(path),
    }
    try:
        record["variables"] = [
            {"name": name, "shape": list(shape), "dtype": dtype}
            for name, shape, dtype in whosmat(path)
        ]
    except Exception as exc:  # retain failures for corruption/v7.3 diagnostics.
        record["header_error"] = f"{type(exc).__name__}: {exc}"
        return record
    if not deep:
        return record
    try:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        values = {key: value for key, value in payload.items() if not key.startswith("__")}
        record["structure"] = {key: describe_value(value) for key, value in values.items()}
        candidates: list[dict[str, Any]] = []
        for key, value in values.items():
            candidates.extend(numeric_candidates(value, key))
        record["signal_candidates"] = candidates[:32]
    except Exception as exc:
        record["deep_error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    paths = mat_paths(data_root)
    if not paths:
        raise FileNotFoundError(
            f"No MAT files found under {data_root}; expected extracted/*.mat."
        )

    root = data_root / "extracted" if (data_root / "extracted").is_dir() else data_root
    sample_count = max(1, min(args.sample_files, len(paths)))
    # Evenly spread deep inspection across the sorted archive rather than only
    # inspecting the first subset.
    positions = np.linspace(0, len(paths) - 1, sample_count, dtype=int)
    deep_indices = set(int(position) for position in positions)
    records = [inspect_file(path, root, index in deep_indices) for index, path in enumerate(paths)]

    by_subset = Counter(record["subset"] for record in records)
    by_signature = Counter(record["filename_signature"] for record in records)
    failed_headers = [record["file"] for record in records if "header_error" in record]
    deep_records = [record for record in records if "structure" in record or "deep_error" in record]
    summary = {
        "data_root": str(data_root),
        "scan_root": str(root),
        "mat_files": len(records),
        "subsets": dict(sorted(by_subset.items())),
        "filename_signatures": dict(by_signature.most_common(40)),
        "header_failures": failed_headers,
        "deep_inspections": deep_records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "multidomain_schema_audit.json"
    csv_path = args.output_dir / "multidomain_file_inventory.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("file", "subset", "bytes", "filename_signature", "variables", "error"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "file": record["file"],
                    "subset": record["subset"],
                    "bytes": record["bytes"],
                    "filename_signature": record["filename_signature"],
                    "variables": json.dumps(record.get("variables", []), ensure_ascii=False),
                    "error": record.get("header_error", record.get("deep_error", "")),
                }
            )
    print(json.dumps({key: summary[key] for key in ("mat_files", "subsets", "header_failures")}, indent=2))
    print(f"Schema audit: {json_path}")
    print(f"File inventory: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
