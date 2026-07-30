"""Lightweight HUSTbearing protocol audit without loading signal arrays."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from fdllm_repro.datasets import HUST_FILE_RE, HUST_LABELS

HUST_FIXED_SPEED_DOMAINS = (20, 25, 30, 35, 40, 60, 65, 70, 75, 80)
_TOTAL_ROWS_RE = re.compile(r"^\s*Total\s+Data\s+Rows\s+(\d+)\s*$", re.IGNORECASE)


def audit_hust_protocol(
    data_root: str | Path,
    domains: Iterable[int] = HUST_FIXED_SPEED_DOMAINS,
    *,
    window_size: int = 2048,
    step_size: int = 1024,
    max_windows_per_file: int | None = 60,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Audit the fixed-speed domain-class grid and planned guarded splits."""

    _validate_window_protocol(window_size, step_size, train_ratio, val_ratio)
    root = _resolve_hust_root(Path(data_root))
    requested_domains = tuple(int(value) for value in domains)
    expected_states = sorted(HUST_LABELS, key=lambda key: HUST_LABELS[key][0])
    expected_pairs = {
        (domain, state)
        for domain in requested_domains
        for state in expected_states
    }

    files_by_pair: dict[tuple[int, str], list[Path]] = defaultdict(list)
    variable_speed_files: list[str] = []
    unknown_files: list[str] = []
    unexpected_fixed_domains: set[int] = set()

    for path in _hust_xls_files(root):
        match = HUST_FILE_RE.match(path.name)
        if match is None:
            unknown_files.append(path.name)
            continue
        state = match.group("state").upper()
        speed_text = match.group("speed")
        if speed_text is None:
            variable_speed_files.append(path.name)
            continue
        domain = int(speed_text)
        if domain not in requested_domains:
            unexpected_fixed_domains.add(domain)
            continue
        files_by_pair[(domain, state)].append(path)

    missing_pairs = sorted(expected_pairs - set(files_by_pair))
    duplicate_pairs = {
        pair: paths
        for pair, paths in files_by_pair.items()
        if len(paths) > 1
    }
    guard_windows = max(1, int(math.ceil(window_size / step_size)))
    rows: list[dict[str, object]] = []
    header_errors: list[dict[str, str]] = []

    for domain, state in sorted(
        files_by_pair,
        key=lambda pair: (pair[0], HUST_LABELS[pair[1]][0]),
    ):
        label_id, label_name, *_ = HUST_LABELS[state]
        for path in sorted(files_by_pair[(domain, state)]):
            try:
                signal_rows = _read_total_data_rows(path)
                planned_windows = _window_count(
                    signal_rows,
                    window_size,
                    step_size,
                    max_windows_per_file,
                )
                split_counts = _guarded_split_counts(
                    planned_windows,
                    train_ratio,
                    val_ratio,
                    guard_windows,
                )
                split_error = ""
            except (OSError, ValueError) as error:
                signal_rows = 0
                planned_windows = 0
                split_counts = {"train": 0, "val": 0, "test": 0}
                split_error = str(error)
                header_errors.append({"file": path.name, "error": split_error})

            rows.append(
                {
                    "domain_id": domain,
                    "condition_name": f"{domain}Hz",
                    "state_key": state,
                    "label_id": label_id,
                    "label_name": label_name,
                    "file_name": path.name,
                    "signal_rows": signal_rows,
                    "planned_windows": planned_windows,
                    "guard_windows": guard_windows,
                    "train_windows": split_counts["train"],
                    "val_windows": split_counts["val"],
                    "test_windows": split_counts["test"],
                    "excluded_guard_windows": (
                        planned_windows - sum(split_counts.values())
                    ),
                    "error": split_error,
                }
            )

    observed_lengths = sorted(
        {int(row["signal_rows"]) for row in rows if int(row["signal_rows"]) > 0}
    )
    totals = {
        "planned_windows": sum(int(row["planned_windows"]) for row in rows),
        "train_windows": sum(int(row["train_windows"]) for row in rows),
        "val_windows": sum(int(row["val_windows"]) for row in rows),
        "test_windows": sum(int(row["test_windows"]) for row in rows),
        "excluded_guard_windows": sum(
            int(row["excluded_guard_windows"]) for row in rows
        ),
    }
    missing_details = [
        {
            "domain_id": domain,
            "state_key": state,
            "label_name": HUST_LABELS[state][1],
        }
        for domain, state in missing_pairs
    ]
    duplicate_details = [
        {
            "domain_id": domain,
            "state_key": state,
            "label_name": HUST_LABELS[state][1],
            "files": [path.name for path in paths],
        }
        for (domain, state), paths in sorted(duplicate_pairs.items())
    ]
    expected_records = len(expected_pairs)
    protocol_ready = (
        len(rows) == expected_records
        and not missing_details
        and not duplicate_details
        and not header_errors
        and len(observed_lengths) == 1
        and all(
            int(row["train_windows"]) > 0
            and int(row["val_windows"]) > 0
            and int(row["test_windows"]) > 0
            for row in rows
        )
    )
    summary: dict[str, object] = {
        "status": "ok" if protocol_ready else "error",
        "dataset": "hustbearing",
        "audit_type": "fixed-speed continual-learning protocol",
        "data_root": str(root.resolve()),
        "domain_order": list(requested_domains),
        "num_domains": len(requested_domains),
        "class_names": [HUST_LABELS[state][1] for state in expected_states],
        "num_classes": len(expected_states),
        "expected_records": expected_records,
        "observed_records": len(rows),
        "complete_domain_class_grid": (
            not missing_details and not duplicate_details
        ),
        "missing_domain_class_pairs": missing_details,
        "duplicate_domain_class_pairs": duplicate_details,
        "unexpected_fixed_domains": sorted(unexpected_fixed_domains),
        "excluded_variable_speed_files": sorted(variable_speed_files),
        "unrecognized_xls_files": sorted(unknown_files),
        "signal_lengths_from_headers": observed_lengths,
        "consistent_signal_length": len(observed_lengths) <= 1,
        "header_or_split_errors": header_errors,
        "window_protocol": {
            "window_size": window_size,
            "step_size": step_size,
            "max_windows_per_file": max_windows_per_file,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 1.0 - train_ratio - val_ratio,
            "guard_windows_per_boundary": guard_windows,
            "split_mode": "contiguous blocked split with overlap guard",
        },
        "totals": totals,
        "protocol_ready": protocol_ready,
        "notes": [
            "Only fixed-speed files are included in the main continual stream.",
            "Variable-speed files are reserved for later open-condition evaluation.",
            "Signal row counts are read from file headers; arrays are not loaded.",
        ],
    }
    return rows, summary


def write_hust_protocol_audit(
    rows: Iterable[dict[str, object]],
    summary: dict[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the detailed CSV and summary JSON for the protocol audit."""

    materialized = list(rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "hust10_protocol_files.csv"
    json_path = destination / "hust10_protocol_summary.json"

    if materialized:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
            writer.writeheader()
            writer.writerows(materialized)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


def _validate_window_protocol(
    window_size: int,
    step_size: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave a positive test split.")


def _window_count(
    signal_rows: int,
    window_size: int,
    step_size: int,
    max_windows_per_file: int | None,
) -> int:
    if signal_rows < window_size:
        return 0
    count = 1 + (signal_rows - window_size) // step_size
    if max_windows_per_file is not None:
        count = min(count, max_windows_per_file)
    return count


def _guarded_split_counts(
    num_windows: int,
    train_ratio: float,
    val_ratio: float,
    guard_windows: int,
) -> dict[str, int]:
    """Mirror the guarded blocked split used by domain_windows.py."""

    minimum = 6 + 4 * guard_windows
    if num_windows < minimum:
        raise ValueError(
            f"A record needs at least {minimum} windows for guarded blocked "
            f"splitting; got {num_windows}."
        )
    train_boundary = max(2, int(math.floor(num_windows * train_ratio)))
    val_boundary = max(
        train_boundary + 2,
        int(math.floor(num_windows * (train_ratio + val_ratio))),
    )
    val_boundary = min(num_windows - 2, val_boundary)
    train_end = max(1, train_boundary - guard_windows)
    val_start = min(num_windows, train_boundary + guard_windows)
    val_end = max(val_start, val_boundary - guard_windows)
    test_start = min(num_windows, val_boundary + guard_windows)
    return {
        "train": train_end,
        "val": val_end - val_start,
        "test": num_windows - test_start,
    }


def _read_total_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            match = _TOTAL_ROWS_RE.match(line)
            if match is not None:
                return int(match.group(1))
            if line.strip().lower() == "data" or line_number >= 128:
                break
    raise ValueError(f"Missing 'Total Data Rows' header in {path.name}.")


def _hust_xls_files(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        return []
    return sorted(
        path
        for path in data_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".xls"
    )


def _resolve_hust_root(data_root: Path) -> Path:
    if _hust_xls_files(data_root):
        return data_root
    for nested in (data_root / "raw data", data_root / "raw"):
        if _hust_xls_files(nested):
            return nested
    recursive_files = (
        sorted(
            path
            for path in data_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".xls"
        )
        if data_root.is_dir()
        else []
    )
    if recursive_files:
        counts: dict[Path, int] = defaultdict(int)
        for path in recursive_files:
            counts[path.parent] += 1
        return max(counts, key=lambda parent: (counts[parent], str(parent)))
    raise FileNotFoundError(
        f"Could not find HUSTbearing .xls files under {data_root}."
    )
