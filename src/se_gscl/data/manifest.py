"""Build auditable dataset manifests before window segmentation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from fdllm_repro.datasets import RawRecord

from .schema import ManifestRow

EXPECTED_LABELS = {
    "paderborn": {"Healthy", "InnerRace", "OuterRace"},
    "hustbearing": {
        "Healthy",
        "InnerRace_Medium",
        "InnerRace_Severe",
        "OuterRace_Medium",
        "OuterRace_Severe",
        "Ball_Medium",
        "Ball_Severe",
        "Compound_Medium",
        "Compound_Severe",
    },
    "multidomain8": {"Healthy", "InnerRace", "Ball", "OuterRace"},
    "multidomain16": {"Healthy", "InnerRace", "Ball", "OuterRace"},
}


def build_manifest_rows(records: Iterable[RawRecord]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for record in records:
        signal = np.asarray(record.signal)
        channels = 1 if signal.ndim == 1 else int(signal.shape[0])
        length = int(signal.shape[-1])
        source_id = record.source_record_id or record.file_id
        rows.append(
            ManifestRow(
                dataset=record.dataset_name or "unknown",
                source_record_id=source_id,
                bearing_id=record.bearing_id or source_id,
                sensor_id=record.sensor_id or "+".join(record.sampling_channels),
                sampling_rate=record.sampling_rate,
                signal_channels=channels,
                signal_length=length,
                fault_label=int(record.label),
                fault_name=record.label_name,
                fault_position=record.fault_position,
                severity=record.severity,
                domain_id=int(record.domain_id),
                condition_name=record.condition_name,
                speed_rpm=record.speed_rpm,
                torque_nm=record.torque_nm,
                radial_force_n=record.radial_force_n,
                source_split=record.split,
            )
        )
    if not rows:
        raise ValueError("Cannot build a manifest from zero records.")
    return rows


def manifest_summary(rows: Iterable[ManifestRow]) -> dict[str, object]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot summarize an empty manifest.")
    labels = Counter(row.fault_name for row in materialized)
    domains = Counter(str(row.domain_id) for row in materialized)
    bearings = sorted({row.bearing_id for row in materialized})
    bearings_per_label = {
        label: sorted({row.bearing_id for row in materialized if row.fault_name == label})
        for label in sorted(labels)
    }
    datasets = sorted({row.dataset for row in materialized})
    expected = set()
    for dataset in datasets:
        expected.update(EXPECTED_LABELS.get(dataset, set()))
    observed = set(labels)
    canonical = json.dumps(
        [row.to_dict() for row in materialized],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "datasets": datasets,
        "records": len(materialized),
        "bearing_ids": bearings,
        "bearing_count": len(bearings),
        "labels": dict(sorted(labels.items())),
        "missing_expected_labels": sorted(expected - observed),
        "expected_label_space_complete": not expected or expected.issubset(observed),
        "bearings_per_label": bearings_per_label,
        "bearing_group_split_ready": all(
            len(label_bearings) >= 3 for label_bearings in bearings_per_label.values()
        ),
        "domains": dict(sorted(domains.items(), key=lambda item: int(item[0]))),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def write_manifest_bundle(
    rows: Iterable[ManifestRow],
    output_dir: str | Path,
    stem: str = "record_manifest",
) -> tuple[Path, Path]:
    materialized = list(rows)
    summary = manifest_summary(materialized)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"{stem}.csv"
    json_path = destination / f"{stem}_summary.json"

    fieldnames = list(materialized[0].to_dict())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in materialized)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path
