from __future__ import annotations

from pathlib import Path

import numpy as np

from fdllm_repro import datasets


def _touch_record(root: Path, name: str) -> None:
    path = (
        root
        / "extracted"
        / "subset1_6204_deep_groove_ball"
        / "BearingType_DeepGrooveBall"
        / "SamplingRate_8000"
        / "RotatingSpeed_600"
        / name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_multidomain8_loader_parses_raw_filename_protocol(monkeypatch, tmp_path: Path) -> None:
    for fault in ("H", "IR", "B", "OR"):
        _touch_record(tmp_path, f"H_{fault}_8_6204_600.mat")

    monkeypatch.setattr(
        datasets,
        "_load_multidomain_signal",
        lambda path: np.zeros(1024, dtype=np.float32),
    )
    records = datasets.load_records("multidomain8", tmp_path, domains=[0])

    assert len(records) == 4
    assert {record.label_name for record in records} == {
        "Normal",
        "InnerRace",
        "Ball",
        "OuterRace",
    }
    assert {record.domain_id for record in records} == {0}
    assert {record.sampling_rate for record in records} == {8000.0}
    assert {record.condition_name for record in records} == {"6204|env_A|slow|600rpm"}


def test_atomic_protocol_assigns_each_source_to_one_domain(monkeypatch, tmp_path: Path) -> None:
    for environment in ("H", "M1", "M2", "M3", "U1", "U2", "U3", "L"):
        for fault in ("H", "IR", "B", "OR"):
            _touch_record(tmp_path, f"{environment}_{fault}_8_6204_600.mat")

    monkeypatch.setattr(
        datasets,
        "_load_multidomain_signal",
        lambda path: np.zeros(1024, dtype=np.float32),
    )
    records = datasets.load_records(
        "multidomain8_atomic",
        tmp_path,
        domains=list(range(0, 16, 2)),
    )

    assert len(records) == 32
    assert {record.domain_id for record in records} == set(range(0, 16, 2))
    memberships: dict[str, set[int]] = {}
    for record in records:
        memberships.setdefault(record.source_record_id, set()).add(record.domain_id)
    assert all(len(domains) == 1 for domains in memberships.values())
