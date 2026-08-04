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
        "Healthy",
        "InnerRace",
        "Ball",
        "OuterRace",
    }
    assert {record.domain_id for record in records} == {0}
    assert {record.sampling_rate for record in records} == {8000.0}
    assert {record.condition_name for record in records} == {"6204|env_A|slow|600rpm"}
