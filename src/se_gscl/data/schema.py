"""Serializable record-level schema shared by all datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ManifestRow:
    dataset: str
    source_record_id: str
    bearing_id: str
    sensor_id: str
    sampling_rate: float | None
    signal_channels: int
    signal_length: int
    fault_label: int
    fault_name: str
    fault_position: str
    severity: str
    domain_id: int
    condition_name: str
    speed_rpm: float | None
    torque_nm: float | None
    radial_force_n: float | None
    source_split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
