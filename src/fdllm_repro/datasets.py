"""Dataset loaders for the FD-LLM bearing reproduction.

The paper uses z-score normalization and sliding windows. This module keeps
that preprocessing explicit so the experimental split is easy to audit.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    from scipy.io import loadmat
except ModuleNotFoundError:  # Protocol-only tests do not need MAT loading.
    def loadmat(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(
            "scipy is required to load .mat bearing records. "
            "Install implementation/requirements.txt before running real experiments."
        )


@dataclass(frozen=True)
class RawRecord:
    signal: np.ndarray
    label: int
    label_name: str
    file_id: str
    split: str
    sampling_rate: Optional[float] = None
    load: Optional[float] = None
    bearing_position: str = ""
    fault_position: str = ""
    fault_size: Optional[float] = None
    fault_size_unit: str = ""
    domain_id: int = -1
    condition_name: str = ""
    severity: str = ""
    sampling_channels: tuple[str, ...] = ()
    dataset_name: str = ""
    source_record_id: str = ""
    bearing_id: str = ""
    sensor_id: str = ""
    speed_rpm: Optional[float] = None
    torque_nm: Optional[float] = None
    radial_force_n: Optional[float] = None


@dataclass(frozen=True)
class WindowedDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    train_records: list[RawRecord]
    test_records: list[RawRecord]


LOCAL_LABELS = {
    "baseline": (0, "Normal"),
    "innerracefault": (1, "InnerRaceFault"),
    "outerracefault": (2, "OuterRaceFault"),
}


CWRU10_INFO = {
    "97.mat": (0, "Normal", 0, None, None),
    "98.mat": (0, "Normal", 1, None, None),
    "99.mat": (0, "Normal", 2, None, None),
    "100.mat": (0, "Normal", 3, None, None),
    "105.mat": (1, "InnerRace_007", 0, "inner race", 0.007),
    "106.mat": (1, "InnerRace_007", 1, "inner race", 0.007),
    "107.mat": (1, "InnerRace_007", 2, "inner race", 0.007),
    "108.mat": (1, "InnerRace_007", 3, "inner race", 0.007),
    "169.mat": (2, "InnerRace_014", 0, "inner race", 0.014),
    "170.mat": (2, "InnerRace_014", 1, "inner race", 0.014),
    "171.mat": (2, "InnerRace_014", 2, "inner race", 0.014),
    "172.mat": (2, "InnerRace_014", 3, "inner race", 0.014),
    "209.mat": (3, "InnerRace_021", 0, "inner race", 0.021),
    "210.mat": (3, "InnerRace_021", 1, "inner race", 0.021),
    "211.mat": (3, "InnerRace_021", 2, "inner race", 0.021),
    "212.mat": (3, "InnerRace_021", 3, "inner race", 0.021),
    "118.mat": (4, "Ball_007", 0, "ball", 0.007),
    "119.mat": (4, "Ball_007", 1, "ball", 0.007),
    "120.mat": (4, "Ball_007", 2, "ball", 0.007),
    "121.mat": (4, "Ball_007", 3, "ball", 0.007),
    "185.mat": (5, "Ball_014", 0, "ball", 0.014),
    "186.mat": (5, "Ball_014", 1, "ball", 0.014),
    "187.mat": (5, "Ball_014", 2, "ball", 0.014),
    "188.mat": (5, "Ball_014", 3, "ball", 0.014),
    "222.mat": (6, "Ball_021", 0, "ball", 0.021),
    "223.mat": (6, "Ball_021", 1, "ball", 0.021),
    "224.mat": (6, "Ball_021", 2, "ball", 0.021),
    "225.mat": (6, "Ball_021", 3, "ball", 0.021),
    "130.mat": (7, "OuterRace_007", 0, "outer race", 0.007),
    "131.mat": (7, "OuterRace_007", 1, "outer race", 0.007),
    "132.mat": (7, "OuterRace_007", 2, "outer race", 0.007),
    "133.mat": (7, "OuterRace_007", 3, "outer race", 0.007),
    "197.mat": (8, "OuterRace_014", 0, "outer race", 0.014),
    "198.mat": (8, "OuterRace_014", 1, "outer race", 0.014),
    "199.mat": (8, "OuterRace_014", 2, "outer race", 0.014),
    "200.mat": (8, "OuterRace_014", 3, "outer race", 0.014),
    "234.mat": (9, "OuterRace_021", 0, "outer race", 0.021),
    "235.mat": (9, "OuterRace_021", 1, "outer race", 0.021),
    "236.mat": (9, "OuterRace_021", 2, "outer race", 0.021),
    "237.mat": (9, "OuterRace_021", 3, "outer race", 0.021),
}


def load_records(
    dataset: str,
    data_root: Path,
    metadata_csv: Optional[Path] = None,
    domains: Optional[Iterable[int]] = None,
) -> list[RawRecord]:
    if dataset == "local_bearings":
        return load_local_bearings(data_root)
    if dataset == "cwru4":
        return load_cwru4(data_root)
    if dataset == "cwru10":
        return load_cwru10(data_root)
    if dataset == "cwru19":
        return load_cwru19(data_root)
    if dataset == "hustbearing":
        return load_hustbearing(data_root, domains=domains)
    if dataset == "paderborn":
        return load_paderborn(data_root, domains=domains)
    if dataset == "multidomain8":
        return load_multidomain_bearing(
            data_root,
            domains=domains,
            sampling_rate=8000,
        )
    if dataset == "multidomain16":
        return load_multidomain_bearing(
            data_root,
            domains=domains,
            sampling_rate=16000,
        )
    if dataset == "multidomain8_atomic":
        return load_multidomain_bearing(
            data_root,
            domains=domains,
            sampling_rate=8000,
            protocol="atomic",
        )
    if dataset == "multidomain16_atomic":
        return load_multidomain_bearing(
            data_root,
            domains=domains,
            sampling_rate=16000,
            protocol="atomic",
        )
    if dataset == "metadata":
        if metadata_csv is None:
            raise ValueError("--metadata-csv is required when --dataset metadata")
        return load_from_metadata(data_root, metadata_csv)
    raise ValueError(f"Unknown dataset mode: {dataset}")


HUST_LABELS = {
    "H": (0, "Healthy", "normal", "none", None),
    "0.5X_I": (1, "InnerRace_Medium", "inner race", "medium", 0.15),
    "I": (2, "InnerRace_Severe", "inner race", "severe", 0.30),
    "0.5X_O": (3, "OuterRace_Medium", "outer race", "medium", 0.15),
    "O": (4, "OuterRace_Severe", "outer race", "severe", 0.30),
    "0.5X_B": (5, "Ball_Medium", "ball", "medium", 0.25),
    "B": (6, "Ball_Severe", "ball", "severe", 0.50),
    "0.5X_C": (7, "Compound_Medium", "inner and outer race", "medium", 0.15),
    "C": (8, "Compound_Severe", "inner and outer race", "severe", 0.30),
}

HUST_FILE_RE = re.compile(
    r"^(?P<state>(?:0\.5X_)?[BC HIO])_"
    r"(?:(?P<speed>\d+)[Hh][Zz]|VS_0_40_0[Hh][Zz])\.xls$".replace(" ", ""),
    re.IGNORECASE,
)

PADERBORN_FILE_RE = re.compile(
    r"^N(?P<speed>\d{2})_M(?P<torque>\d{2})_F(?P<force>\d{2})_"
    r"(?P<bearing>(?:K\d{3}|K[A-Z]\d{2}))_(?P<run>\d+)\.mat$",
    re.IGNORECASE,
)

PADERBORN_DOMAINS = {
    (1500, 0.7, 1000): 0,
    (900, 0.7, 1000): 1,
    (1500, 0.1, 1000): 2,
    (1500, 0.7, 400): 3,
}


MULTIDOMAIN_FILE_RE = re.compile(
    r"^(?P<environment>H|M[123]|U[123]|L)_"
    r"(?P<fault>H|B|IR|OR)_"
    r"(?P<sampling_khz>8|16)_"
    r"(?P<bearing>[A-Z0-9]+)_"
    r"(?P<speed>600|800|1000|1200|1400|1600)\.mat$",
    re.IGNORECASE,
)

MULTIDOMAIN_LABELS = {
    "H": (0, "Normal", "normal"),
    "IR": (1, "InnerRace", "inner race"),
    "B": (2, "Ball", "ball"),
    "OR": (3, "OuterRace", "outer race"),
}

# These condition sets reproduce the compound-domain construction reported by
# Risca et al. A raw baseline/environment recording can occur in more than one
# protocol domain; its source_record_id remains unchanged for auditability.
MULTIDOMAIN_ENVIRONMENT_GROUPS = (
    ("A", frozenset({"H", "M1", "U1", "L"})),
    ("B", frozenset({"H", "U1", "U2", "U3"})),
    ("C", frozenset({"H", "M1", "M2", "M3"})),
)
MULTIDOMAIN_SPEED_GROUPS = (
    ("slow", frozenset({600, 800, 1000})),
    ("fast", frozenset({1200, 1400, 1600})),
)
MULTIDOMAIN_ATOMIC_ENVIRONMENTS = ("H", "M1", "M2", "M3", "U1", "U2", "U3", "L")
MULTIDOMAIN_BEARING_GROUPS = (
    ("6204", "subset1_6204_deep_groove_ball"),
    ("N204_NJ204", "subset2_N204_NJ204_cylindrical_roller"),
    ("30204", "subset3_30204_tapered_roller"),
)


def load_hustbearing(
    data_root: Path,
    domains: Optional[Iterable[int]] = None,
    channels: tuple[str, ...] = ("X", "Y", "Z"),
) -> list[RawRecord]:
    """Load the Zhao-Zio-Shen HUSTbearing text-export files.

    The files use an ``.xls`` suffix but contain tab-separated text. The five
    numeric columns are time, tachometer, X, Y, and Z. The official dataset
    notes that the speed column is not the authoritative operating-condition
    value, so the domain is parsed from the filename.
    """

    wanted_domains = None if domains is None else {int(value) for value in domains}
    root = _resolve_hust_data_root(data_root)
    records: list[RawRecord] = []
    for path in _hust_xls_files(root):
        match = HUST_FILE_RE.match(path.name)
        if match is None:
            continue
        state_key = match.group("state").upper()
        speed_text = match.group("speed")
        domain_id = int(speed_text) if speed_text is not None else 0
        if wanted_domains is not None and domain_id not in wanted_domains:
            continue
        if state_key not in HUST_LABELS:
            raise ValueError(f"Unsupported HUSTbearing state in {path.name}: {state_key}")

        label, label_name, fault_position, severity, fault_size_mm = HUST_LABELS[state_key]
        condition_name = f"{domain_id}Hz" if speed_text is not None else "VS_0_40_0Hz"
        signal = _load_hust_text_signal(path, channels)
        records.append(
            RawRecord(
                signal=signal,
                label=label,
                label_name=label_name,
                file_id=path.name,
                split="unsplit",
                sampling_rate=25600.0,
                load=float(domain_id),
                bearing_position="test bearing",
                fault_position=fault_position,
                fault_size=fault_size_mm,
                fault_size_unit="mm",
                domain_id=domain_id,
                condition_name=condition_name,
                severity=severity,
                sampling_channels=channels,
                dataset_name="hustbearing",
                source_record_id=path.name,
                bearing_id="HUST_test_bearing",
                sensor_id="+".join(channels),
                speed_rpm=float(domain_id * 60) if speed_text is not None else None,
            )
        )
    if not records:
        domain_hint = "" if wanted_domains is None else f" for domains {sorted(wanted_domains)}"
        raise FileNotFoundError(f"No HUSTbearing .xls records found under {data_root}{domain_hint}")
    return records


def load_local_bearings(data_root: Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    for path in sorted(data_root.rglob("*.mat")):
        lower = path.name.lower()
        label_pair = None
        for token, pair in LOCAL_LABELS.items():
            if token in lower:
                label_pair = pair
                break
        if label_pair is None:
            continue

        signal = _load_local_bearing_signal(path)
        split = "test" if any(part.lower() == "test" for part in path.parts) else "train"
        label, label_name = label_pair
        records.append(
            RawRecord(
                signal=signal,
                label=label,
                label_name=label_name,
                file_id=path.name,
                split=split,
                sampling_rate=_load_local_sampling_rate(path),
                fault_position=_local_fault_position(label_name),
            )
        )
    if not records:
        raise FileNotFoundError(f"No local bearing .mat records found under {data_root}")
    return records


def load_cwru10(data_root: Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    for path in sorted(data_root.glob("*.mat")):
        info = CWRU10_INFO.get(path.name.lower())
        if info is None:
            continue
        label, label_name, load, fault_position, fault_size = info
        signal = _load_cwru_signal(path)
        split = "test" if load == 3 else "train"
        records.append(
            RawRecord(
                signal=signal,
                label=label,
                label_name=label_name,
                file_id=path.name,
                split=split,
                sampling_rate=12000,
                load=float(load),
                bearing_position="drive end",
                fault_position=fault_position or "",
                fault_size=fault_size,
                fault_size_unit="inch",
                domain_id=int(load),
                condition_name=f"{load}HP",
                dataset_name="cwru",
                source_record_id=path.name,
                bearing_id=f"CWRU_{label_name}",
                sensor_id="drive_end",
                speed_rpm=_load_cwru_rpm(path, load),
            )
        )
    if not records:
        raise FileNotFoundError(
            f"No recognized CWRU .mat files found under {data_root}. "
            "Expected files such as 97.mat, 105.mat, 118.mat, 130.mat."
        )
    return records


def load_cwru4(data_root: Path) -> list[RawRecord]:
    """Collapse CWRU fault severities into four fixed fault identities."""

    coarse_names = {
        "normal": (0, "Normal"),
        "inner race": (1, "InnerRace"),
        "ball": (2, "Ball"),
        "outer race": (3, "OuterRace"),
    }
    records: list[RawRecord] = []
    for row in load_cwru10(data_root):
        key = "normal" if row.label == 0 else row.fault_position.lower()
        if key not in coarse_names:
            raise ValueError(f"Unsupported CWRU coarse label: {row.fault_position!r}")
        label, label_name = coarse_names[key]
        records.append(
            replace(
                row,
                label=label,
                label_name=label_name,
                bearing_id=f"CWRU_{label_name}",
            )
        )
    return records


def load_cwru19(data_root: Path) -> list[RawRecord]:
    """Load a paper-style 19-class CWRU setup.

    The downloaded CWRU files contain both DE_time and FE_time channels. We use
    FE_time for the nine fan-end classes, DE_time for the nine drive-end
    classes, plus one normal class from DE_time.
    """

    records: list[RawRecord] = []
    for path in sorted(data_root.glob("*.mat")):
        info = CWRU10_INFO.get(path.name.lower())
        if info is None:
            continue
        label, label_name, load, fault_position, fault_size = info
        split = "test" if load == 3 else "train"
        if label == 0:
            records.append(
                RawRecord(
                    signal=_load_cwru_channel(path, "DE"),
                    label=0,
                    label_name="Normal",
                    file_id=f"{path.name}:DE",
                    split=split,
                    sampling_rate=12000,
                    load=float(load),
                    bearing_position="drive end",
                    fault_position="normal",
                    fault_size=None,
                    domain_id=int(load),
                    condition_name=f"{load}HP",
                    dataset_name="cwru",
                    source_record_id=path.name,
                    bearing_id="CWRU_Normal",
                    sensor_id="drive_end",
                )
            )
            continue

        fan_label, drive_label, base_name = _cwru19_labels(fault_position or "", fault_size)
        records.append(
            RawRecord(
                signal=_load_cwru_channel(path, "FE"),
                label=fan_label,
                label_name=f"FanEnd_{base_name}",
                file_id=f"{path.name}:FE",
                split=split,
                sampling_rate=12000,
                load=float(load),
                bearing_position="fan end",
                fault_position=fault_position or "",
                fault_size=fault_size,
                fault_size_unit="inch",
                domain_id=int(load),
                condition_name=f"{load}HP",
                dataset_name="cwru",
                source_record_id=path.name,
                bearing_id=f"CWRU_FE_{base_name}",
                sensor_id="fan_end",
            )
        )
        records.append(
            RawRecord(
                signal=_load_cwru_channel(path, "DE"),
                label=drive_label,
                label_name=f"DriveEnd_{base_name}",
                file_id=f"{path.name}:DE",
                split=split,
                sampling_rate=12000,
                load=float(load),
                bearing_position="drive end",
                fault_position=fault_position or "",
                fault_size=fault_size,
                fault_size_unit="inch",
                domain_id=int(load),
                condition_name=f"{load}HP",
                dataset_name="cwru",
                source_record_id=path.name,
                bearing_id=f"CWRU_DE_{base_name}",
                sensor_id="drive_end",
            )
        )
    if not records:
        raise FileNotFoundError(
            f"No recognized CWRU .mat files found under {data_root}. "
            "Expected files such as 97.mat, 105.mat, 118.mat, 130.mat."
        )
    return records


def load_paderborn(
    data_root: Path,
    domains: Optional[Iterable[int]] = None,
    channel: str = "vibration_1",
) -> list[RawRecord]:
    """Load Paderborn bearing records with bearing-level metadata.

    The official file name encodes speed, torque, radial force, bearing ID,
    and repetition. The vibration channel is stored inside the root MATLAB
    struct's ``Y`` field.
    """

    wanted_domains = None if domains is None else {int(value) for value in domains}
    records: list[RawRecord] = []
    seen_paths: set[str] = set()
    mat_paths = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mat"
    )
    unmatched_names: list[str] = []
    unsupported_conditions: set[tuple[int, float, int]] = set()
    for path in mat_paths:
        match = PADERBORN_FILE_RE.match(path.name)
        if match is None:
            if len(unmatched_names) < 8:
                unmatched_names.append(path.name)
            continue
        canonical_path = str(path.resolve()).lower()
        if canonical_path in seen_paths:
            continue
        seen_paths.add(canonical_path)

        speed_rpm = int(match.group("speed")) * 100
        torque_nm = int(match.group("torque")) / 10.0
        radial_force_n = int(match.group("force")) * 100
        condition_key = (speed_rpm, torque_nm, radial_force_n)
        if condition_key not in PADERBORN_DOMAINS:
            unsupported_conditions.add(condition_key)
            continue
        domain_id = PADERBORN_DOMAINS[condition_key]
        if wanted_domains is not None and domain_id not in wanted_domains:
            continue

        bearing_id = match.group("bearing").upper()
        label, label_name, fault_position = _paderborn_label(bearing_id)
        relative_id = path.relative_to(data_root).as_posix()
        records.append(
            RawRecord(
                signal=_load_paderborn_channel(path, channel),
                label=label,
                label_name=label_name,
                file_id=relative_id,
                split="unsplit",
                sampling_rate=64000.0,
                load=float(domain_id),
                bearing_position="test bearing",
                fault_position=fault_position,
                domain_id=domain_id,
                condition_name=(
                    f"{speed_rpm}rpm_{torque_nm:g}Nm_{radial_force_n}N"
                ),
                sampling_channels=(channel,),
                dataset_name="paderborn",
                source_record_id=relative_id,
                bearing_id=bearing_id,
                sensor_id=channel,
                speed_rpm=float(speed_rpm),
                torque_nm=float(torque_nm),
                radial_force_n=float(radial_force_n),
            )
        )
    if not records:
        domain_hint = "" if wanted_domains is None else f" for domains {sorted(wanted_domains)}"
        if not mat_paths:
            archive_count = sum(
                1
                for path in data_root.parent.joinpath("archives").glob("*.rar")
                if path.is_file()
            )
            archive_hint = (
                f" Found {archive_count} RAR archive(s) next to this directory; "
                "extract them before preparing the dataset."
                if archive_count
                else " No MAT files or adjacent RAR archives were detected."
            )
            raise FileNotFoundError(
                f"No Paderborn MAT files found under {data_root}.{archive_hint}"
            )
        detail_parts = [f"scanned_mat_files={len(mat_paths)}"]
        if unmatched_names:
            detail_parts.append(f"unmatched_filename_examples={unmatched_names}")
        if unsupported_conditions:
            detail_parts.append(
                f"unsupported_condition_tuples={sorted(unsupported_conditions)}"
            )
        raise FileNotFoundError(
            f"No supported Paderborn MAT records found under {data_root}{domain_hint}; "
            + "; ".join(detail_parts)
        )
    return records


def load_multidomain_bearing(
    data_root: Path,
    domains: Optional[Iterable[int]] = None,
    sampling_rate: int = 8000,
    protocol: str = "overlap18",
) -> list[RawRecord]:
    """Load either the published 18-domain or source-disjoint 48-domain protocol.

    Each MATLAB file stores a raw one-dimensional ``Data`` vector. File names
    encode environment, fault identity, sampling rate, bearing identifier and
    speed. The protocol domain is the Cartesian product of three bearing
    groups, three compound-environment groups and two speed groups. Files in
    overlapping environment groups are deliberately represented in each
    applicable protocol domain, matching the referenced continual-learning
    setup while retaining a shared source_record_id for leakage audits. The
    ``atomic`` protocol instead treats every measured environment as a distinct
    condition, so each raw recording belongs to exactly one of 48 domains.
    """

    if sampling_rate not in {8000, 16000}:
        raise ValueError("sampling_rate must be either 8000 or 16000.")
    if protocol not in {"overlap18", "atomic"}:
        raise ValueError("protocol must be either 'overlap18' or 'atomic'.")
    wanted_domains = None if domains is None else {int(value) for value in domains}
    root = _resolve_multidomain_data_root(data_root)
    records: list[RawRecord] = []
    unmatched: list[str] = []

    subset_to_group = {
        subset: (group_index, group_name)
        for group_index, (group_name, subset) in enumerate(MULTIDOMAIN_BEARING_GROUPS)
    }
    for path in sorted(root.rglob("*.mat")):
        relative = path.relative_to(root).as_posix()
        subset = path.relative_to(root).parts[0]
        bearing_group = subset_to_group.get(subset)
        if bearing_group is None:
            continue
        match = MULTIDOMAIN_FILE_RE.match(path.name)
        if match is None:
            if len(unmatched) < 12:
                unmatched.append(relative)
            continue

        environment = match.group("environment").upper()
        fault_code = match.group("fault").upper()
        rate_hz = int(match.group("sampling_khz")) * 1000
        speed_rpm = int(match.group("speed"))
        if rate_hz != sampling_rate:
            continue
        if fault_code not in MULTIDOMAIN_LABELS:
            raise ValueError(f"Unsupported fault code {fault_code!r} in {relative}")

        label, label_name, fault_position = MULTIDOMAIN_LABELS[fault_code]
        bearing_index, protocol_bearing = bearing_group
        if protocol == "atomic":
            environment_matches = [
                (MULTIDOMAIN_ATOMIC_ENVIRONMENTS.index(environment), environment)
            ]
        else:
            environment_matches = [
                (index, name)
                for index, (name, members) in enumerate(MULTIDOMAIN_ENVIRONMENT_GROUPS)
                if environment in members
            ]
        speed_match = next(
            (
                (index, name)
                for index, (name, members) in enumerate(MULTIDOMAIN_SPEED_GROUPS)
                if speed_rpm in members
            ),
            None,
        )
        if speed_match is None:
            raise ValueError(f"Unsupported speed {speed_rpm} in {relative}")
        speed_index, speed_group = speed_match

        for environment_index, environment_group in environment_matches:
            domains_per_bearing = 16 if protocol == "atomic" else 6
            domain_id = (
                bearing_index * domains_per_bearing
                + environment_index * 2
                + speed_index
            )
            if wanted_domains is not None and domain_id not in wanted_domains:
                continue
            records.append(
                RawRecord(
                    signal=_load_multidomain_signal(path),
                    label=label,
                    label_name=label_name,
                    file_id=relative,
                    split="unsplit",
                    sampling_rate=float(rate_hz),
                    load=float(domain_id),
                    bearing_position="test bearing",
                    fault_position=fault_position,
                    domain_id=domain_id,
                    condition_name=(
                        f"{protocol_bearing}|env_{environment_group}|"
                        f"{speed_group}|{speed_rpm}rpm"
                    ),
                    sampling_channels=("Data",),
                    dataset_name=(
                        f"multidomain{rate_hz // 1000}_atomic"
                        if protocol == "atomic"
                        else f"multidomain{rate_hz // 1000}"
                    ),
                    source_record_id=relative,
                    bearing_id=protocol_bearing,
                    sensor_id="Data",
                    speed_rpm=float(speed_rpm),
                )
            )

    if not records:
        suffix_hint = (
            "Expected extracted/subset*/.../"
            "<environment>_<fault>_<8|16>_<bearing>_<speed>.mat"
        )
        raise FileNotFoundError(
            f"No supported MultiDomainBearing records found under {root} "
            f"for sampling_rate={sampling_rate}. {suffix_hint}"
        )
    if unmatched:
        preview = "; ".join(unmatched)
        raise ValueError(
            "Found MultiDomainBearing MAT files with unsupported names. "
            f"Examples: {preview}"
        )
    _assert_multidomain_coverage(
        records,
        wanted_domains,
        domain_count=(48 if protocol == "atomic" else 18),
    )
    return records


def load_from_metadata(data_root: Path, metadata_csv: Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = data_root / row["file"]
            signal = _load_mat_signal(file_path, row.get("signal_key") or None)
            records.append(
                RawRecord(
                    signal=signal,
                    label=int(row["label"]),
                    label_name=row["label_name"],
                    file_id=row["file"],
                    split=row.get("split", "train").strip().lower() or "train",
                    sampling_rate=_maybe_float(row.get("sampling_rate")),
                    load=_maybe_float(row.get("load")),
                    bearing_position=row.get("bearing_position", ""),
                    fault_position=row.get("fault_position", ""),
                    fault_size=_maybe_float(row.get("fault_size")),
                )
            )
    if not records:
        raise FileNotFoundError(f"No rows found in metadata CSV: {metadata_csv}")
    return records


def make_windowed_dataset(
    records: list[RawRecord],
    window_size: int,
    step_size: int,
    max_windows_per_file: int,
) -> WindowedDataset:
    train_records = [r for r in records if r.split == "train"]
    test_records = [r for r in records if r.split == "test"]
    if not train_records or not test_records:
        raise ValueError("Both train and test records are required.")

    x_train, y_train = _window_records(train_records, window_size, step_size, max_windows_per_file)
    x_test, y_test = _window_records(test_records, window_size, step_size, max_windows_per_file)
    x_train, x_test = zscore_from_train(x_train, x_test)

    names_by_label = {r.label: r.label_name for r in records}
    class_names = [names_by_label[i] for i in sorted(names_by_label)]
    return WindowedDataset(x_train, y_train, x_test, y_test, class_names, train_records, test_records)


def zscore_from_train(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean()
    std = x_train.std()
    if std < 1e-8:
        std = 1.0
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def _window_records(
    records: Iterable[RawRecord],
    window_size: int,
    step_size: int,
    max_windows_per_file: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for rec in records:
        sig = np.asarray(rec.signal, dtype=np.float32)
        signal_length = sig.shape[-1]
        count = 0
        for start in range(0, signal_length - window_size + 1, step_size):
            xs.append(sig[..., start : start + window_size])
            ys.append(rec.label)
            count += 1
            if max_windows_per_file > 0 and count >= max_windows_per_file:
                break
    if not xs:
        raise ValueError("No windows were produced. Check window size and signal length.")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def _load_local_bearing_signal(path: Path) -> np.ndarray:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    bearing = mat.get("bearing")
    if bearing is not None and hasattr(bearing, "gs"):
        return np.asarray(bearing.gs, dtype=np.float32).reshape(-1)
    return _first_numeric_vector(mat)


def _load_local_sampling_rate(path: Path) -> Optional[float]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    bearing = mat.get("bearing")
    if bearing is not None and hasattr(bearing, "sr"):
        return float(bearing.sr)
    return None


def _load_cwru_signal(path: Path) -> np.ndarray:
    return _load_mat_signal(path, preferred_key=None)


def _load_cwru_rpm(path: Path, load: int) -> float:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    for key, value in mat.items():
        if "rpm" not in key.lower():
            continue
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size:
            return float(finite[0])
    # Official nominal speeds for the 0, 1, 2, and 3 hp conditions.
    fallback = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
    return fallback[int(load)]


def _load_cwru_channel(path: Path, channel: str) -> np.ndarray:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    token = f"_{channel.upper()}_time".lower()
    for key, value in mat.items():
        if token in key.lower():
            return np.asarray(value, dtype=np.float32).reshape(-1)
    raise KeyError(f"No {channel}_time channel found in {path}")


def _load_mat_signal(path: Path, preferred_key: Optional[str]) -> np.ndarray:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if preferred_key:
        if preferred_key not in mat:
            raise KeyError(f"{preferred_key!r} not found in {path}")
        return np.asarray(mat[preferred_key], dtype=np.float32).reshape(-1)

    for pattern in ("DE.*time", "FE.*time", "BA.*time"):
        regex = re.compile(pattern, re.IGNORECASE)
        for key, value in mat.items():
            if regex.search(key):
                return np.asarray(value, dtype=np.float32).reshape(-1)
    return _first_numeric_vector(mat)


def _load_paderborn_channel(path: Path, channel: str) -> np.ndarray:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    payloads = [value for key, value in mat.items() if not key.startswith("__")]
    if not payloads:
        raise ValueError(f"No root data struct found in Paderborn file: {path}")
    payload = payloads[0]
    channels = np.atleast_1d(_mat_field(payload, "Y"))
    for entry in channels:
        name = str(_mat_field(entry, "Name"))
        if name == channel:
            data = np.asarray(_mat_field(entry, "Data"), dtype=np.float32).reshape(-1)
            if data.size < 1024:
                raise ValueError(f"Paderborn channel {channel!r} is too short in {path}")
            return data
    available = [str(_mat_field(entry, "Name")) for entry in channels]
    raise KeyError(f"Paderborn channel {channel!r} not found in {path}; available={available}")


def _resolve_multidomain_data_root(data_root: Path) -> Path:
    """Accept either MultiDomainBearing/ or its extracted/ directory."""

    direct = data_root / "extracted"
    if direct.is_dir():
        return direct
    if data_root.is_dir():
        return data_root
    raise FileNotFoundError(f"MultiDomainBearing root does not exist: {data_root}")


def _load_multidomain_signal(path: Path) -> np.ndarray:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "Data" not in mat:
        raise KeyError(f"MultiDomainBearing MAT file has no 'Data' vector: {path}")
    signal = np.asarray(mat["Data"], dtype=np.float32).reshape(-1)
    if signal.size < 1024:
        raise ValueError(f"MultiDomainBearing Data vector is too short in {path}")
    if not np.isfinite(signal).all():
        raise ValueError(f"MultiDomainBearing Data vector contains non-finite values in {path}")
    return signal


def _assert_multidomain_coverage(
    records: list[RawRecord],
    wanted_domains: Optional[set[int]],
    domain_count: int = 18,
) -> None:
    """Fail early when a requested composite domain lacks a fault class."""

    requested = (
        sorted(wanted_domains)
        if wanted_domains is not None
        else list(range(domain_count))
    )
    observed: dict[int, set[int]] = {}
    for record in records:
        observed.setdefault(record.domain_id, set()).add(record.label)
    expected = set(range(len(MULTIDOMAIN_LABELS)))
    incomplete = {
        domain: sorted(expected - observed.get(domain, set()))
        for domain in requested
        if observed.get(domain, set()) != expected
    }
    if incomplete:
        raise ValueError(
            "MultiDomainBearing domain/class coverage is incomplete. "
            f"Missing labels by domain: {incomplete}"
        )


def _mat_field(value: object, name: str) -> object:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
        return value[name]
    if isinstance(value, dict) and name in value:
        return value[name]
    raise KeyError(f"MATLAB struct field {name!r} is missing")


def _hust_xls_files(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        return []
    return sorted(
        path
        for path in data_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".xls"
    )


def _resolve_hust_data_root(data_root: Path) -> Path:
    direct_files = _hust_xls_files(data_root)
    if direct_files:
        return data_root
    nested_candidates = (data_root / "raw data", data_root / "raw")
    for nested in nested_candidates:
        if _hust_xls_files(nested):
            return nested
    recursive_files = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".xls"
    ) if data_root.is_dir() else []
    if recursive_files:
        parent_counts: dict[Path, int] = {}
        for path in recursive_files:
            parent_counts[path.parent] = parent_counts.get(path.parent, 0) + 1
        return max(
            parent_counts,
            key=lambda parent: (parent_counts[parent], str(parent)),
        )
    raise FileNotFoundError(
        "Could not find HUSTbearing .xls files recursively under "
        f"{data_root}."
    )


def _load_hust_text_signal(path: Path, channels: tuple[str, ...]) -> np.ndarray:
    channel_columns = {"X": 2, "Y": 3, "Z": 4}
    normalized_channels = tuple(channel.upper() for channel in channels)
    unknown = sorted(set(normalized_channels) - set(channel_columns))
    if unknown:
        raise ValueError(f"Unknown HUSTbearing channels {unknown}; choose X, Y, and/or Z.")

    data_line = None
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number, line in enumerate(stream):
            if line.strip() == "Data":
                data_line = line_number + 1
                break
    if data_line is None:
        raise ValueError(f"No Data marker found in HUSTbearing file: {path}")

    values = np.loadtxt(
        path,
        delimiter="\t",
        skiprows=data_line,
        usecols=tuple(channel_columns[channel] for channel in normalized_channels),
        dtype=np.float32,
    )
    if values.ndim == 1:
        values = values[:, None]
    return np.ascontiguousarray(values.T, dtype=np.float32)


def _first_numeric_vector(mat: dict) -> np.ndarray:
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.size >= 1024 and np.issubdtype(arr.dtype, np.number):
            return arr.astype(np.float32).reshape(-1)
    raise ValueError("No numeric signal vector found in .mat file")


def _local_fault_position(label_name: str) -> str:
    if label_name == "InnerRaceFault":
        return "inner race"
    if label_name == "OuterRaceFault":
        return "outer race"
    return "normal"


def _maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _cwru19_labels(fault_position: str, fault_size: Optional[float]) -> tuple[int, int, str]:
    size_key = {
        0.007: ("007", 0),
        0.014: ("014", 1),
        0.021: ("021", 2),
    }.get(fault_size)
    if size_key is None:
        raise ValueError(f"Unsupported CWRU fault size for 19-class setup: {fault_size}")
    size_name, size_offset = size_key

    position = fault_position.lower()
    if "inner" in position:
        group_offset = 0
        fault_name = "InnerRace"
    elif "ball" in position:
        group_offset = 3
        fault_name = "Ball"
    elif "outer" in position:
        group_offset = 6
        fault_name = "OuterRace"
    else:
        raise ValueError(f"Unsupported CWRU fault position for 19-class setup: {fault_position}")

    fan_label = 1 + group_offset + size_offset
    drive_label = 10 + group_offset + size_offset
    return fan_label, drive_label, f"{fault_name}_{size_name}"


def _paderborn_label(bearing_id: str) -> tuple[int, str, str]:
    bearing = bearing_id.upper()
    if re.fullmatch(r"K\d{3}", bearing):
        return 0, "Healthy", "normal"
    if bearing.startswith("KI"):
        return 1, "InnerRace", "inner race"
    if bearing.startswith(("KA", "KB")):
        return 2, "OuterRace", "outer race"
    raise ValueError(f"Unsupported Paderborn bearing ID: {bearing_id}")
