"""Domain-aware CWRU window builder for continual FD-LLM experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from fdllm_repro.datasets import RawRecord, load_records


DomainWindowDataset = dict[str, object]


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float
    fitted_domain: int
    fitted_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SplitAuditRow:
    domain_id: int
    label: int
    file_id: str
    split: str
    method: str
    windows: int


def build_domain_window_dataset(
    data_root: str | Path,
    dataset: str = "cwru19",
    domains: Iterable[int] | None = None,
    window_size: int = 1024,
    step_size: int = 1024,
    max_windows_per_file: int | None = None,
    normalize: bool = True,
    seed: int = 42,
) -> DomainWindowDataset:
    """Build domain-aware windows while keeping per-window audit metadata.

    P0 callers should pass ``normalize=False``, split by file/record blocks,
    fit normalization on the first domain's training split only, and then apply
    those frozen statistics to every split. ``normalize=True`` is retained only
    for backward compatibility with the copied exploratory scripts.
    """

    del seed  # deterministic window order is inherited from sorted file order.
    records = load_records(dataset, Path(data_root), None, domains=domains)
    max_count = 0 if max_windows_per_file is None else int(max_windows_per_file)

    xs: list[np.ndarray] = []
    ys: list[int] = []
    loads: list[int] = []
    domain_ids: list[int] = []
    file_ids: list[str] = []
    window_indices: list[int] = []
    window_starts: list[int] = []
    sampling_rates: list[float] = []
    label_names: list[str] = []
    condition_names: list[str] = []
    group_ids: list[str] = []
    source_record_ids: list[str] = []
    dataset_names: list[str] = []
    bearing_ids: list[str] = []
    sensor_ids: list[str] = []
    speed_rpms: list[float] = []
    torque_nms: list[float] = []
    radial_forces: list[float] = []
    sample_ids: list[int] = []

    for rec in records:
        load = int(rec.load if rec.load is not None else -1)
        domain_id = int(rec.domain_id if rec.domain_id >= 0 else load)
        for local_window_index, (window_start, window) in enumerate(
            _iter_record_windows(rec, window_size, step_size, max_count)
        ):
            sample_ids.append(len(sample_ids))
            xs.append(window)
            ys.append(int(rec.label))
            loads.append(load)
            domain_ids.append(domain_id)
            file_ids.append(rec.file_id)
            window_indices.append(local_window_index)
            window_starts.append(window_start)
            sampling_rates.append(float(rec.sampling_rate or 0.0))
            label_names.append(rec.label_name)
            condition_names.append(rec.condition_name or f"domain_{domain_id}")
            source_id = rec.source_record_id or rec.file_id
            # Source-disjoint MultiDomainBearing protocols contain three exact
            # speeds inside each slow/fast group. Split by exact speed so every
            # class uses the same speed for train/validation/test and no raw
            # recording crosses those splits.
            record_dataset = rec.dataset_name or dataset
            group_id = (
                f"{rec.bearing_id}|{int(rec.speed_rpm or 0)}rpm"
                if record_dataset.endswith(("_atomic", "_disjoint18"))
                else (
                    source_id
                    if record_dataset.startswith("multidomain")
                    else rec.bearing_id or source_id
                )
            )
            group_ids.append(group_id)
            source_record_ids.append(source_id)
            dataset_names.append(rec.dataset_name or dataset)
            bearing_ids.append(rec.bearing_id or group_id)
            sensor_ids.append(rec.sensor_id or "+".join(rec.sampling_channels))
            speed_rpms.append(float("nan") if rec.speed_rpm is None else float(rec.speed_rpm))
            torque_nms.append(float("nan") if rec.torque_nm is None else float(rec.torque_nm))
            radial_forces.append(float("nan") if rec.radial_force_n is None else float(rec.radial_force_n))

    if not xs:
        raise ValueError("No windows were produced. Check window size, step size, and data root.")

    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    load_arr = np.asarray(loads, dtype=np.int64)
    domain_arr = np.asarray(domain_ids, dtype=np.int64)
    window_arr = np.asarray(window_indices, dtype=np.int64)
    start_arr = np.asarray(window_starts, dtype=np.int64)
    sampling_arr = np.asarray(sampling_rates, dtype=np.float32)
    sample_arr = np.asarray(sample_ids, dtype=np.int64)

    if normalize:
        old_mask = np.isin(load_arr, np.asarray([0, 1, 2], dtype=np.int64))
        reference = x[old_mask] if old_mask.any() else x
        mean = float(reference.mean())
        std = float(reference.std())
        if std < 1e-8:
            std = 1.0
        x = ((x - mean) / std).astype(np.float32)
        normalization = NormalizationStats(mean, std, -1, len(reference)).to_dict()
    else:
        normalization = None

    names_by_label = {rec.label: rec.label_name for rec in records}
    class_names = [names_by_label[i] for i in sorted(names_by_label)]

    return {
        "x": x,
        "y": y,
        "load": load_arr,
        "domain_id": domain_arr,
        "file_id": file_ids,
        "window_index": window_arr,
        "window_start": start_arr,
        "sampling_rate": sampling_arr,
        "label_name": label_names,
        "condition_name": condition_names,
        "group_id": group_ids,
        "source_record_id": source_record_ids,
        "dataset_name": dataset_names,
        "bearing_id": bearing_ids,
        "sensor_id": sensor_ids,
        "speed_rpm": np.asarray(speed_rpms, dtype=np.float32),
        "torque_nm": np.asarray(torque_nms, dtype=np.float32),
        "radial_force_n": np.asarray(radial_forces, dtype=np.float32),
        "class_names": class_names,
        "sample_id": sample_arr,
        "records": records,
        "window_size": int(window_size),
        "step_size": int(step_size),
        "normalization": normalization,
    }


def fit_normalization(dataset: DomainWindowDataset, fitted_domain: int) -> NormalizationStats:
    """Fit scalar z-score statistics from one training split only."""

    x = np.asarray(dataset["x"], dtype=np.float32)
    if len(x) == 0:
        raise ValueError("Cannot fit normalization on an empty dataset.")
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-8:
        std = 1.0
    return NormalizationStats(mean=mean, std=std, fitted_domain=int(fitted_domain), fitted_samples=len(x))


def apply_normalization(dataset: DomainWindowDataset, stats: NormalizationStats) -> DomainWindowDataset:
    """Apply frozen normalization statistics without mutating the input."""

    out = subset_by_mask(dataset, np.ones(len(np.asarray(dataset["y"])), dtype=bool))
    out["x"] = ((np.asarray(dataset["x"], dtype=np.float32) - stats.mean) / stats.std).astype(np.float32)
    out["normalization"] = stats.to_dict()
    return out


def build_protocol_splits(
    dataset: DomainWindowDataset,
    domain_order: Iterable[int],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[
    dict[int, DomainWindowDataset],
    dict[int, DomainWindowDataset],
    dict[int, DomainWindowDataset],
    list[SplitAuditRow],
]:
    """Create leakage-resistant train/validation/test splits for every domain.

    Strict file-group splitting is used when a domain/class has at least three
    independent files. CWRU19 usually has only one file for each load/class, so
    the auditable fallback uses contiguous record blocks and skips a guard band
    at both boundaries. This prevents overlapping neighboring windows from
    crossing splits while retaining every class in each split.
    """

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("val_ratio must be non-negative and train_ratio + val_ratio must be < 1.")

    train_by_domain: dict[int, DomainWindowDataset] = {}
    val_by_domain: dict[int, DomainWindowDataset] = {}
    test_by_domain: dict[int, DomainWindowDataset] = {}
    audit: list[SplitAuditRow] = []
    y_all = np.asarray(dataset["y"], dtype=np.int64)
    domain_all = np.asarray(dataset["domain_id"], dtype=np.int64)
    file_all = np.asarray(dataset["file_id"], dtype=object)
    group_all = np.asarray(dataset.get("group_id", dataset["file_id"]), dtype=object)
    dataset_names = set(str(v) for v in dataset.get("dataset_name", []))
    speed_group_protocol = any(
        name.endswith(("_atomic", "_disjoint18")) for name in dataset_names
    )
    split_seed = 1729 if speed_group_protocol else seed
    start_all = np.asarray(dataset["window_start"], dtype=np.int64)
    window_size = int(dataset.get("window_size", 1))
    step_size = max(1, int(dataset.get("step_size", window_size)))
    guard_windows = max(1, int(math.ceil(window_size / step_size)))

    for domain in [int(v) for v in domain_order]:
        split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        labels = sorted(np.unique(y_all[domain_all == domain]).tolist())
        if not labels:
            raise ValueError(f"Domain {domain} has no samples.")

        for label in labels:
            group_mask = (domain_all == domain) & (y_all == int(label))
            label_indices = np.flatnonzero(group_mask)
            groups = sorted(set(str(v) for v in group_all[label_indices]))
            if len(groups) >= 3 or (speed_group_protocol and len(groups) >= 2):
                assignments = _assign_file_groups(
                    groups,
                    train_ratio,
                    val_ratio,
                    split_seed
                    + domain * 1009
                    + (0 if speed_group_protocol else int(label)),
                )
                for split, split_groups in assignments.items():
                    chosen = label_indices[
                        np.isin(group_all[label_indices], np.asarray(split_groups, dtype=object))
                    ]
                    split_indices[split].extend(chosen.tolist())
                    for group_id in split_groups:
                        audit.append(
                            SplitAuditRow(
                                domain,
                                int(label),
                                group_id,
                                split,
                                (
                                    "exact_speed_group"
                                    if speed_group_protocol
                                    else "bearing_group"
                                ),
                                int(np.sum(group_all[chosen] == group_id)),
                            )
                        )
            else:
                files = sorted(set(str(v) for v in file_all[label_indices]))
                for file_id in files:
                    file_indices = label_indices[file_all[label_indices] == file_id]
                    file_indices = file_indices[np.argsort(start_all[file_indices])]
                    blocked = _blocked_indices(file_indices, train_ratio, val_ratio, guard_windows)
                    for split, chosen in blocked.items():
                        if len(chosen):
                            split_indices[split].extend(chosen.tolist())
                        audit.append(
                            SplitAuditRow(
                                domain,
                                int(label),
                                file_id,
                                split,
                                "blocked_with_guard",
                                len(chosen),
                            )
                        )

        for split in split_indices:
            split_indices[split].sort()
        if not split_indices["train"] or not split_indices["test"]:
            raise ValueError(
                f"Domain {domain} produced an empty train/test split. "
                "Increase max_windows_per_file or reduce window_size."
            )
        train_by_domain[domain] = subset_by_indices(dataset, split_indices["train"])
        val_by_domain[domain] = subset_by_indices(dataset, split_indices["val"])
        test_by_domain[domain] = subset_by_indices(dataset, split_indices["test"])
        if speed_group_protocol:
            assert_no_source_record_leakage(
                train_by_domain[domain],
                val_by_domain[domain],
                test_by_domain[domain],
            )

    return train_by_domain, val_by_domain, test_by_domain, audit


def assert_no_source_record_leakage(
    train_ds: DomainWindowDataset,
    val_ds: DomainWindowDataset,
    test_ds: DomainWindowDataset,
) -> None:
    """Raise when one raw recording appears in more than one data split."""

    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    sources = {
        name: set(str(v) for v in ds.get("source_record_id", ds["file_id"]))
        for name, ds in splits.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sources[left] & sources[right]
        if overlap:
            raise AssertionError(
                f"{left}/{right} share raw source records: {sorted(overlap)[:5]}"
            )


def assert_no_window_leakage(
    train_ds: DomainWindowDataset,
    val_ds: DomainWindowDataset,
    test_ds: DomainWindowDataset,
) -> None:
    """Raise when the same sample or overlapping windows cross split boundaries."""

    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    sample_sets = {name: set(np.asarray(ds["sample_id"], dtype=np.int64).tolist()) for name, ds in splits.items()}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sample_sets[left] & sample_sets[right]
        if overlap:
            raise AssertionError(f"{left}/{right} share sample ids: {sorted(overlap)[:5]}")

    window_size = int(train_ds.get("window_size", 1))
    intervals: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for split, ds in splits.items():
        intervals[split] = {}
        for file_id, start in zip(ds["file_id"], np.asarray(ds["window_start"], dtype=np.int64)):  # type: ignore[arg-type]
            intervals[split].setdefault(str(file_id), []).append((int(start), int(start) + window_size))
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        for file_id in set(intervals[left]) & set(intervals[right]):
            for a0, a1 in intervals[left][file_id]:
                for b0, b1 in intervals[right][file_id]:
                    if max(a0, b0) < min(a1, b1):
                        raise AssertionError(f"{left}/{right} contain overlapping windows from {file_id}.")


def split_by_load(dataset: DomainWindowDataset, loads: Iterable[int]) -> DomainWindowDataset:
    """Return a subset containing only the requested CWRU load domains."""

    wanted = np.asarray([int(v) for v in loads], dtype=np.int64)
    mask = np.isin(np.asarray(dataset["load"]), wanted)
    return subset_by_mask(dataset, mask)


def subset_by_indices(dataset: DomainWindowDataset, indices: np.ndarray | list[int]) -> DomainWindowDataset:
    """Return a subset by row indices while preserving shared class metadata."""

    idx = np.asarray(indices, dtype=np.int64)
    mask = np.zeros(len(np.asarray(dataset["y"])), dtype=bool)
    mask[idx] = True
    return subset_by_mask(dataset, mask)


def stratified_limit(dataset: DomainWindowDataset, max_samples: int | None, seed: int = 42) -> DomainWindowDataset:
    """Limit a dataset while keeping each class represented when possible."""

    if max_samples is None or max_samples <= 0:
        return dataset
    y = np.asarray(dataset["y"])
    if len(y) <= max_samples:
        return dataset

    rng = np.random.default_rng(seed)
    labels = np.unique(y)
    quota = max(1, max_samples // max(1, len(labels)))
    selected: list[int] = []
    leftovers: list[int] = []
    for label in labels:
        label_idx = np.flatnonzero(y == label)
        rng.shuffle(label_idx)
        selected.extend(label_idx[:quota].tolist())
        leftovers.extend(label_idx[quota:].tolist())

    if len(selected) < max_samples:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])
    selected = selected[:max_samples]
    selected.sort()
    return subset_by_indices(dataset, np.asarray(selected, dtype=np.int64))


def subset_by_sample_ids(dataset: DomainWindowDataset, sample_ids: Iterable[int]) -> DomainWindowDataset:
    wanted = set(int(v) for v in sample_ids)
    all_ids = np.asarray(dataset["sample_id"])
    mask = np.asarray([int(v) in wanted for v in all_ids], dtype=bool)
    return subset_by_mask(dataset, mask)


def subset_by_mask(dataset: DomainWindowDataset, mask: np.ndarray) -> DomainWindowDataset:
    mask = np.asarray(mask, dtype=bool)
    out: DomainWindowDataset = {}
    row_keys = {
        "x",
        "y",
        "load",
        "domain_id",
        "window_index",
        "window_start",
        "sampling_rate",
        "sample_id",
        "speed_rpm",
        "torque_nm",
        "radial_force_n",
    }
    list_row_keys = {
        "file_id",
        "label_name",
        "condition_name",
        "group_id",
        "source_record_id",
        "dataset_name",
        "bearing_id",
        "sensor_id",
    }
    for key, value in dataset.items():
        if key in row_keys:
            out[key] = np.asarray(value)[mask]
        elif key in list_row_keys:
            out[key] = [item for item, keep in zip(value, mask) if keep]  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def merge_datasets(*datasets: DomainWindowDataset) -> DomainWindowDataset:
    """Concatenate row fields from datasets with compatible class metadata."""

    if not datasets:
        raise ValueError("At least one dataset is required.")
    merged: DomainWindowDataset = {
        "class_names": datasets[0]["class_names"],
        "records": datasets[0].get("records", []),
    }
    numeric_row_keys = [
        "x",
        "y",
        "load",
        "domain_id",
        "window_index",
        "window_start",
        "sampling_rate",
        "sample_id",
        "speed_rpm",
        "torque_nm",
        "radial_force_n",
    ]
    for key in numeric_row_keys:
        if all(key in ds for ds in datasets):
            merged[key] = np.concatenate([np.asarray(ds[key]) for ds in datasets], axis=0)
    list_row_keys = [
        "file_id",
        "label_name",
        "condition_name",
        "group_id",
        "source_record_id",
        "dataset_name",
        "bearing_id",
        "sensor_id",
    ]
    for key in list_row_keys:
        if not all(key in ds for ds in datasets):
            continue
        rows: list[str] = []
        for ds in datasets:
            rows.extend(list(ds[key]))  # type: ignore[arg-type]
        merged[key] = rows
    return merged


def _iter_record_windows(
    rec: RawRecord,
    window_size: int,
    step_size: int,
    max_windows_per_file: int,
) -> Iterable[tuple[int, np.ndarray]]:
    signal = np.asarray(rec.signal, dtype=np.float32)
    signal_length = signal.shape[-1]
    count = 0
    for start in range(0, signal_length - window_size + 1, step_size):
        yield start, signal[..., start : start + window_size]
        count += 1
        if max_windows_per_file > 0 and count >= max_windows_per_file:
            break


def _assign_file_groups(
    files: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n < 2:
        raise ValueError("At least two source groups are required for a train/test split.")
    if n == 2:
        return {"train": shuffled[:1], "val": [], "test": shuffled[1:]}
    n_train = min(n - 2, max(1, int(round(n * train_ratio))))
    n_val = min(n - n_train - 1, max(1, int(round(n * val_ratio))))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _blocked_indices(
    indices: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    guard_windows: int,
) -> dict[str, np.ndarray]:
    n = len(indices)
    if n < 6 + 4 * guard_windows:
        raise ValueError(
            f"A record needs at least {6 + 4 * guard_windows} windows for guarded blocked splitting; got {n}."
        )
    train_boundary = max(2, int(math.floor(n * train_ratio)))
    val_boundary = max(train_boundary + 2, int(math.floor(n * (train_ratio + val_ratio))))
    val_boundary = min(n - 2, val_boundary)

    train_end = max(1, train_boundary - guard_windows)
    val_start = min(n, train_boundary + guard_windows)
    val_end = max(val_start, val_boundary - guard_windows)
    test_start = min(n, val_boundary + guard_windows)
    return {
        "train": indices[:train_end],
        "val": indices[val_start:val_end],
        "test": indices[test_start:],
    }
