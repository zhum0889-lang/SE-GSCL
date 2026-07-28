"""Dataset-calibrated physical attributes for bearing symptom supervision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from scipy.signal import hilbert
except ModuleNotFoundError:  # pragma: no cover - real experiments require scipy.
    hilbert = None


PHYSICS_KEYS = (
    "stable_vibration",
    "no_characteristic_peak",
    "low_impulsiveness",
    "bpfi_impact_train",
    "bpfi_rotational_sidebands",
    "inner_resonance_bursts",
    "bsf_components",
    "bsf_load_zone_modulation",
    "irregular_impacts",
    "bpfo_impact_train",
    "bpfo_stationary_amplitude",
    "outer_resonance_bursts",
)


@dataclass(frozen=True)
class BearingKinematics:
    """Characteristic-frequency ratios relative to shaft frequency."""

    bpfi_ratio: float
    bpfo_ratio: float
    bsf_ratio: float
    ftf_ratio: float
    name: str = "bearing"


# CWRU drive-end SKF 6205-2RS JEM bearing at zero contact angle.
CWRU_DRIVE_END_KINEMATICS = BearingKinematics(
    bpfi_ratio=5.4152,
    bpfo_ratio=3.5848,
    bsf_ratio=2.3570,
    ftf_ratio=0.3983,
    name="CWRU drive-end SKF 6205-2RS JEM",
)


@dataclass(frozen=True)
class PhysicalAttributeBatch:
    values: np.ndarray
    reliability: np.ndarray
    keys: tuple[str, ...] = PHYSICS_KEYS

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape [N,R].")
        if self.reliability.shape != self.values.shape:
            raise ValueError("reliability must match values.")
        if self.values.shape[1] != len(self.keys):
            raise ValueError("Attribute count must match keys.")


@dataclass(frozen=True)
class RobustAttributeCalibrator:
    """Map raw attributes to [0,1] using initial-domain train statistics."""

    keys: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90

    @classmethod
    def fit(
        cls,
        batch: PhysicalAttributeBatch,
        *,
        lower_quantile: float = 0.10,
        upper_quantile: float = 0.90,
    ) -> "RobustAttributeCalibrator":
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("Invalid calibration quantiles.")
        lows: list[float] = []
        highs: list[float] = []
        for index in range(batch.values.shape[1]):
            valid = (
                batch.reliability[:, index] > 0
            ) & np.isfinite(batch.values[:, index])
            if not valid.any():
                lows.append(0.0)
                highs.append(1.0)
                continue
            values = batch.values[valid, index]
            low = float(np.quantile(values, lower_quantile))
            high = float(np.quantile(values, upper_quantile))
            if high - low < 1e-8:
                high = low + 1.0
            lows.append(low)
            highs.append(high)
        return cls(
            keys=batch.keys,
            low=tuple(lows),
            high=tuple(highs),
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    def transform(self, batch: PhysicalAttributeBatch) -> PhysicalAttributeBatch:
        if batch.keys != self.keys:
            raise ValueError("Physical attribute order does not match calibrator.")
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        scaled = (batch.values - low[None, :]) / (high - low)[None, :]
        scaled = np.clip(scaled, 0.0, 1.0).astype(np.float32)
        return PhysicalAttributeBatch(
            values=scaled,
            reliability=batch.reliability.astype(np.float32),
            keys=batch.keys,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


def extract_physical_attributes(
    signals: np.ndarray,
    sampling_rates: np.ndarray,
    speed_rpms: np.ndarray,
    *,
    physics_keys: Sequence[str] = PHYSICS_KEYS,
    kinematics: BearingKinematics = CWRU_DRIVE_END_KINEMATICS,
) -> PhysicalAttributeBatch:
    """Extract scale-invariant time, envelope-order, and resonance attributes."""

    if hilbert is None:
        raise ModuleNotFoundError(
            "scipy is required for Hilbert-envelope physical attributes."
        )
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3:
        raise ValueError("signals must have shape [N,L] or [N,C,L].")
    rates = np.asarray(sampling_rates, dtype=np.float64)
    speeds = np.asarray(speed_rpms, dtype=np.float64)
    if rates.shape != (len(x),) or speeds.shape != (len(x),):
        raise ValueError("sampling_rates and speed_rpms must have shape [N].")
    requested = tuple(str(value) for value in physics_keys)
    unknown = sorted(set(requested) - set(PHYSICS_KEYS))
    if unknown:
        raise ValueError(f"Unknown physics keys: {unknown}")

    values = np.zeros((len(x), len(requested)), dtype=np.float32)
    reliability = np.zeros_like(values)
    for sample_index in range(len(x)):
        channel_rows: list[dict[str, float]] = []
        channel_reliability: list[dict[str, float]] = []
        for channel in x[sample_index]:
            row, row_reliability = _channel_attributes(
                channel,
                float(rates[sample_index]),
                float(speeds[sample_index]),
                kinematics,
            )
            channel_rows.append(row)
            channel_reliability.append(row_reliability)
        for attribute_index, key in enumerate(requested):
            values[sample_index, attribute_index] = float(
                np.mean([row[key] for row in channel_rows])
            )
            reliability[sample_index, attribute_index] = float(
                np.min([row[key] for row in channel_reliability])
            )
    return PhysicalAttributeBatch(
        values=values,
        reliability=reliability,
        keys=requested,
    )


def build_symptom_soft_targets(
    calibrated: PhysicalAttributeBatch,
    labels: np.ndarray,
    symptom_class_ids: Sequence[int],
    *,
    positive_floor: float = 0.25,
    negative_target: float = 0.02,
    negative_weight: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Create class-gated soft symptom targets and reliability weights."""

    if not 0.0 <= negative_target < positive_floor <= 1.0:
        raise ValueError("Expected negative_target < positive_floor in [0,1].")
    if not 0.0 <= negative_weight <= 1.0:
        raise ValueError("negative_weight must be in [0,1].")
    labels_array = np.asarray(labels, dtype=np.int64)
    class_ids = np.asarray(symptom_class_ids, dtype=np.int64)
    if labels_array.shape != (len(calibrated.values),):
        raise ValueError("labels must have shape [N].")
    if class_ids.shape != (calibrated.values.shape[1],):
        raise ValueError("symptom_class_ids must match attribute count.")
    active = labels_array[:, None] == class_ids[None, :]
    positive = positive_floor + (1.0 - positive_floor) * calibrated.values
    targets = np.where(active, positive, negative_target).astype(np.float32)
    class_weights = np.where(active, 1.0, negative_weight).astype(np.float32)
    weights = calibrated.reliability * class_weights
    return targets, weights.astype(np.float32)


def _channel_attributes(
    signal: np.ndarray,
    sampling_rate: float,
    speed_rpm: float,
    kinematics: BearingKinematics,
) -> tuple[dict[str, float], dict[str, float]]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    values = values - np.mean(values)
    eps = 1e-12
    rms = float(np.sqrt(np.mean(values**2) + eps))
    normalized = values / max(rms, eps)
    kurtosis = float(np.mean(normalized**4))
    crest = float(np.max(np.abs(normalized)))
    impulsiveness = max(0.0, kurtosis - 3.0) + max(0.0, crest - 3.0)

    segments = np.array_split(values, min(8, len(values)))
    segment_rms = np.asarray(
        [np.sqrt(np.mean(segment**2) + eps) for segment in segments]
    )
    rms_cv = float(np.std(segment_rms) / (np.mean(segment_rms) + eps))

    spectrum = np.abs(np.fft.rfft(values)) ** 2
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / max(sampling_rate, eps))
    high_frequency = frequencies >= 0.25 * (sampling_rate / 2.0)
    high_frequency_ratio = float(
        spectrum[high_frequency].sum() / (spectrum[1:].sum() + eps)
    )
    resonance = impulsiveness * np.sqrt(max(0.0, high_frequency_ratio))

    envelope = np.abs(hilbert(values))
    envelope = envelope - np.mean(envelope)
    envelope_nfft = 4 * len(envelope)
    envelope_spectrum = np.abs(np.fft.rfft(envelope, n=envelope_nfft))
    envelope_frequencies = np.fft.rfftfreq(
        envelope_nfft,
        d=1.0 / max(sampling_rate, eps),
    )
    order_valid = bool(
        sampling_rate > 0
        and np.isfinite(speed_rpm)
        and speed_rpm > 0
        and len(values) >= 64
    )
    if order_valid:
        shaft_hz = speed_rpm / 60.0
        orders = envelope_frequencies / shaft_hz
        bpfi = _harmonic_prominence(
            orders,
            envelope_spectrum,
            kinematics.bpfi_ratio,
        )
        bpfo = _harmonic_prominence(
            orders,
            envelope_spectrum,
            kinematics.bpfo_ratio,
        )
        bsf = _harmonic_prominence(
            orders,
            envelope_spectrum,
            kinematics.bsf_ratio,
        )
        bpfi_sidebands = _sideband_prominence(
            orders,
            envelope_spectrum,
            kinematics.bpfi_ratio,
            1.0,
        )
        bsf_sidebands = _sideband_prominence(
            orders,
            envelope_spectrum,
            kinematics.bsf_ratio,
            kinematics.ftf_ratio,
        )
        bpfo_sidebands = _sideband_prominence(
            orders,
            envelope_spectrum,
            kinematics.bpfo_ratio,
            1.0,
        )
        envelope_entropy = _normalized_spectral_entropy(
            orders,
            envelope_spectrum,
        )
        strongest_characteristic = max(bpfi, bpfo, bsf)
    else:
        bpfi = bpfo = bsf = 0.0
        bpfi_sidebands = bsf_sidebands = bpfo_sidebands = 0.0
        envelope_entropy = 0.0
        strongest_characteristic = 0.0

    raw = {
        "stable_vibration": -rms_cv,
        "no_characteristic_peak": -strongest_characteristic,
        "low_impulsiveness": -impulsiveness,
        "bpfi_impact_train": bpfi,
        "bpfi_rotational_sidebands": bpfi_sidebands,
        "inner_resonance_bursts": resonance,
        "bsf_components": bsf,
        "bsf_load_zone_modulation": bsf_sidebands,
        "irregular_impacts": impulsiveness * envelope_entropy,
        "bpfo_impact_train": bpfo,
        "bpfo_stationary_amplitude": bpfo - bpfo_sidebands,
        "outer_resonance_bursts": resonance,
    }
    order_keys = {
        "no_characteristic_peak",
        "bpfi_impact_train",
        "bpfi_rotational_sidebands",
        "bsf_components",
        "bsf_load_zone_modulation",
        "irregular_impacts",
        "bpfo_impact_train",
        "bpfo_stationary_amplitude",
    }
    confidence = {
        key: float(order_valid) if key in order_keys else 1.0
        for key in PHYSICS_KEYS
    }
    return raw, confidence


def _harmonic_prominence(
    orders: np.ndarray,
    spectrum: np.ndarray,
    base_order: float,
    harmonics: int = 3,
) -> float:
    valid = (orders >= 0.2) & (orders <= 20.0)
    noise = float(np.median(spectrum[valid])) if valid.any() else 0.0
    noise = max(noise, 1e-12)
    bandwidth = _order_bandwidth(orders)
    rows: list[float] = []
    for harmonic in range(1, harmonics + 1):
        target = base_order * harmonic
        mask = np.abs(orders - target) <= bandwidth
        peak = float(np.max(spectrum[mask])) if mask.any() else 0.0
        rows.append(float(np.log1p(peak / noise)))
    return float(np.mean(rows))


def _sideband_prominence(
    orders: np.ndarray,
    spectrum: np.ndarray,
    carrier_order: float,
    offset_order: float,
) -> float:
    valid = (orders >= 0.2) & (orders <= 20.0)
    noise = float(np.median(spectrum[valid])) if valid.any() else 0.0
    noise = max(noise, 1e-12)
    bandwidth = _order_bandwidth(orders)
    peaks: list[float] = []
    for harmonic in (1, 2):
        carrier = harmonic * carrier_order
        for direction in (-1.0, 1.0):
            target = carrier + direction * offset_order
            if target <= 0:
                continue
            mask = np.abs(orders - target) <= bandwidth
            peak = float(np.max(spectrum[mask])) if mask.any() else 0.0
            peaks.append(float(np.log1p(peak / noise)))
    return float(np.mean(peaks)) if peaks else 0.0


def _normalized_spectral_entropy(
    orders: np.ndarray,
    spectrum: np.ndarray,
) -> float:
    mask = (orders >= 0.2) & (orders <= 20.0)
    values = np.asarray(spectrum[mask], dtype=np.float64)
    if values.size <= 1 or values.sum() <= 0:
        return 0.0
    probabilities = values / values.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return entropy / np.log(len(probabilities))


def _order_bandwidth(orders: np.ndarray) -> float:
    if len(orders) < 2:
        return 0.10
    resolution = float(np.median(np.diff(orders)))
    return max(0.10, 1.5 * resolution)
