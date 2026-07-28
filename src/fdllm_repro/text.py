"""Fault description generation for data-text alignment."""

from __future__ import annotations

from collections import defaultdict

from .datasets import RawRecord


def build_class_descriptions(records: list[RawRecord], class_names: list[str]) -> list[str]:
    by_label: dict[int, list[RawRecord]] = defaultdict(list)
    for rec in records:
        by_label[rec.label].append(rec)

    descriptions: list[str] = []
    for label, name in enumerate(class_names):
        class_records = by_label.get(label)
        if not class_records:
            descriptions.append(f"This bearing operating state is {name}.")
            continue
        descriptions.append(_describe_class(name, class_records))
    return descriptions


def _describe_class(class_name: str, records: list[RawRecord]) -> str:
    rec = records[0]
    if class_name.lower() in {"normal", "healthy"}:
        return (
            "This is a healthy rolling bearing state. The inner race, outer race, "
            "rolling elements, and cage have no labeled localized fault. Its "
            "vibration should not contain persistent fault-characteristic impacts."
        )

    fault = rec.fault_position or rec.label_name
    article = "an" if fault[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    sentence = "This rolling bearing state is used for vibration-based fault diagnosis"
    if rec.bearing_position:
        sentence += f" at the {rec.bearing_position}"
    sentence += f". The diagnostic class is {class_name}. It represents {article} {fault} fault"
    if rec.severity:
        sentence += f" with {rec.severity} severity"
    if rec.fault_size is not None:
        unit = rec.fault_size_unit or "dataset units"
        sentence += f" and nominal fault size {rec.fault_size:g} {unit}"
    sentence += ". "
    sentence += _fault_mechanism(fault)
    return sentence


def _fault_mechanism(fault_position: str) -> str:
    fault = fault_position.lower()
    if "inner and outer" in fault or "compound" in fault:
        return (
            "Defects on both races can produce superimposed periodic impacts, "
            "harmonics, and speed-dependent modulation in the measured vibration."
        )
    if "inner" in fault:
        return (
            "As the inner race rotates through the load zone, rolling-element "
            "passages can excite periodic impacts near the inner-race fault frequency."
        )
    if "outer" in fault:
        return (
            "Rolling elements repeatedly pass the stationary outer-race defect, "
            "which can excite periodic impacts near the outer-race fault frequency."
        )
    if "ball" in fault or "rolling" in fault:
        return (
            "A rolling-element defect can generate modulated impacts associated "
            "with ball spin and cage motion as it enters and leaves the load zone."
        )
    return "The fault can change the impulsive and spectral structure of the vibration signal."
