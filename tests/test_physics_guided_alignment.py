from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.losses import physics_guided_local_alignment_loss  # noqa: E402
from se_gscl.models import LocalSymptomMatcher  # noqa: E402
from se_gscl.physics import (  # noqa: E402
    CWRU_DRIVE_END_KINEMATICS,
    HUST_ER16K_KINEMATICS,
    PHYSICS_KEYS,
    RobustAttributeCalibrator,
    build_symptom_soft_targets,
    extract_physical_attributes,
)
from se_gscl.semantics import (  # noqa: E402
    ProjectedSymptomPrototypeBank,
    SymptomEmbeddingCache,
)


class PhysicsGuidedAlignmentTests(unittest.TestCase):
    def test_hust_er16k_characteristic_frequency_ratios(self) -> None:
        self.assertAlmostEqual(HUST_ER16K_KINEMATICS.bpfi_ratio, 5.4087)
        self.assertAlmostEqual(HUST_ER16K_KINEMATICS.bpfo_ratio, 3.5913)
        self.assertAlmostEqual(HUST_ER16K_KINEMATICS.bsf_ratio, 2.3751)
        self.assertAlmostEqual(HUST_ER16K_KINEMATICS.ftf_ratio, 0.3990)

    def test_bpfo_modulation_activates_outer_race_attribute(self) -> None:
        sampling_rate = 12000.0
        speed_rpm = 1800.0
        shaft_hz = speed_rpm / 60.0
        bpfo_hz = CWRU_DRIVE_END_KINEMATICS.bpfo_ratio * shaft_hz
        time = np.arange(4096, dtype=np.float64) / sampling_rate
        carrier = np.sin(2.0 * np.pi * 2200.0 * time)
        modulation = 1.0 + 0.9 * np.sin(2.0 * np.pi * bpfo_hz * time)
        signal = (modulation * carrier).astype(np.float32)
        batch = extract_physical_attributes(
            signal[None, :],
            np.asarray([sampling_rate], dtype=np.float32),
            np.asarray([speed_rpm], dtype=np.float32),
        )
        bpfo = PHYSICS_KEYS.index("bpfo_impact_train")
        bpfi = PHYSICS_KEYS.index("bpfi_impact_train")
        self.assertGreater(batch.values[0, bpfo], batch.values[0, bpfi])
        self.assertEqual(batch.reliability[0, bpfo], 1.0)

    def test_calibration_and_class_gated_soft_targets(self) -> None:
        rng = np.random.default_rng(7)
        signals = rng.normal(size=(8, 2048)).astype(np.float32)
        raw = extract_physical_attributes(
            signals,
            np.full(8, 12000.0, dtype=np.float32),
            np.full(8, 1800.0, dtype=np.float32),
        )
        calibrated = RobustAttributeCalibrator.fit(raw).transform(raw)
        self.assertTrue(np.all(calibrated.values >= 0.0))
        self.assertTrue(np.all(calibrated.values <= 1.0))
        class_ids = np.repeat(np.arange(4), 3)
        labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
        targets, weights = build_symptom_soft_targets(
            calibrated,
            labels,
            class_ids,
        )
        self.assertEqual(targets.shape, (8, 12))
        self.assertEqual(weights.shape, targets.shape)
        self.assertTrue(np.all(targets[0, :3] >= 0.25))
        self.assertTrue(np.allclose(targets[0, 3:], 0.02))

    def test_physics_loss_updates_independent_text_projection(self) -> None:
        cache = SymptomEmbeddingCache(
            embeddings=torch.randn(12, 8),
            class_ids=torch.repeat_interleave(torch.arange(4), 3),
            class_names=("Normal", "Inner", "Ball", "Outer"),
            symptom_ids=tuple(f"s{i}" for i in range(12)),
            symptom_names=tuple(f"S{i}" for i in range(12)),
            physics_keys=PHYSICS_KEYS,
            texts=tuple(f"text {i}" for i in range(12)),
            model_id="unit-test",
            ontology="bearing",
            version="v1",
        )
        projection = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 5))
        bank = ProjectedSymptomPrototypeBank(
            cache,
            projection,
            cache.embeddings.mean(dim=0, keepdim=True),
        )
        anchor = bank().detach().clone()
        matcher = LocalSymptomMatcher(bank, top_tokens=3)
        output = matcher(torch.randn(4, 7, 5))
        targets = torch.full((4, 12), 0.02)
        for index in range(4):
            targets[index, index * 3 : (index + 1) * 3] = 0.8
        loss, components = physics_guided_local_alignment_loss(
            output,
            torch.arange(4),
            targets,
            torch.ones_like(targets),
            anchor,
        )
        loss.backward()
        self.assertIsNotNone(projection[1].weight.grad)
        self.assertGreater(float(components["physics"]), 0.0)


if __name__ == "__main__":
    unittest.main()
