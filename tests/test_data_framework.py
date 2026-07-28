from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.continual_fdllm.domain_windows import build_protocol_splits  # noqa: E402
from fdllm_repro.datasets import load_paderborn  # noqa: E402
from se_gscl.data import build_manifest_rows, manifest_summary  # noqa: E402


class UnifiedDataFrameworkTests(unittest.TestCase):
    def test_paderborn_adapter_parses_labels_conditions_and_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_paderborn(root / "N15_M07_F10_K006_1.mat", offset=0)
            self._write_paderborn(root / "N09_M07_F10_KI01_1.mat", offset=1)
            self._write_paderborn(root / "N15_M01_F10_KA01_1.mat", offset=2)

            records = load_paderborn(root)
            by_bearing = {row.bearing_id: row for row in records}
            self.assertEqual(by_bearing["K006"].label_name, "Healthy")
            self.assertEqual(by_bearing["K006"].domain_id, 0)
            self.assertEqual(by_bearing["KI01"].label_name, "InnerRace")
            self.assertEqual(by_bearing["KI01"].domain_id, 1)
            self.assertEqual(by_bearing["KA01"].label_name, "OuterRace")
            self.assertEqual(by_bearing["KA01"].domain_id, 2)
            self.assertEqual(by_bearing["KI01"].signal.shape, (4096,))
            self.assertEqual(by_bearing["KI01"].sampling_rate, 64000.0)

    def test_manifest_is_deterministic_and_contains_group_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_paderborn(root / "N15_M07_F04_KI01_1.mat", offset=3)
            rows = build_manifest_rows(load_paderborn(root))
            first = manifest_summary(rows)
            second = manifest_summary(rows)

            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(first["bearing_ids"], ["KI01"])
            self.assertEqual(first["domains"], {"3": 1})
            self.assertEqual(rows[0].speed_rpm, 1500.0)
            self.assertEqual(rows[0].radial_force_n, 400.0)

    def test_protocol_uses_bearing_groups_when_available(self) -> None:
        dataset = _bearing_group_dataset()
        train, val, test, audit = build_protocol_splits(
            dataset,
            domain_order=[0],
            train_ratio=0.5,
            val_ratio=0.25,
            seed=7,
        )

        train_groups = set(train[0]["group_id"])
        val_groups = set(val[0]["group_id"])
        test_groups = set(test[0]["group_id"])
        self.assertFalse(train_groups & val_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(val_groups & test_groups)
        self.assertTrue(all(row.method == "bearing_group" for row in audit))

    @staticmethod
    def _write_paderborn(path: Path, offset: int) -> None:
        signal = np.arange(4096, dtype=np.float32) + float(offset)
        payload = {
            "Y": np.array(
                [
                    {
                        "Name": "vibration_1",
                        "Data": signal,
                        "Raster": "HostService",
                    }
                ],
                dtype=object,
            )
        }
        savemat(path, {path.stem: payload})


def _bearing_group_dataset() -> dict[str, object]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[str] = []
    files: list[str] = []
    labels: list[str] = []
    for label in (0, 1):
        for bearing_index in range(4):
            group = f"Y{label}_B{bearing_index}"
            xs.append(np.full(32, label * 10 + bearing_index, dtype=np.float32))
            ys.append(label)
            groups.append(group)
            files.append(f"{group}_run1.mat")
            labels.append(f"class_{label}")
    n = len(xs)
    return {
        "x": np.stack(xs),
        "y": np.asarray(ys, dtype=np.int64),
        "load": np.zeros(n, dtype=np.int64),
        "domain_id": np.zeros(n, dtype=np.int64),
        "file_id": files,
        "group_id": groups,
        "bearing_id": groups,
        "window_index": np.zeros(n, dtype=np.int64),
        "window_start": np.zeros(n, dtype=np.int64),
        "sampling_rate": np.full(n, 64000.0, dtype=np.float32),
        "label_name": labels,
        "condition_name": ["D0"] * n,
        "sample_id": np.arange(n, dtype=np.int64),
        "class_names": ["class_0", "class_1"],
        "records": [],
        "window_size": 32,
        "step_size": 32,
    }


if __name__ == "__main__":
    unittest.main()
