from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from se_gscl.data import audit_hust_protocol  # noqa: E402


class HustProtocolTests(unittest.TestCase):
    def test_complete_domain_class_grid_and_guarded_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            root.mkdir()
            for state in (
                "H",
                "0.5X_I",
                "I",
                "0.5X_O",
                "O",
                "0.5X_B",
                "B",
                "0.5X_C",
                "C",
            ):
                _write_hust_header(root / f"{state}_20Hz.xls", rows=262144)
            _write_hust_header(root / "H_VS_0_40_0Hz.xls", rows=262144)

            rows, summary = audit_hust_protocol(
                temp_dir,
                domains=[20],
                window_size=2048,
                step_size=1024,
                max_windows_per_file=60,
            )

        self.assertTrue(summary["protocol_ready"])
        self.assertTrue(summary["complete_domain_class_grid"])
        self.assertEqual(summary["expected_records"], 9)
        self.assertEqual(summary["observed_records"], 9)
        self.assertEqual(len(summary["excluded_variable_speed_files"]), 1)
        self.assertEqual(summary["totals"]["planned_windows"], 540)
        self.assertEqual(summary["totals"]["train_windows"], 306)
        self.assertEqual(summary["totals"]["val_windows"], 72)
        self.assertEqual(summary["totals"]["test_windows"], 90)
        self.assertTrue(all(row["excluded_guard_windows"] == 8 for row in rows))

    def test_missing_class_fails_protocol_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            root.mkdir()
            _write_hust_header(root / "H_20Hz.xls", rows=262144)

            _, summary = audit_hust_protocol(temp_dir, domains=[20])

        self.assertFalse(summary["protocol_ready"])
        self.assertEqual(summary["status"], "error")
        self.assertEqual(len(summary["missing_domain_class_pairs"]), 8)


def _write_hust_header(path: Path, rows: int) -> None:
    header = [
        f"Title:\t\t{path.stem}",
        "Parameters:",
        "Speed",
        "X",
        "Y",
        "Z",
        "",
        "DAQ Settings:",
        "Frequency Limit\t10000.000000",
        f"Total Data Rows\t{rows}",
        "Channels:",
        "Data",
    ]
    path.write_text("\n".join(header), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
