from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.continual_fdllm.domain_windows import build_domain_window_dataset  # noqa: E402
from fdllm_repro.datasets import load_hustbearing  # noqa: E402
from fdllm_repro.models import ConvLSTMSignalEncoder  # noqa: E402


class HustAdapterTests(unittest.TestCase):
    def test_text_export_is_parsed_as_three_axis_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw data"
            root.mkdir()
            _write_hust_record(root / "0.5X_I_20Hz.xls", rows=32)

            records = load_hustbearing(Path(temp_dir), domains=[20])
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.label, 1)
            self.assertEqual(record.label_name, "InnerRace_Medium")
            self.assertEqual(record.domain_id, 20)
            self.assertEqual(record.condition_name, "20Hz")
            self.assertEqual(record.fault_size, 0.15)
            self.assertEqual(record.fault_size_unit, "mm")
            self.assertEqual(record.signal.shape, (3, 32))
            np.testing.assert_allclose(record.signal[:, 0], [0.1, 0.2, 0.3])

    def test_domain_window_builder_preserves_channel_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            root.mkdir()
            _write_hust_record(root / "H_25hz.XLS", rows=48)

            dataset = build_domain_window_dataset(
                temp_dir,
                dataset="hustbearing",
                domains=[25],
                window_size=16,
                step_size=16,
                max_windows_per_file=2,
                normalize=False,
            )
            self.assertEqual(np.asarray(dataset["x"]).shape, (2, 3, 16))
            self.assertEqual(set(np.asarray(dataset["domain_id"]).tolist()), {25})
            self.assertEqual(dataset["condition_name"], ["25Hz", "25Hz"])

    def test_conv_lstm_accepts_hust_three_axis_windows(self) -> None:
        model = ConvLSTMSignalEncoder(
            embed_dim=16,
            hidden_dim=8,
            conv_dim=8,
            input_channels=3,
        )
        output = model(torch.randn(4, 3, 128))
        self.assertEqual(tuple(output.shape), (4, 16))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(4))


def _write_hust_record(path: Path, rows: int) -> None:
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
        "Spectral Lines\t12800",
        "Number of Blocks\t8",
        f"Total Data Rows\t{rows}",
        "Channels:",
        "Legend\tSpeed\tX\tY\tZ",
        "\tTacho1",
        "",
        "On/Off\tON\tON\tON\tON",
        "\tOFF",
        "",
        "Volts/Unit\t1.000000\t1.000000\t1.000000\t1.000000",
        "",
        "Data",
    ]
    data = [
        f"{index / 25600:.6f}\t0.0\t{0.1 + index:.4f}\t{0.2 + index:.4f}\t{0.3 + index:.4f}"
        for index in range(rows)
    ]
    path.write_text("\n".join(header + data), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
