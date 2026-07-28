from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.continual_fdllm.domain_windows import (  # noqa: E402
    apply_normalization,
    assert_no_window_leakage,
    build_protocol_splits,
    fit_normalization,
)
from experiments.continual_fdllm.protocol import (  # noqa: E402
    assert_seen_only_matrix,
    update_persistent_memory,
)
from experiments.continual_fdllm.replay_buffer import ReplayBuffer, ReplayRecord  # noqa: E402


class P0ProtocolTests(unittest.TestCase):
    def test_guarded_block_split_has_no_overlap(self) -> None:
        dataset = _synthetic_window_dataset()
        train, val, test, audit = build_protocol_splits(
            dataset,
            domain_order=[0, 1],
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )
        self.assertTrue(audit)
        self.assertTrue(all(row.method == "blocked_with_guard" for row in audit))
        for domain in [0, 1]:
            assert_no_window_leakage(train[domain], val[domain], test[domain])
            self.assertGreater(len(train[domain]["y"]), 0)
            self.assertGreater(len(test[domain]["y"]), 0)

    def test_normalization_is_fitted_from_given_training_split_only(self) -> None:
        dataset = _minimal_dataset(np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32))
        stats = fit_normalization(dataset, fitted_domain=0)
        self.assertAlmostEqual(stats.mean, 4.0)
        self.assertAlmostEqual(stats.std, float(np.std(dataset["x"])))

        target = _minimal_dataset(np.asarray([[100.0, 102.0]], dtype=np.float32))
        normalized = apply_normalization(target, stats)
        expected = (np.asarray(target["x"]) - stats.mean) / stats.std
        np.testing.assert_allclose(normalized["x"], expected.astype(np.float32))
        self.assertEqual(normalized["normalization"]["fitted_domain"], 0)

    def test_persistent_memory_is_bounded_and_deduplicated(self) -> None:
        previous = ReplayBuffer([_record(1, 0, 0, 0.2), _record(2, 0, 1, 0.8)], capacity=2, version=1)
        current = [_record(2, 0, 1, 0.9), _record(3, 1, 0, 0.7), _record(4, 1, 1, 0.6)]
        updated = update_persistent_memory(
            previous,
            current,
            capacity=3,
            episode=1,
            seed=42,
            strategy="balanced_semantic_replay",
        )
        self.assertEqual(len(updated.records), 3)
        self.assertEqual(len(set(updated.sample_ids())), 3)
        self.assertEqual(updated.capacity, 3)
        self.assertEqual(updated.version, 2)

    def test_accuracy_matrix_rejects_future_domain_evaluation(self) -> None:
        valid = np.asarray(
            [
                [0.8, np.nan, np.nan],
                [0.7, 0.75, np.nan],
                [0.68, 0.72, 0.78],
            ],
            dtype=np.float32,
        )
        assert_seen_only_matrix(valid)
        invalid = valid.copy()
        invalid[0, 2] = 0.2
        with self.assertRaises(AssertionError):
            assert_seen_only_matrix(invalid)


def _synthetic_window_dataset() -> dict[str, object]:
    window_size = 8
    step_size = 4
    xs: list[np.ndarray] = []
    ys: list[int] = []
    domains: list[int] = []
    files: list[str] = []
    starts: list[int] = []
    sample_ids: list[int] = []
    labels: list[str] = []
    for domain in [0, 1]:
        for label in [0, 1]:
            file_id = f"d{domain}_y{label}.mat"
            for window_idx in range(30):
                xs.append(np.full(window_size, domain * 10 + label + window_idx / 100, dtype=np.float32))
                ys.append(label)
                domains.append(domain)
                files.append(file_id)
                starts.append(window_idx * step_size)
                sample_ids.append(len(sample_ids))
                labels.append(f"class_{label}")
    n = len(xs)
    return {
        "x": np.stack(xs),
        "y": np.asarray(ys, dtype=np.int64),
        "load": np.asarray(domains, dtype=np.int64),
        "domain_id": np.asarray(domains, dtype=np.int64),
        "file_id": files,
        "window_index": np.tile(np.arange(30, dtype=np.int64), 4),
        "window_start": np.asarray(starts, dtype=np.int64),
        "sampling_rate": np.full(n, 12000, dtype=np.float32),
        "label_name": labels,
        "class_names": ["class_0", "class_1"],
        "sample_id": np.asarray(sample_ids, dtype=np.int64),
        "records": [],
        "window_size": window_size,
        "step_size": step_size,
        "normalization": None,
    }


def _minimal_dataset(x: np.ndarray) -> dict[str, object]:
    n = len(x)
    return {
        "x": x,
        "y": np.zeros(n, dtype=np.int64),
        "load": np.zeros(n, dtype=np.int64),
        "domain_id": np.zeros(n, dtype=np.int64),
        "file_id": ["f"] * n,
        "window_index": np.arange(n, dtype=np.int64),
        "window_start": np.arange(n, dtype=np.int64) * x.shape[1],
        "sampling_rate": np.full(n, 1.0, dtype=np.float32),
        "label_name": ["class_0"] * n,
        "class_names": ["class_0"],
        "sample_id": np.arange(n, dtype=np.int64),
        "records": [],
        "window_size": int(x.shape[1]),
        "step_size": int(x.shape[1]),
        "normalization": None,
    }


def _record(sample_id: int, domain: int, label: int, priority: float) -> ReplayRecord:
    return ReplayRecord(
        sample_id=sample_id,
        domain_id=domain,
        load=domain,
        true_label=label,
        label_name=f"class_{label}",
        file_id=f"f{sample_id}",
        window_index=sample_id,
        predicted_label=label,
        is_correct=True,
        fse_entropy=0.2,
        top1_top2_margin=0.8,
        replay_priority=priority,
        confusion_type="correct",
        snapshot_probs=[0.9, 0.1] if label == 0 else [0.1, 0.9],
    )


if __name__ == "__main__":
    unittest.main()
