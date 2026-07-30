from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.continual import (  # noqa: E402
    ClassDomainBatchSampler,
    GlobalRelationSnapshot,
    summarize_accuracy_matrix,
)
from se_gscl.losses import (  # noqa: E402
    global_relation_snapshot_loss,
    snapshot_probabilities,
)
from se_gscl.semantics import (  # noqa: E402
    ProjectedTextPrototypeBank,
    TextEmbeddingCache,
    masked_mean_pool,
)


class P1SemanticPipelineTests(unittest.TestCase):
    def test_masked_pool_excludes_padding(self) -> None:
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
                [[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        pooled = masked_mean_pool(hidden, mask)
        torch.testing.assert_close(
            pooled,
            torch.tensor([[2.0, 3.0], [4.0, 8.0]]),
        )

    def test_text_cache_roundtrip_and_projection(self) -> None:
        cache = TextEmbeddingCache(
            embeddings=torch.randn(6, 12),
            class_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
            class_names=("A", "B", "C"),
            texts=("a1", "a2", "b1", "b2", "c1", "c2"),
            model_id="unit-test",
            ontology="test",
            version="v1",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache.save(temp_dir)
            restored = TextEmbeddingCache.load(temp_dir)
        torch.testing.assert_close(cache.embeddings, restored.embeddings)
        bank = ProjectedTextPrototypeBank(restored, semantic_dim=8)
        prototypes = bank()
        self.assertEqual(prototypes.shape, (3, 8))
        torch.testing.assert_close(
            prototypes.norm(dim=1),
            torch.ones(3),
            atol=1e-5,
            rtol=1e-5,
        )
        frozen = bank.freeze("frozen-v1")
        self.assertFalse(frozen.prototypes.requires_grad)

    def test_balanced_sampler_constructs_cross_domain_pairs(self) -> None:
        labels = np.asarray([0] * 4 + [1] * 8 + [2] * 12 + [3] * 16)
        domains = np.asarray(
            [0, 0, 1, 1]
            + [0] * 4
            + [1] * 4
            + [0] * 6
            + [1] * 6
            + [0] * 8
            + [1] * 8
        )
        sampler = ClassDomainBatchSampler(labels, domains, batch_size=16, seed=7)
        for batch in sampler:
            batch_labels = labels[batch]
            batch_domains = domains[batch]
            counts = [int(np.sum(batch_labels == label)) for label in range(4)]
            self.assertLessEqual(max(counts) - min(counts), 1)
            for label in range(4):
                self.assertGreaterEqual(
                    len(np.unique(batch_domains[batch_labels == label])),
                    2,
                )

    def test_relation_snapshot_loss_only_uses_replay_rows(self) -> None:
        old_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = snapshot_probabilities(old_logits)
        current = old_logits.clone().requires_grad_(True)
        mask = torch.tensor([True, False])
        loss = global_relation_snapshot_loss(current, targets, mask)
        self.assertLess(float(loss.detach()), 1e-6)
        changed = current.detach().clone()
        changed[0] = torch.tensor([0.0, 2.0])
        changed[1] = torch.tensor([20.0, -20.0])
        changed_loss = global_relation_snapshot_loss(changed, targets, mask)
        self.assertGreater(float(changed_loss), 0.1)

    def test_relation_snapshot_roundtrip_and_lookup(self) -> None:
        snapshot = GlobalRelationSnapshot(
            sample_ids=torch.tensor([10, 12]),
            probabilities=torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
            version="stage-0",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "snapshot.npz"
            snapshot.save(path)
            restored = GlobalRelationSnapshot.load(path)
        rows, mask = restored.lookup(torch.tensor([9, 10, 12]))
        self.assertEqual(mask.tolist(), [False, True, True])
        torch.testing.assert_close(rows, snapshot.probabilities)

    def test_sequence_metrics_use_only_seen_stage_history(self) -> None:
        matrix = np.asarray(
            [
                [0.80, 0.40, 0.30],
                [0.70, 0.90, 0.35],
                [0.75, 0.85, 0.95],
            ]
        )
        summary = summarize_accuracy_matrix(matrix)
        self.assertAlmostEqual(summary["final_average_accuracy"], 0.85)
        self.assertAlmostEqual(summary["average_forgetting"], 0.05)
        self.assertAlmostEqual(summary["maximum_forgetting"], 0.05)
        self.assertAlmostEqual(summary["average_backward_transfer"], -0.05)
        self.assertAlmostEqual(
            summary["average_old_domain_retention"],
            (0.75 / 0.80 + 0.85 / 0.90) / 2.0,
        )
        np.testing.assert_allclose(
            summary["stage_seen_average"],
            [0.80, 0.80, 0.85],
        )


if __name__ == "__main__":
    unittest.main()
