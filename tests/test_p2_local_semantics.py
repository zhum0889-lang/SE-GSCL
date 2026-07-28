from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.diagnostics import build_semantic_diagnostic_packet  # noqa: E402
from se_gscl.losses import local_symptom_alignment_loss  # noqa: E402
from se_gscl.models import LocalSymptomMatcher  # noqa: E402
from se_gscl.semantics import (  # noqa: E402
    ProjectedSymptomPrototypeBank,
    SymptomEmbeddingCache,
)


class P2LocalSemanticTests(unittest.TestCase):
    def _cache(self) -> SymptomEmbeddingCache:
        return SymptomEmbeddingCache(
            embeddings=torch.randn(6, 8),
            class_ids=torch.tensor([0, 0, 1, 2, 2, 2]),
            class_names=("Normal", "Inner", "Outer"),
            symptom_ids=("n1", "n2", "i1", "o1", "o2", "o3"),
            symptom_names=("N1", "N2", "I1", "O1", "O2", "O3"),
            texts=("nt1", "nt2", "it1", "ot1", "ot2", "ot3"),
            model_id="unit-test",
            ontology="bearing",
            version="v1",
        )

    def test_symptom_cache_roundtrip(self) -> None:
        cache = self._cache()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache.save(temp_dir)
            restored = SymptomEmbeddingCache.load(temp_dir)
        torch.testing.assert_close(cache.embeddings, restored.embeddings)
        self.assertEqual(restored.symptom_ids, cache.symptom_ids)
        self.assertEqual(restored.class_names, cache.class_names)

    def test_local_matcher_normalizes_hierarchical_probabilities(self) -> None:
        cache = self._cache()
        projection = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 5))
        bank = ProjectedSymptomPrototypeBank(
            cache,
            projection,
            cache.embeddings.mean(dim=0, keepdim=True),
        ).freeze()
        matcher = LocalSymptomMatcher(
            bank,
            top_tokens=3,
            learnable_symptom_weights=True,
        )
        output = matcher(torch.randn(4, 7, 5))
        self.assertEqual(output.token_similarities.shape, (4, 7, 6))
        self.assertEqual(output.class_scores.shape, (4, 3))
        torch.testing.assert_close(
            output.class_probabilities.sum(dim=1),
            torch.ones(4),
        )
        torch.testing.assert_close(
            output.joint_probabilities.sum(dim=1),
            torch.ones(4),
            atol=1e-5,
            rtol=1e-5,
        )
        for class_id in range(3):
            mask = bank.class_ids == class_id
            torch.testing.assert_close(
                output.conditional_symptom_probabilities[:, mask].sum(dim=1),
                torch.ones(4),
            )
        loss = local_symptom_alignment_loss(
            output,
            torch.tensor([0, 1, 2, 1]),
        )
        loss.backward()
        self.assertIsNotNone(matcher.token_adapter[1].weight.grad)
        self.assertIsNotNone(matcher.symptom_weight_logits.grad)

    def test_semantic_packet_keeps_global_and_local_evidence(self) -> None:
        packet = build_semantic_diagnostic_packet(
            sample_id=7,
            domain_id=2,
            class_names=("Normal", "Inner", "Outer"),
            symptom_names=("N1", "I1", "O1", "O2"),
            symptom_class_ids=(0, 1, 2, 2),
            global_probabilities=np.asarray([0.1, 0.7, 0.2]),
            local_probabilities=np.asarray([0.1, 0.6, 0.3]),
            symptom_joint_probabilities=np.asarray([0.1, 0.55, 0.25, 0.1]),
            local_weight=0.3,
            top_k=2,
            top_symptoms=2,
        )
        self.assertEqual(packet.predicted_class_name, "Inner")
        self.assertTrue(packet.global_local_agreement)
        self.assertEqual(len(packet.top_candidates), 2)
        self.assertEqual(packet.top_symptoms[0]["symptom_name"], "I1")
        self.assertGreater(packet.top1_top2_margin, 0.0)
        self.assertGreaterEqual(packet.normalized_entropy, 0.0)
        self.assertLessEqual(packet.normalized_entropy, 1.0)


if __name__ == "__main__":
    unittest.main()
