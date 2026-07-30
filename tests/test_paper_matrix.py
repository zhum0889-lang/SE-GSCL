from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_paper_downstream_matrix import build_p3_command  # noqa: E402
from scripts.run_paper_p1_matrix import (  # noqa: E402
    build_train_command,
    load_matrix,
)
from scripts.summarize_paper_matrix import _aggregate  # noqa: E402
from se_gscl.physics import PHYSICS_KEYS  # noqa: E402
from se_gscl.semantics import read_ontology  # noqa: E402


class PaperMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix_path = (
            ROOT / "configs" / "experiments" / "paper_matrix.json"
        )
        self.matrix = load_matrix(self.matrix_path)

    def test_matrix_has_three_seeds_and_second_dataset(self) -> None:
        self.assertEqual(self.matrix["seeds"], [42, 52, 62])
        self.assertIn("cwru4", self.matrix["datasets"])
        self.assertIn("hustbearing", self.matrix["datasets"])
        self.assertEqual(
            self.matrix["datasets"]["hustbearing"]["role"],
            "second_dataset",
        )
        self.assertIn(
            "lwf",
            {job["strategy"] for job in self.matrix["p1_jobs"]},
        )

    def test_p1_ablation_command_applies_loss_override(self) -> None:
        job = next(
            row
            for row in self.matrix["p1_jobs"]
            if row["id"] == "wo_relation"
        )
        command = build_train_command(
            python_bin="python",
            dataset="cwru4",
            data_root=Path("data"),
            text_cache=Path("cache"),
            output_dir=Path("out"),
            dataset_config=self.matrix["datasets"]["cwru4"],
            job=job,
            common=self.matrix["p1_common"],
            seed=52,
            device="cuda",
        )
        self.assertEqual(command[command.index("--lambda-rel") + 1], "0.0")
        self.assertEqual(command[command.index("--strategy") + 1], "full")
        self.assertEqual(command[command.index("--seed") + 1], "52")

    def test_p3_commands_preserve_seed_and_unlock_ablation(self) -> None:
        continuous = next(
            row
            for row in self.matrix["p3_jobs"]
            if row["id"] == "continuous_full"
        )
        command = build_p3_command(
            python_bin="python",
            job=continuous,
            p2_dir=Path("p2"),
            p31_dir=Path("p31"),
            model=Path("qwen"),
            output_dir=Path("out"),
            device="cuda",
            dtype="bfloat16",
            prompt_epochs=10,
            seed=62,
            local_files_only=True,
        )
        self.assertEqual(command[command.index("--seed") + 1], "62")
        self.assertIn("--local-files-only", command)

        unlocked = next(
            row
            for row in self.matrix["p3_jobs"]
            if row["id"] == "explanation_unlocked"
        )
        command = build_p3_command(
            python_bin="python",
            job=unlocked,
            p2_dir=Path("p2"),
            p31_dir=Path("p31"),
            model=Path("qwen"),
            output_dir=Path("out"),
            device="cuda",
            dtype="bfloat16",
            prompt_epochs=10,
            seed=42,
            local_files_only=False,
        )
        self.assertIn("--unlock-diagnosis", command)

    def test_hust_ontology_matches_loader_and_physics_contract(self) -> None:
        ontology = read_ontology(
            ROOT / "configs" / "semantics" / "hustbearing_faults_9.json"
        )
        expected_names = [
            "Healthy",
            "InnerRace_Medium",
            "InnerRace_Severe",
            "OuterRace_Medium",
            "OuterRace_Severe",
            "Ball_Medium",
            "Ball_Severe",
            "Compound_Medium",
            "Compound_Severe",
        ]
        self.assertEqual(
            [row["name"] for row in ontology["classes"]],
            expected_names,
        )
        symptoms = [
            symptom
            for row in ontology["classes"]
            for symptom in row["symptoms"]
        ]
        self.assertEqual(len(symptoms), 27)
        self.assertTrue(
            all(row["physics_key"] in PHYSICS_KEYS for row in symptoms)
        )

    def test_population_mean_std_aggregation(self) -> None:
        rows = [
            {"dataset": "cwru4", "job": "full", "seed": 42, "score": 0.8},
            {"dataset": "cwru4", "job": "full", "seed": 52, "score": 0.9},
            {"dataset": "cwru4", "job": "full", "seed": 62, "score": 1.0},
        ]
        summary = _aggregate(
            rows,
            identity_keys=("dataset", "job"),
        )
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["score_mean"], 0.9)
        self.assertAlmostEqual(
            summary[0]["score_std"],
            (2.0 / 300.0) ** 0.5,
        )
        self.assertEqual(summary[0]["num_seeds"], 3)

    def test_matrix_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "matrix.json"
            path.write_text(
                json.dumps(self.matrix),
                encoding="utf-8",
            )
            restored = load_matrix(path)
        self.assertEqual(restored["version"], self.matrix["version"])


if __name__ == "__main__":
    unittest.main()
