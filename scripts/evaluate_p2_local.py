"""Evaluate P2 local symptom semantics on a frozen P1 specialist."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.continual_fdllm.domain_windows import (  # noqa: E402
    NormalizationStats,
    apply_normalization,
    build_domain_window_dataset,
    build_protocol_splits,
)
from se_gscl.diagnostics import build_semantic_diagnostic_packet  # noqa: E402
from se_gscl.losses import (  # noqa: E402
    local_symptom_alignment_loss,
    physics_guided_local_alignment_loss,
)
from se_gscl.models import LocalSymptomMatcher, SEGSCLSpecialist  # noqa: E402
from se_gscl.physics import (  # noqa: E402
    CWRU_DRIVE_END_KINEMATICS,
    RobustAttributeCalibrator,
    build_symptom_soft_targets,
    extract_physical_attributes,
)
from se_gscl.semantics import (  # noqa: E402
    FrozenPrototypeBank,
    ProjectedSymptomPrototypeBank,
    ProjectedTextPrototypeBank,
    SymptomEmbeddingCache,
    TextEmbeddingCache,
)


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        dataset: dict[str, object],
        *,
        symptom_targets: np.ndarray | None = None,
        symptom_target_weights: np.ndarray | None = None,
    ) -> None:
        self.x = torch.from_numpy(np.asarray(dataset["x"], dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(dataset["y"], dtype=np.int64))
        self.domains = torch.from_numpy(
            np.asarray(dataset["domain_id"], dtype=np.int64)
        )
        self.sample_ids = torch.from_numpy(
            np.asarray(dataset["sample_id"], dtype=np.int64)
        )
        self.symptom_targets = (
            None
            if symptom_targets is None
            else torch.from_numpy(
                np.asarray(symptom_targets, dtype=np.float32)
            )
        )
        self.symptom_target_weights = (
            None
            if symptom_target_weights is None
            else torch.from_numpy(
                np.asarray(symptom_target_weights, dtype=np.float32)
            )
        )
        if (self.symptom_targets is None) != (
            self.symptom_target_weights is None
        ):
            raise ValueError("Both symptom target arrays must be provided.")
        if (
            self.symptom_targets is not None
            and len(self.symptom_targets) != len(self.labels)
        ):
            raise ValueError("Symptom targets must match dataset length.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = {
            "x": self.x[index],
            "label": self.labels[index],
            "domain": self.domains[index],
            "sample_id": self.sample_ids[index],
        }
        if self.symptom_targets is not None:
            row["symptom_targets"] = self.symptom_targets[index]
            row["symptom_target_weights"] = self.symptom_target_weights[index]
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--global-text-cache", required=True)
    parser.add_argument("--symptom-text-cache", required=True)
    parser.add_argument("--p1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--top-tokens", type=int, default=4)
    parser.add_argument("--local-temperature", type=float, default=0.1)
    parser.add_argument("--local-weight", type=float, default=0.3)
    parser.add_argument("--learnable-symptom-weights", action="store_true")
    parser.add_argument("--physics-guided", action="store_true")
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-symptoms", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_physics(
    dataset: dict[str, object],
    physics_keys: tuple[str, ...],
):
    return extract_physical_attributes(
        np.asarray(dataset["x"], dtype=np.float32),
        np.asarray(dataset["sampling_rate"], dtype=np.float32),
        np.asarray(dataset["speed_rpm"], dtype=np.float32),
        physics_keys=physics_keys,
    )


def _fuse_probabilities(
    global_probabilities: np.ndarray,
    local_probabilities: np.ndarray,
    local_weight: float,
) -> np.ndarray:
    global_values = np.clip(global_probabilities, 1e-12, 1.0)
    local_values = np.clip(local_probabilities, 1e-12, 1.0)
    fused_log = (
        (1.0 - local_weight) * np.log(global_values)
        + local_weight * np.log(local_values)
    )
    fused = np.exp(fused_log - fused_log.max(axis=1, keepdims=True))
    return fused / fused.sum(axis=1, keepdims=True)


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    recalls = {
        str(class_id): float(
            np.mean(predictions[labels == class_id] == class_id)
        )
        for class_id in range(num_classes)
    }
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "per_class_recall": recalls,
    }


def _grounding_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    symptom_class_ids: np.ndarray,
) -> dict[str, float]:
    denominator = max(float(weights.sum()), 1.0)
    weighted_mae = float((np.abs(predicted - targets) * weights).sum())
    active = labels[:, None] == symptom_class_ids[None, :]
    active_weights = weights * active
    active_denominator = max(float(active_weights.sum()), 1.0)
    active_mae = float(
        (np.abs(predicted - targets) * active_weights).sum()
        / active_denominator
    )
    top_matches: list[bool] = []
    for index, label in enumerate(labels):
        mask = symptom_class_ids == int(label)
        valid = mask & (weights[index] > 0)
        if not valid.any():
            continue
        candidates = np.flatnonzero(valid)
        predicted_top = candidates[np.argmax(predicted[index, candidates])]
        target_top = candidates[np.argmax(targets[index, candidates])]
        top_matches.append(bool(predicted_top == target_top))
    return {
        "weighted_mae": weighted_mae / denominator,
        "true_class_symptom_mae": active_mae,
        "true_class_top1_agreement": (
            float(np.mean(top_matches)) if top_matches else 0.0
        ),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.local_weight <= 1.0:
        raise ValueError("local_weight must be in [0,1].")
    device = torch.device(args.device)
    p1_dir = Path(args.p1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((p1_dir / "p1_report.json").read_text(encoding="utf-8"))
    if args.physics_guided and str(report["dataset"]) not in {"cwru4", "cwru10"}:
        raise ValueError(
            "P2.1 currently has validated bearing kinematics only for CWRU. "
            "Add the target dataset's bearing geometry before enabling it."
        )
    if "model_config" not in report:
        raise ValueError(
            "The P1 report predates P2 metadata. Rerun P1 with the current code."
        )
    projector_path = p1_dir / "projected_text_bank.pt"
    if not projector_path.is_file():
        raise ValueError(
            "P1 text projector is missing. Rerun P1 with the current code."
        )

    domains = [int(value) for value in report["domains"]]
    class_names = tuple(str(value) for value in report["class_names"])
    model_config = report["model_config"]
    seed = int(report["training_config"]["seed"])
    _set_seed(seed)
    raw = build_domain_window_dataset(
        args.data_root,
        dataset=str(report["dataset"]),
        domains=domains,
        window_size=int(model_config["window_size"]),
        step_size=int(model_config["step_size"]),
        max_windows_per_file=int(model_config["max_windows_per_file"]),
        normalize=False,
    )
    train_by_domain, _, test_by_domain, _ = build_protocol_splits(
        raw,
        domains,
        seed=seed,
    )
    normalization = report["normalization"]
    stats = NormalizationStats(
        mean=float(normalization["mean"]),
        std=float(normalization["std"]),
        fitted_domain=int(normalization["fitted_domain"]),
        fitted_samples=int(normalization["fitted_samples"]),
    )
    normalized_train = {
        domain: apply_normalization(dataset, stats)
        for domain, dataset in train_by_domain.items()
    }
    normalized_test = {
        domain: apply_normalization(dataset, stats)
        for domain, dataset in test_by_domain.items()
    }

    specialist = SEGSCLSpecialist(
        input_channels=int(model_config["input_channels"]),
        token_dim=int(model_config["token_dim"]),
        num_tokens=int(model_config["num_tokens"]),
        num_domains=int(model_config["num_domains"]),
    )
    specialist.load_state_dict(_load_state(p1_dir / "specialist_final.pt"))
    specialist.to(device).eval().requires_grad_(False)

    global_cache = TextEmbeddingCache.load(args.global_text_cache)
    symptom_cache = SymptomEmbeddingCache.load(args.symptom_text_cache)
    if global_cache.class_names != class_names:
        raise ValueError("Global text cache class order does not match P1.")
    if symptom_cache.class_names != class_names:
        raise ValueError("Symptom text cache class order does not match P1.")
    if symptom_cache.model_id != global_cache.model_id:
        raise ValueError("Global and symptom caches must use the same text model.")
    projected_global = ProjectedTextPrototypeBank(
        global_cache,
        semantic_dim=int(model_config["token_dim"]),
    )
    projected_global.load_state_dict(_load_state(projector_path))
    projected_global.to(device).eval().requires_grad_(False)

    frozen_global = FrozenPrototypeBank(
        torch.zeros(len(class_names), int(model_config["token_dim"])),
        class_names,
        version="p1-loaded",
    )
    frozen_global.load_state_dict(_load_state(p1_dir / "frozen_prototypes.pt"))
    frozen_global.to(device)
    if args.physics_guided:
        local_projection = copy.deepcopy(projected_global.projection)
        local_projection.requires_grad_(True)
        symptom_bank = ProjectedSymptomPrototypeBank(
            symptom_cache,
            local_projection,
            projected_global.text_center,
        ).to(device)
        anchor_prototypes = symptom_bank().detach().clone()
    else:
        symptom_bank = ProjectedSymptomPrototypeBank(
            symptom_cache,
            projected_global.projection,
            projected_global.text_center,
        ).to(device).freeze("p2-local-symptoms").to(device)
        anchor_prototypes = symptom_bank().detach().clone()

    calibrator: RobustAttributeCalibrator | None = None
    physical_test_batches = {}
    if args.physics_guided:
        initial_raw_attributes = _extract_physics(
            normalized_train[domains[0]],
            symptom_cache.physics_keys,
        )
        calibrator = RobustAttributeCalibrator.fit(initial_raw_attributes)
        calibrator.save(output_dir / "physics_calibrator.json")

        train_sets = {}
        test_sets = {}
        for domain in domains:
            train_attributes = calibrator.transform(
                _extract_physics(
                    normalized_train[domain],
                    symptom_cache.physics_keys,
                )
            )
            train_targets, train_weights = build_symptom_soft_targets(
                train_attributes,
                np.asarray(normalized_train[domain]["y"], dtype=np.int64),
                symptom_cache.class_ids.numpy(),
            )
            train_sets[domain] = WindowDataset(
                normalized_train[domain],
                symptom_targets=train_targets,
                symptom_target_weights=train_weights,
            )
            test_attributes = calibrator.transform(
                _extract_physics(
                    normalized_test[domain],
                    symptom_cache.physics_keys,
                )
            )
            test_targets, test_weights = build_symptom_soft_targets(
                test_attributes,
                np.asarray(normalized_test[domain]["y"], dtype=np.int64),
                symptom_cache.class_ids.numpy(),
            )
            test_sets[domain] = WindowDataset(
                normalized_test[domain],
                symptom_targets=test_targets,
                symptom_target_weights=test_weights,
            )
            physical_test_batches[domain] = (
                test_targets,
                test_weights,
                test_attributes.values,
            )
    else:
        train_sets = {
            domain: WindowDataset(dataset)
            for domain, dataset in normalized_train.items()
        }
        test_sets = {
            domain: WindowDataset(dataset)
            for domain, dataset in normalized_test.items()
        }

    matcher = LocalSymptomMatcher(
        symptom_bank,
        top_tokens=args.top_tokens,
        temperature=args.local_temperature,
        learnable_symptom_weights=args.learnable_symptom_weights,
    ).to(device)

    history: list[dict[str, float | int]] = []
    if args.adapter_epochs > 0:
        optimizer = torch.optim.AdamW(
            matcher.parameters(),
            lr=args.learning_rate,
        )
        loader = DataLoader(
            train_sets[domains[0]],
            batch_size=args.batch_size,
            shuffle=True,
        )
        for epoch in range(args.adapter_epochs):
            matcher.train()
            total = 0.0
            component_totals = {
                "classification": 0.0,
                "physics": 0.0,
                "prototype_anchor": 0.0,
            }
            steps = 0
            for batch in loader:
                with torch.no_grad():
                    specialist_output = specialist(batch["x"].to(device))
                local_output = matcher(specialist_output.fault_tokens)
                if args.physics_guided:
                    loss, components = physics_guided_local_alignment_loss(
                        local_output,
                        batch["label"].to(device),
                        batch["symptom_targets"].to(device),
                        batch["symptom_target_weights"].to(device),
                        anchor_prototypes,
                        temperature=args.local_temperature,
                        physics_weight=args.physics_weight,
                        anchor_weight=args.anchor_weight,
                    )
                    for key, value in components.items():
                        component_totals[key] += float(value.detach().cpu())
                else:
                    loss = local_symptom_alignment_loss(
                        local_output,
                        batch["label"].to(device),
                        temperature=args.local_temperature,
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu())
                steps += 1
            history.append(
                {
                    "epoch": epoch,
                    "loss": total / max(1, steps),
                    **(
                        {
                            key: value / max(1, steps)
                            for key, value in component_totals.items()
                        }
                        if args.physics_guided
                        else {}
                    ),
                }
            )

    matcher.eval()
    domain_metrics: dict[str, dict[str, object]] = {}
    evaluation_rows: list[dict[str, object]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "labels": [],
        "domains": [],
        "sample_ids": [],
        "global_probabilities": [],
        "local_probabilities": [],
        "fused_probabilities": [],
        "symptom_joint_probabilities": [],
        "symptom_probabilities": [],
        "fuzzy_symptom_embeddings": [],
    }
    if args.physics_guided:
        arrays["physical_symptom_targets"] = []
        arrays["physical_target_weights"] = []
        arrays["physical_attributes"] = []
    with torch.inference_mode():
        for domain in domains:
            labels_rows: list[np.ndarray] = []
            global_rows: list[np.ndarray] = []
            local_rows: list[np.ndarray] = []
            joint_rows: list[np.ndarray] = []
            symptom_probability_rows: list[np.ndarray] = []
            fuzzy_rows: list[np.ndarray] = []
            sample_rows: list[np.ndarray] = []
            target_rows: list[np.ndarray] = []
            target_weight_rows: list[np.ndarray] = []
            for batch in DataLoader(
                test_sets[domain],
                batch_size=args.batch_size,
                shuffle=False,
            ):
                specialist_output = specialist(batch["x"].to(device))
                global_probabilities = torch.softmax(
                    frozen_global.similarities(
                        specialist_output.fault_embedding
                    ) / 0.07,
                    dim=1,
                )
                local_output = matcher(specialist_output.fault_tokens)
                labels_rows.append(batch["label"].numpy())
                sample_rows.append(batch["sample_id"].numpy())
                global_rows.append(global_probabilities.cpu().numpy())
                local_rows.append(local_output.class_probabilities.cpu().numpy())
                joint_rows.append(local_output.joint_probabilities.cpu().numpy())
                symptom_probability_rows.append(
                    torch.sigmoid(
                        local_output.symptom_scores / args.local_temperature
                    ).cpu().numpy()
                )
                fuzzy_rows.append(
                    local_output.fuzzy_symptom_embedding.cpu().numpy()
                )
                if args.physics_guided:
                    target_rows.append(batch["symptom_targets"].numpy())
                    target_weight_rows.append(
                        batch["symptom_target_weights"].numpy()
                    )
            labels = np.concatenate(labels_rows)
            sample_ids = np.concatenate(sample_rows)
            global_probabilities = np.concatenate(global_rows)
            local_probabilities = np.concatenate(local_rows)
            joint_probabilities = np.concatenate(joint_rows)
            symptom_probabilities = np.concatenate(
                symptom_probability_rows
            )
            fuzzy_embeddings = np.concatenate(fuzzy_rows)
            fused_probabilities = _fuse_probabilities(
                global_probabilities,
                local_probabilities,
                args.local_weight,
            )
            predictions = fused_probabilities.argmax(axis=1)
            domain_metrics[str(domain)] = {
                "global": _metrics(
                    labels,
                    global_probabilities.argmax(axis=1),
                    len(class_names),
                ),
                "local": _metrics(
                    labels,
                    local_probabilities.argmax(axis=1),
                    len(class_names),
                ),
                "fused": _metrics(labels, predictions, len(class_names)),
            }
            if args.physics_guided:
                physical_targets = np.concatenate(target_rows)
                physical_weights = np.concatenate(target_weight_rows)
                domain_metrics[str(domain)]["grounding"] = _grounding_metrics(
                    symptom_probabilities,
                    physical_targets,
                    physical_weights,
                    labels,
                    symptom_cache.class_ids.numpy(),
                )
            for index in range(len(labels)):
                packet = build_semantic_diagnostic_packet(
                    sample_id=int(sample_ids[index]),
                    domain_id=domain,
                    class_names=class_names,
                    symptom_names=symptom_bank.symptom_names,
                    symptom_class_ids=symptom_bank.class_ids.cpu().tolist(),
                    global_probabilities=global_probabilities[index],
                    local_probabilities=local_probabilities[index],
                    symptom_joint_probabilities=joint_probabilities[index],
                    local_weight=args.local_weight,
                    top_k=args.top_k,
                    top_symptoms=args.top_symptoms,
                )
                evaluation_rows.append(
                    {
                        **packet.to_dict(),
                        "ground_truth_class_id": int(labels[index]),
                        "ground_truth_class_name": class_names[
                            int(labels[index])
                        ],
                        "is_correct": bool(
                            packet.predicted_class_id == int(labels[index])
                        ),
                        **(
                            {
                                "physical_symptom_targets": physical_targets[
                                    index
                                ].tolist(),
                                "predicted_symptom_probabilities": (
                                    symptom_probabilities[index].tolist()
                                ),
                            }
                            if args.physics_guided
                            else {}
                        ),
                    }
                )
            arrays["labels"].append(labels)
            arrays["domains"].append(np.full(len(labels), domain, dtype=np.int64))
            arrays["sample_ids"].append(sample_ids)
            arrays["global_probabilities"].append(global_probabilities)
            arrays["local_probabilities"].append(local_probabilities)
            arrays["fused_probabilities"].append(fused_probabilities)
            arrays["symptom_joint_probabilities"].append(joint_probabilities)
            arrays["symptom_probabilities"].append(symptom_probabilities)
            arrays["fuzzy_symptom_embeddings"].append(fuzzy_embeddings)
            if args.physics_guided:
                arrays["physical_symptom_targets"].append(physical_targets)
                arrays["physical_target_weights"].append(physical_weights)
                arrays["physical_attributes"].append(
                    physical_test_batches[domain][2]
                )

    final_symptom_prototypes = matcher.bank().detach()
    np.savez_compressed(
        output_dir / "p2_outputs.npz",
        **{key: np.concatenate(value) for key, value in arrays.items()},
        symptom_prototypes=final_symptom_prototypes.cpu().numpy(),
        symptom_class_ids=symptom_bank.class_ids.cpu().numpy(),
    )
    with (output_dir / "evaluation_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in evaluation_rows:
            handle.write(json.dumps(row) + "\n")
    torch.save(matcher.state_dict(), output_dir / "local_symptom_matcher.pt")
    torch.save(
        {
            "prototypes": final_symptom_prototypes.cpu(),
            "class_ids": symptom_bank.class_ids.detach().cpu(),
        },
        output_dir / "frozen_symptom_prototypes.pt",
    )
    summary = {
        "status": "ok",
        "stage": (
            "P2.1 physics-guided local symptom alignment"
            if args.physics_guided
            else "P2 local symptom probe"
        ),
        "dataset": report["dataset"],
        "domains": domains,
        "class_names": class_names,
        "symptom_names": symptom_bank.symptom_names,
        "symptom_class_ids": symptom_bank.class_ids.cpu().tolist(),
        "p1_dir": str(p1_dir.resolve()),
        "adapter_trained_on_domain": domains[0],
        "adapter_epochs": args.adapter_epochs,
        "specialist_frozen": True,
        "global_text_projector_frozen": True,
        "local_symptom_projector_trainable": args.physics_guided,
        "physics_guided": args.physics_guided,
        "physics_keys": symptom_cache.physics_keys,
        "bearing_kinematics": (
            {
                "name": CWRU_DRIVE_END_KINEMATICS.name,
                "bpfi_ratio": CWRU_DRIVE_END_KINEMATICS.bpfi_ratio,
                "bpfo_ratio": CWRU_DRIVE_END_KINEMATICS.bpfo_ratio,
                "bsf_ratio": CWRU_DRIVE_END_KINEMATICS.bsf_ratio,
                "ftf_ratio": CWRU_DRIVE_END_KINEMATICS.ftf_ratio,
            }
            if args.physics_guided
            else None
        ),
        "model_config": {
            "top_tokens": args.top_tokens,
            "local_temperature": args.local_temperature,
            "local_weight": args.local_weight,
            "learnable_symptom_weights": args.learnable_symptom_weights,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "physics_weight": args.physics_weight,
            "anchor_weight": args.anchor_weight,
            "seed": seed,
        },
        "diagnostic_output": {
            "type": "hierarchical_semantic_prototype_classification",
            "global_branch": "fault identity prototypes",
            "local_branch": "token-to-symptom TopAvg matching",
            "local_supervision": (
                "class-gated physical soft attributes"
                if args.physics_guided
                else "fault class labels"
            ),
            "fusion": "weighted geometric probability fusion",
            "llm_text_generation_enabled": False,
        },
        "domain_metrics": domain_metrics,
        "history": history,
        "physics_calibration": (
            None if calibrator is None else calibrator.to_dict()
        ),
        "diagnostic_packet_schema": (
            "Top-k classes + global/local probabilities + uncertainty "
            "+ Top symptoms; ready for later continuous-prompt assembly."
        ),
        "note": (
            "The local semantic branch is trained only on the initial domain; "
            "the P1 specialist and global semantic branch remain frozen. "
            "No LLM text generation is used."
        ),
    }
    (output_dir / "p2_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
