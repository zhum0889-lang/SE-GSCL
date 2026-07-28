"""Evaluate P2 local symptom semantics on a frozen P1 specialist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from se_gscl.losses import local_symptom_alignment_loss  # noqa: E402
from se_gscl.models import LocalSymptomMatcher, SEGSCLSpecialist  # noqa: E402
from se_gscl.semantics import (  # noqa: E402
    FrozenPrototypeBank,
    ProjectedSymptomPrototypeBank,
    ProjectedTextPrototypeBank,
    SymptomEmbeddingCache,
    TextEmbeddingCache,
)


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, dataset: dict[str, object]) -> None:
        self.x = torch.from_numpy(np.asarray(dataset["x"], dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(dataset["y"], dtype=np.int64))
        self.domains = torch.from_numpy(
            np.asarray(dataset["domain_id"], dtype=np.int64)
        )
        self.sample_ids = torch.from_numpy(
            np.asarray(dataset["sample_id"], dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "label": self.labels[index],
            "domain": self.domains[index],
            "sample_id": self.sample_ids[index],
        }


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
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-symptoms", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.local_weight <= 1.0:
        raise ValueError("local_weight must be in [0,1].")
    device = torch.device(args.device)
    p1_dir = Path(args.p1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((p1_dir / "p1_report.json").read_text(encoding="utf-8"))
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
        seed=int(report["training_config"]["seed"]),
    )
    normalization = report["normalization"]
    stats = NormalizationStats(
        mean=float(normalization["mean"]),
        std=float(normalization["std"]),
        fitted_domain=int(normalization["fitted_domain"]),
        fitted_samples=int(normalization["fitted_samples"]),
    )
    train_sets = {
        domain: WindowDataset(apply_normalization(dataset, stats))
        for domain, dataset in train_by_domain.items()
    }
    test_sets = {
        domain: WindowDataset(apply_normalization(dataset, stats))
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
    symptom_bank = ProjectedSymptomPrototypeBank(
        symptom_cache,
        projected_global.projection,
        projected_global.text_center,
    ).to(device).freeze("p2-local-symptoms").to(device)
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
            steps = 0
            for batch in loader:
                with torch.no_grad():
                    specialist_output = specialist(batch["x"].to(device))
                local_output = matcher(specialist_output.fault_tokens)
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
        "fuzzy_symptom_embeddings": [],
    }
    with torch.inference_mode():
        for domain in domains:
            labels_rows: list[np.ndarray] = []
            global_rows: list[np.ndarray] = []
            local_rows: list[np.ndarray] = []
            joint_rows: list[np.ndarray] = []
            fuzzy_rows: list[np.ndarray] = []
            sample_rows: list[np.ndarray] = []
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
                fuzzy_rows.append(
                    local_output.fuzzy_symptom_embedding.cpu().numpy()
                )
            labels = np.concatenate(labels_rows)
            sample_ids = np.concatenate(sample_rows)
            global_probabilities = np.concatenate(global_rows)
            local_probabilities = np.concatenate(local_rows)
            joint_probabilities = np.concatenate(joint_rows)
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
                    }
                )
            arrays["labels"].append(labels)
            arrays["domains"].append(np.full(len(labels), domain, dtype=np.int64))
            arrays["sample_ids"].append(sample_ids)
            arrays["global_probabilities"].append(global_probabilities)
            arrays["local_probabilities"].append(local_probabilities)
            arrays["fused_probabilities"].append(fused_probabilities)
            arrays["symptom_joint_probabilities"].append(joint_probabilities)
            arrays["fuzzy_symptom_embeddings"].append(fuzzy_embeddings)

    np.savez_compressed(
        output_dir / "p2_outputs.npz",
        **{key: np.concatenate(value) for key, value in arrays.items()},
        symptom_prototypes=symptom_bank.prototypes.cpu().numpy(),
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
        symptom_bank.state_dict(),
        output_dir / "frozen_symptom_prototypes.pt",
    )
    summary = {
        "status": "ok",
        "stage": "P2 local symptom probe",
        "dataset": report["dataset"],
        "domains": domains,
        "class_names": class_names,
        "symptom_names": symptom_bank.symptom_names,
        "symptom_class_ids": symptom_bank.class_ids.cpu().tolist(),
        "p1_dir": str(p1_dir.resolve()),
        "adapter_trained_on_domain": domains[0],
        "adapter_epochs": args.adapter_epochs,
        "specialist_frozen": True,
        "text_projector_frozen": True,
        "model_config": {
            "top_tokens": args.top_tokens,
            "local_temperature": args.local_temperature,
            "local_weight": args.local_weight,
            "learnable_symptom_weights": args.learnable_symptom_weights,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "diagnostic_output": {
            "type": "hierarchical_semantic_prototype_classification",
            "global_branch": "fault identity prototypes",
            "local_branch": "token-to-symptom TopAvg matching",
            "fusion": "weighted geometric probability fusion",
            "llm_text_generation_enabled": False,
        },
        "domain_metrics": domain_metrics,
        "history": history,
        "diagnostic_packet_schema": (
            "Top-k classes + global/local probabilities + uncertainty "
            "+ Top symptoms; ready for later continuous-prompt assembly."
        ),
        "note": (
            "P2 mechanism probe. The local adapter is trained only on the "
            "initial domain; no LLM generation is used."
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
