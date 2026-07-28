"""Run ordered P1 global-semantic continual learning over two or more domains."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.continual_fdllm.domain_windows import (  # noqa: E402
    apply_normalization,
    build_domain_window_dataset,
    build_protocol_splits,
    fit_normalization,
)
from se_gscl.continual import (  # noqa: E402
    ClassDomainBatchSampler,
    GlobalRelationSnapshot,
    summarize_accuracy_matrix,
)
from se_gscl.losses import global_prototype_alignment_loss, snapshot_probabilities  # noqa: E402
from se_gscl.models import SEGSCLSpecialist  # noqa: E402
from se_gscl.semantics import (  # noqa: E402
    FrozenPrototypeBank,
    ProjectedTextPrototypeBank,
    TextEmbeddingCache,
)
from se_gscl.training import GlobalSemanticTrainer, P1LossWeights  # noqa: E402


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
        sample_ids: np.ndarray,
        *,
        snapshot_probs: np.ndarray | None = None,
        is_replay: np.ndarray | None = None,
    ) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.domains = torch.from_numpy(np.asarray(domains, dtype=np.int64))
        self.sample_ids = torch.from_numpy(np.asarray(sample_ids, dtype=np.int64))
        self.snapshot_probs = (
            None
            if snapshot_probs is None
            else torch.from_numpy(np.asarray(snapshot_probs, dtype=np.float32))
        )
        self.is_replay = (
            None
            if is_replay is None
            else torch.from_numpy(np.asarray(is_replay, dtype=bool))
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = {
            "x": self.x[index],
            "label": self.labels[index],
            "domain": self.domains[index],
            "sample_id": self.sample_ids[index],
        }
        if self.snapshot_probs is not None:
            row["snapshot_probs"] = self.snapshot_probs[index]
        if self.is_replay is not None:
            row["is_replay"] = self.is_replay[index]
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="cwru4")
    parser.add_argument("--domains", default="0,1")
    parser.add_argument(
        "--strategy",
        choices=("sequential", "balanced_replay", "full"),
        default="full",
    )
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--step-size", type=int, default=1024)
    parser.add_argument("--max-windows-per-file", type=int, default=24)
    parser.add_argument("--semantic-dim", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--initial-epochs", type=int, default=2)
    parser.add_argument("--continual-epochs", type=int, default=2)
    parser.add_argument("--replay-per-class", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-cc", type=float, default=0.1)
    parser.add_argument("--lambda-dec", type=float, default=0.01)
    parser.add_argument("--lambda-rel", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _as_window_dataset(dataset: dict[str, object]) -> WindowDataset:
    return WindowDataset(
        np.asarray(dataset["x"], dtype=np.float32),
        np.asarray(dataset["y"], dtype=np.int64),
        np.asarray(dataset["domain_id"], dtype=np.int64),
        np.asarray(dataset["sample_id"], dtype=np.int64),
    )


@torch.inference_mode()
def _collect_outputs(
    model: SEGSCLSpecialist,
    prototypes: torch.Tensor,
    dataset: WindowDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        output = model(batch["x"].to(device))
        _, logits = global_prototype_alignment_loss(
            output.fault_embedding,
            prototypes.to(device),
            batch["label"].to(device),
        )
        predictions.append(logits.argmax(dim=1).cpu())
        probabilities.append(snapshot_probabilities(logits).cpu())
        embeddings.append(output.fault_embedding.cpu())
    return (
        torch.cat(predictions).numpy(),
        torch.cat(probabilities).numpy(),
        torch.cat(embeddings).numpy(),
    )


def _predict(
    model: SEGSCLSpecialist,
    prototypes: torch.Tensor,
    dataset: WindowDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions, probabilities, _ = _collect_outputs(
        model,
        prototypes,
        dataset,
        device,
        batch_size,
    )
    return predictions, probabilities


def _save_stage_outputs(
    output_dir: Path,
    trained_domain: int,
    model: SEGSCLSpecialist,
    prototypes: torch.Tensor,
    test_sets: dict[int, WindowDataset],
    domains: list[int],
    class_names: tuple[str, ...],
    device: torch.device,
    batch_size: int,
) -> None:
    """Persist sample-level outputs needed for paper-facing diagnostics."""

    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    sample_domains: list[np.ndarray] = []
    for domain in domains:
        dataset = test_sets[domain]
        pred, prob, embedding = _collect_outputs(
            model,
            prototypes,
            dataset,
            device,
            batch_size,
        )
        predictions.append(pred)
        probabilities.append(prob)
        embeddings.append(embedding)
        labels.append(dataset.labels.numpy())
        sample_ids.append(dataset.sample_ids.numpy())
        sample_domains.append(dataset.domains.numpy())
    np.savez_compressed(
        output_dir / f"stage_outputs_after_domain_{trained_domain}.npz",
        predictions=np.concatenate(predictions),
        probabilities=np.concatenate(probabilities),
        embeddings=np.concatenate(embeddings),
        labels=np.concatenate(labels),
        sample_ids=np.concatenate(sample_ids),
        domains=np.concatenate(sample_domains),
        prototypes=prototypes.detach().cpu().numpy(),
        class_names=np.asarray(class_names, dtype="U"),
        trained_domain=np.asarray([trained_domain], dtype=np.int64),
    )


def _accuracy(
    model: SEGSCLSpecialist,
    prototypes: torch.Tensor,
    dataset: WindowDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    predictions, _ = _predict(model, prototypes, dataset, device, batch_size)
    labels = dataset.labels.numpy()
    per_class = {
        str(label): float(np.mean(predictions[labels == label] == label))
        for label in sorted(np.unique(labels).tolist())
    }
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "per_class_accuracy": per_class,
    }


def _select_replay_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        candidates = np.flatnonzero(labels == int(label))
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _balanced_loader(
    dataset: WindowDataset,
    batch_size: int,
    seed: int,
) -> DataLoader:
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels)
    sample_weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _concatenate_datasets(*datasets: WindowDataset) -> WindowDataset:
    if not datasets:
        raise ValueError("At least one WindowDataset is required.")
    return WindowDataset(
        torch.cat([dataset.x for dataset in datasets]).numpy(),
        torch.cat([dataset.labels for dataset in datasets]).numpy(),
        torch.cat([dataset.domains for dataset in datasets]).numpy(),
        torch.cat([dataset.sample_ids for dataset in datasets]).numpy(),
    )


def _select_balanced_memory(
    candidates: WindowDataset,
    per_class: int,
    seed: int,
) -> WindowDataset:
    """Keep fixed per-class capacity while covering every seen domain."""

    rng = np.random.default_rng(seed)
    labels = candidates.labels.numpy()
    domains = candidates.domains.numpy()
    selected: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        class_indices = np.flatnonzero(labels == label)
        class_domains = sorted(np.unique(domains[class_indices]).tolist())
        base = per_class // len(class_domains)
        extra = per_class % len(class_domains)
        class_selected: list[int] = []
        for position, domain in enumerate(class_domains):
            domain_indices = class_indices[domains[class_indices] == domain]
            rng.shuffle(domain_indices)
            quota = base + int(position < extra)
            class_selected.extend(domain_indices[:quota].tolist())
        if len(class_selected) < per_class:
            remaining = class_indices[
                ~np.isin(class_indices, np.asarray(class_selected, dtype=np.int64))
            ]
            rng.shuffle(remaining)
            class_selected.extend(remaining[: per_class - len(class_selected)].tolist())
        selected.extend(class_selected[:per_class])
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    return WindowDataset(
        candidates.x[selected_array].numpy(),
        candidates.labels[selected_array].numpy(),
        candidates.domains[selected_array].numpy(),
        candidates.sample_ids[selected_array].numpy(),
    )


def _evaluate_all_domains(
    model: SEGSCLSpecialist,
    prototypes: torch.Tensor,
    test_sets: dict[int, WindowDataset],
    domains: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, object]]:
    return {
        str(domain): _accuracy(
            model,
            prototypes,
            test_sets[domain],
            device,
            batch_size,
        )
        for domain in domains
    }


def _write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    domains: list[int],
    *,
    seen_only: bool,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage"] + [f"domain_{domain}" for domain in domains])
        for stage_index, row in enumerate(matrix):
            values: list[str | float] = []
            for domain_index, value in enumerate(row):
                if seen_only and domain_index > stage_index:
                    values.append("")
                else:
                    values.append(float(value))
            writer.writerow([f"after_domain_{domains[stage_index]}"] + values)


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    domains = [int(value) for value in args.domains.split(",") if value.strip()]
    if len(domains) < 2:
        raise ValueError("P1 continual learning requires at least two domains.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    raw = build_domain_window_dataset(
        args.data_root,
        dataset=args.dataset,
        domains=domains,
        window_size=args.window_size,
        step_size=args.step_size,
        max_windows_per_file=args.max_windows_per_file,
        normalize=False,
    )
    train_by_domain, _, test_by_domain, _ = build_protocol_splits(
        raw,
        domains,
        seed=args.seed,
    )
    stats = fit_normalization(train_by_domain[domains[0]], domains[0])
    train_by_domain = {
        domain: apply_normalization(dataset, stats)
        for domain, dataset in train_by_domain.items()
    }
    test_by_domain = {
        domain: apply_normalization(dataset, stats)
        for domain, dataset in test_by_domain.items()
    }

    text_cache = TextEmbeddingCache.load(args.text_cache)
    class_names = tuple(str(value) for value in raw["class_names"])
    if text_cache.class_names != class_names:
        raise ValueError(
            f"Text cache class order {text_cache.class_names} does not match "
            f"dataset order {class_names}."
        )
    initial_train = _as_window_dataset(train_by_domain[domains[0]])
    input_channels = 1 if initial_train.x.ndim == 2 else int(initial_train.x.shape[1])
    model = SEGSCLSpecialist(
        input_channels=input_channels,
        token_dim=args.semantic_dim,
        num_tokens=args.num_tokens,
        num_domains=len(domains),
    ).to(device)
    projected_bank = ProjectedTextPrototypeBank(
        text_cache,
        semantic_dim=args.semantic_dim,
    ).to(device)
    initial_trainer = GlobalSemanticTrainer(
        model,
        projected_bank,
        P1LossWeights(
            global_alignment=1.0,
            decorrelation=args.lambda_dec,
        ),
        device=device,
    )
    initial_optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projected_bank.parameters()),
        lr=args.learning_rate,
    )
    history: list[dict[str, float | int | str]] = []
    initial_loader = _balanced_loader(
        initial_train,
        args.batch_size,
        args.seed,
    )
    for epoch in range(args.initial_epochs):
        metrics = initial_trainer.train_epoch(initial_loader, initial_optimizer)
        history.append(
            {
                "stage": "initial",
                "stage_index": 0,
                "domain": domains[0],
                "epoch": epoch,
                **metrics,
            }
        )

    frozen_bank = projected_bank.freeze("p1-after-initial-domain").to(device)
    torch.save(frozen_bank.state_dict(), output_dir / "frozen_prototypes.pt")
    test_sets = {
        domain: _as_window_dataset(dataset)
        for domain, dataset in test_by_domain.items()
    }
    initial_metrics = _evaluate_all_domains(
        model,
        frozen_bank.prototypes,
        test_sets,
        domains,
        device,
        args.batch_size,
    )
    stage_metrics: list[dict[str, object]] = [
        {
            "stage_index": 0,
            "trained_domain": domains[0],
            "domain_metrics": initial_metrics,
        }
    ]
    _save_stage_outputs(
        output_dir,
        domains[0],
        model,
        frozen_bank.prototypes,
        test_sets,
        domains,
        class_names,
        device,
        args.batch_size,
    )
    torch.save(model.state_dict(), output_dir / f"specialist_after_domain_{domains[0]}.pt")

    memory = _select_balanced_memory(
        initial_train,
        args.replay_per_class,
        args.seed,
    ) if args.strategy != "sequential" else None

    for stage_index, domain in enumerate(domains[1:], start=1):
        current = _as_window_dataset(train_by_domain[domain])
        relation_enabled = args.strategy == "full"
        if args.strategy == "sequential":
            continual_train = current
            continual_loader = _balanced_loader(
                continual_train,
                args.batch_size,
                args.seed + stage_index,
            )
            batch_sampler = None
        else:
            if memory is None:
                raise RuntimeError("Replay strategy requires initialized memory.")
            combined_x = torch.cat([current.x, memory.x]).numpy()
            combined_labels = torch.cat([current.labels, memory.labels]).numpy()
            combined_domains = torch.cat([current.domains, memory.domains]).numpy()
            combined_ids = torch.cat([current.sample_ids, memory.sample_ids]).numpy()
            if relation_enabled:
                _, old_probs = _predict(
                    model,
                    frozen_bank.prototypes,
                    memory,
                    device,
                    args.batch_size,
                )
                snapshot = GlobalRelationSnapshot(
                    memory.sample_ids,
                    torch.from_numpy(old_probs),
                    version=f"before-domain-{domain}",
                )
                snapshot.save(
                    output_dir / f"global_relation_snapshot_before_{domain}.npz"
                )
                snapshot_rows = np.full(
                    (len(current) + len(memory), len(class_names)),
                    1.0 / len(class_names),
                    dtype=np.float32,
                )
                snapshot_rows[len(current) :] = old_probs
                is_replay = np.concatenate(
                    [
                        np.zeros(len(current), dtype=bool),
                        np.ones(len(memory), dtype=bool),
                    ]
                )
                continual_train = WindowDataset(
                    combined_x,
                    combined_labels,
                    combined_domains,
                    combined_ids,
                    snapshot_probs=snapshot_rows,
                    is_replay=is_replay,
                )
            else:
                continual_train = WindowDataset(
                    combined_x,
                    combined_labels,
                    combined_domains,
                    combined_ids,
                )
            batch_sampler = ClassDomainBatchSampler(
                combined_labels,
                combined_domains,
                args.batch_size,
                seed=args.seed + stage_index * 100,
            )
            continual_loader = DataLoader(
                continual_train,
                batch_sampler=batch_sampler,
            )

        continual_trainer = GlobalSemanticTrainer(
            model,
            frozen_bank.prototypes,
            P1LossWeights(
                global_alignment=1.0,
                cross_condition=args.lambda_cc if relation_enabled else 0.0,
                decorrelation=args.lambda_dec,
                global_relation=args.lambda_rel if relation_enabled else 0.0,
            ),
            device=device,
        )
        continual_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
        )
        for epoch in range(args.continual_epochs):
            if batch_sampler is not None:
                batch_sampler.set_epoch(epoch)
            metrics = continual_trainer.train_epoch(
                continual_loader,
                continual_optimizer,
            )
            history.append(
                {
                    "stage": "continual",
                    "stage_index": stage_index,
                    "domain": domain,
                    "epoch": epoch,
                    **metrics,
                }
            )

        torch.save(
            model.state_dict(),
            output_dir / f"specialist_after_domain_{domain}.pt",
        )
        domain_metrics = _evaluate_all_domains(
            model,
            frozen_bank.prototypes,
            test_sets,
            domains,
            device,
            args.batch_size,
        )
        stage_metrics.append(
            {
                "stage_index": stage_index,
                "trained_domain": domain,
                "domain_metrics": domain_metrics,
            }
        )
        _save_stage_outputs(
            output_dir,
            domain,
            model,
            frozen_bank.prototypes,
            test_sets,
            domains,
            class_names,
            device,
            args.batch_size,
        )
        if memory is not None:
            memory_candidates = _concatenate_datasets(memory, current)
            memory = _select_balanced_memory(
                memory_candidates,
                args.replay_per_class,
                args.seed + stage_index * 1009,
            )

    final_metrics = stage_metrics[-1]["domain_metrics"]
    accuracy_matrix = np.asarray(
        [
            [
                row["domain_metrics"][str(domain)]["accuracy"]
                for domain in domains
            ]
            for row in stage_metrics
        ],
        dtype=np.float64,
    )
    balanced_matrix = np.asarray(
        [
            [
                row["domain_metrics"][str(domain)]["balanced_accuracy"]
                for domain in domains
            ]
            for row in stage_metrics
        ],
        dtype=np.float64,
    )
    sequence_summary = {
        "accuracy": summarize_accuracy_matrix(accuracy_matrix),
        "balanced_accuracy": summarize_accuracy_matrix(balanced_matrix),
    }
    _write_matrix_csv(
        output_dir / "accuracy_matrix_full.csv",
        accuracy_matrix,
        domains,
        seen_only=False,
    )
    _write_matrix_csv(
        output_dir / "accuracy_matrix_seen_only.csv",
        accuracy_matrix,
        domains,
        seen_only=True,
    )
    _write_matrix_csv(
        output_dir / "balanced_accuracy_matrix_seen_only.csv",
        balanced_matrix,
        domains,
        seen_only=True,
    )

    old_domain = str(domains[0])
    forgetting = {
        "accuracy": float(
            initial_metrics[old_domain]["accuracy"]
            - final_metrics[old_domain]["accuracy"]
        ),
        "balanced_accuracy": float(
            initial_metrics[old_domain]["balanced_accuracy"]
            - final_metrics[old_domain]["balanced_accuracy"]
        ),
    }
    report = {
        "status": "ok",
        "dataset": args.dataset,
        "strategy": args.strategy,
        "domains": domains,
        "class_names": class_names,
        "text_model": text_cache.model_id,
        "text_hidden_size": text_cache.hidden_size,
        "text_pooling": text_cache.pooling,
        "text_centering": "ontology_global_mean",
        "semantic_dim": args.semantic_dim,
        "diagnostic_output": {
            "type": "semantic_prototype_classification",
            "producer": "lightweight_specialist",
            "decision_rule": "argmax cosine similarity to frozen text prototypes",
            "llm_text_generation_enabled": False,
        },
        "training_config": {
            "initial_epochs": args.initial_epochs,
            "continual_epochs": args.continual_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lambda_cross_condition": args.lambda_cc,
            "lambda_decorrelation": args.lambda_dec,
            "lambda_global_relation": args.lambda_rel,
            "seed": args.seed,
        },
        "normalization": stats.to_dict(),
        "memory_capacity_per_class": args.replay_per_class
        if memory is not None
        else 0,
        "replay_samples": len(memory) if memory is not None else 0,
        "initial_stage_metrics": initial_metrics,
        "final_stage_metrics": final_metrics,
        "old_domain_forgetting": forgetting,
        "accuracies": {
            domain: values["accuracy"]
            for domain, values in final_metrics.items()
        },
        "stage_metrics": stage_metrics,
        "sequence_summary": sequence_summary,
        "history": history,
        "note": "Sequence result requires multi-seed baseline comparison before paper use.",
    }
    (output_dir / "p1_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), output_dir / "specialist_final.pt")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
