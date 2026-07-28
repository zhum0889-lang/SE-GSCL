"""Run a two-domain P1 global-semantic continual-learning experiment."""

from __future__ import annotations

import argparse
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
from se_gscl.continual import ClassDomainBatchSampler, GlobalRelationSnapshot  # noqa: E402
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


def _as_window_dataset(dataset: dict[str, object]) -> WindowDataset:
    return WindowDataset(
        np.asarray(dataset["x"], dtype=np.float32),
        np.asarray(dataset["y"], dtype=np.int64),
        np.asarray(dataset["domain_id"], dtype=np.int64),
        np.asarray(dataset["sample_id"], dtype=np.int64),
    )


@torch.inference_mode()
def _predict(
    model: SEGSCLSpecialist,
    prototypes: torch.Tensor,
    dataset: WindowDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        output = model(batch["x"].to(device))
        _, logits = global_prototype_alignment_loss(
            output.fault_embedding,
            prototypes.to(device),
            batch["label"].to(device),
        )
        predictions.append(logits.argmax(dim=1).cpu())
        probabilities.append(snapshot_probabilities(logits).cpu())
    return (
        torch.cat(predictions).numpy(),
        torch.cat(probabilities).numpy(),
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


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    domains = [int(value) for value in args.domains.split(",") if value.strip()]
    if len(domains) != 2:
        raise ValueError("P1 smoke currently expects exactly two ordered domains.")
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
    initial_labels = initial_train.labels.numpy()
    class_counts = np.bincount(initial_labels, minlength=len(class_names))
    sample_weights = 1.0 / class_counts[initial_labels]
    initial_sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(initial_train),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    initial_loader = DataLoader(
        initial_train,
        batch_size=args.batch_size,
        sampler=initial_sampler,
    )
    for epoch in range(args.initial_epochs):
        metrics = initial_trainer.train_epoch(initial_loader, initial_optimizer)
        history.append({"stage": "initial", "epoch": epoch, **metrics})

    frozen_bank = projected_bank.freeze("p1-after-initial-domain").to(device)
    torch.save(frozen_bank.state_dict(), output_dir / "frozen_prototypes.pt")
    initial_metrics = {
        str(domain): _accuracy(
            model,
            frozen_bank.prototypes,
            _as_window_dataset(test_by_domain[domain]),
            device,
            args.batch_size,
        )
        for domain in domains
    }
    replay_indices = _select_replay_indices(
        initial_train.labels.numpy(),
        args.replay_per_class,
        args.seed,
    )
    replay = WindowDataset(
        initial_train.x[replay_indices].numpy(),
        initial_train.labels[replay_indices].numpy(),
        initial_train.domains[replay_indices].numpy(),
        initial_train.sample_ids[replay_indices].numpy(),
    )
    _, old_probs = _predict(
        model,
        frozen_bank.prototypes,
        replay,
        device,
        args.batch_size,
    )
    snapshot = GlobalRelationSnapshot(
        replay.sample_ids,
        torch.from_numpy(old_probs),
        version="before-domain-1",
    )
    snapshot.save(output_dir / "global_relation_snapshot.npz")

    current = _as_window_dataset(train_by_domain[domains[1]])
    combined_x = torch.cat([current.x, replay.x]).numpy()
    combined_labels = torch.cat([current.labels, replay.labels]).numpy()
    combined_domains = torch.cat([current.domains, replay.domains]).numpy()
    combined_ids = torch.cat([current.sample_ids, replay.sample_ids]).numpy()
    is_replay = np.concatenate(
        [
            np.zeros(len(current), dtype=bool),
            np.ones(len(replay), dtype=bool),
        ]
    )
    snapshot_rows = np.full(
        (len(current) + len(replay), len(class_names)),
        1.0 / len(class_names),
        dtype=np.float32,
    )
    snapshot_rows[len(current) :] = old_probs
    continual_train = WindowDataset(
        combined_x,
        combined_labels,
        combined_domains,
        combined_ids,
        snapshot_probs=snapshot_rows,
        is_replay=is_replay,
    )
    batch_sampler = ClassDomainBatchSampler(
        combined_labels,
        combined_domains,
        args.batch_size,
        seed=args.seed,
    )
    continual_loader = DataLoader(continual_train, batch_sampler=batch_sampler)
    continual_trainer = GlobalSemanticTrainer(
        model,
        frozen_bank.prototypes,
        P1LossWeights(
            global_alignment=1.0,
            cross_condition=args.lambda_cc,
            decorrelation=args.lambda_dec,
            global_relation=args.lambda_rel,
        ),
        device=device,
    )
    continual_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    for epoch in range(args.continual_epochs):
        batch_sampler.set_epoch(epoch)
        metrics = continual_trainer.train_epoch(
            continual_loader,
            continual_optimizer,
        )
        history.append({"stage": "continual", "epoch": epoch, **metrics})

    test_sets = {
        domain: _as_window_dataset(dataset)
        for domain, dataset in test_by_domain.items()
    }
    final_metrics = {
        str(domain): _accuracy(
            model,
            frozen_bank.prototypes,
            test_sets[domain],
            device,
            args.batch_size,
        )
        for domain in domains
    }
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
        "domains": domains,
        "class_names": class_names,
        "text_model": text_cache.model_id,
        "text_hidden_size": text_cache.hidden_size,
        "text_pooling": text_cache.pooling,
        "text_centering": "ontology_global_mean",
        "semantic_dim": args.semantic_dim,
        "normalization": stats.to_dict(),
        "replay_samples": len(replay),
        "initial_stage_metrics": initial_metrics,
        "final_stage_metrics": final_metrics,
        "old_domain_forgetting": forgetting,
        "accuracies": {
            domain: values["accuracy"]
            for domain, values in final_metrics.items()
        },
        "history": history,
        "note": "Smoke result for pipeline validation; not a paper benchmark.",
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
