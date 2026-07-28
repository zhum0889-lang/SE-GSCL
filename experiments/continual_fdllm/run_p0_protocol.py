"""P0 continual-learning experiment with an auditable, leakage-resistant protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in [ROOT / "src", ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.continual_fdllm.continual_metrics import compute_accuracy, compute_macro_f1  # noqa: E402
from experiments.continual_fdllm.domain_windows import (  # noqa: E402
    NormalizationStats,
    apply_normalization,
    assert_no_window_leakage,
    build_domain_window_dataset,
    build_protocol_splits,
    fit_normalization,
    merge_datasets,
    stratified_limit,
    subset_by_sample_ids,
)
from experiments.continual_fdllm.fse_replay_selector import select_replay_samples  # noqa: E402
from experiments.continual_fdllm.protocol import (  # noqa: E402
    assert_seen_only_matrix,
    compute_sequential_metrics,
    update_persistent_memory,
)
from experiments.continual_fdllm.replay_buffer import ReplayBuffer  # noqa: E402
from fdllm_repro.models import (  # noqa: E402
    ConvLSTMSignalEncoder,
    encode_text_descriptions,
    fuzzy_semantic_predict,
    train_alignment,
)
from fdllm_repro.text import build_class_descriptions  # noqa: E402
from se_gscl.data import build_manifest_rows, write_manifest_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--dataset",
        choices=("cwru4", "cwru10", "cwru19", "paderborn", "hustbearing"),
        default="cwru4",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "p0_protocol"))
    parser.add_argument("--domain-order", default="0,1,2,3")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--strategies", default="no_replay,random_replay,balanced_semantic_replay")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--max-windows-per-file", type=int)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--conv-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--predict-temperature", type=float, default=0.5)
    parser.add_argument("--memory-ratio", type=float, default=0.10)
    parser.add_argument("--memory-max", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domain_order = _parse_int_list(args.domain_order)
    seeds = _parse_int_list(args.seeds)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    allowed = {"no_replay", "random_replay", "balanced_semantic_replay"}
    unknown = sorted(set(strategies) - allowed)
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")
    if args.smoke:
        seeds = seeds[:1]
        args.epochs = 1
        args.max_samples_per_split = args.max_samples_per_split or 96

    run_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", vars(args) | {"resolved_domain_order": domain_order, "resolved_seeds": seeds})

    all_metrics: list[dict[str, object]] = []
    for seed in seeds:
        all_metrics.extend(run_seed(args, domain_order, strategies, seed, run_dir))
    _write_csv(run_dir / "metrics_by_seed.csv", all_metrics)
    _write_csv(run_dir / "metrics_summary.csv", _aggregate_metrics(all_metrics))
    print(f"P0 outputs: {run_dir}")
    return 0


def run_seed(
    args: argparse.Namespace,
    domain_order: list[int],
    strategies: list[str],
    seed: int,
    run_dir: Path,
) -> list[dict[str, object]]:
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    _set_seed(seed)
    device = _resolve_device(args.device)

    raw_dataset = build_domain_window_dataset(
        args.data_root,
        dataset=args.dataset,
        domains=domain_order,
        window_size=args.window_size,
        step_size=args.step_size,
        max_windows_per_file=args.max_windows_per_file,
        normalize=False,
        seed=seed,
    )
    manifest_rows = build_manifest_rows(list(raw_dataset["records"]))  # type: ignore[arg-type]
    write_manifest_bundle(manifest_rows, seed_dir)
    train_by_domain, val_by_domain, test_by_domain, split_audit = build_protocol_splits(
        raw_dataset,
        domain_order,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=seed,
    )
    for domain in domain_order:
        assert_no_window_leakage(
            train_by_domain[domain],
            val_by_domain[domain],
            test_by_domain[domain],
        )

    initial_domain = domain_order[0]
    norm_stats = fit_normalization(train_by_domain[initial_domain], fitted_domain=initial_domain)
    normalized_full = apply_normalization(raw_dataset, norm_stats)
    for split_map in [train_by_domain, val_by_domain, test_by_domain]:
        for domain in domain_order:
            split_map[domain] = apply_normalization(split_map[domain], norm_stats)
            if args.max_samples_per_split > 0:
                split_map[domain] = stratified_limit(
                    split_map[domain],
                    args.max_samples_per_split,
                    seed=seed + domain * 101 + len(split_map),
                )

    _write_json(seed_dir / "normalization.json", norm_stats.to_dict())
    _write_csv(seed_dir / "split_audit.csv", [row.__dict__ for row in split_audit])
    _write_csv(seed_dir / "split_counts.csv", _split_counts(train_by_domain, val_by_domain, test_by_domain))

    class_names = list(raw_dataset["class_names"])  # type: ignore[arg-type]
    descriptions = build_class_descriptions(list(raw_dataset["records"]), class_names)  # type: ignore[arg-type]
    text_embeddings = encode_text_descriptions(descriptions, embed_dim=args.embed_dim)
    np.save(seed_dir / "frozen_text_prototypes.npy", text_embeddings)
    _write_json(
        seed_dir / "text_prototypes.json",
        {
            "encoder": "deterministic_hashing_placeholder",
            "warning": "Replace with frozen Qwen text hidden states for formal experiments.",
            "class_names": class_names,
            "descriptions": descriptions,
        },
    )

    metrics: list[dict[str, object]] = []
    for strategy in strategies:
        metrics.append(
            run_strategy(
                args,
                domain_order,
                seed,
                strategy,
                train_by_domain,
                test_by_domain,
                normalized_full,
                text_embeddings,
                class_names,
                device,
                seed_dir,
                norm_stats,
            )
        )
    return metrics


def run_strategy(
    args: argparse.Namespace,
    domain_order: list[int],
    seed: int,
    strategy: str,
    train_by_domain: dict[int, dict[str, object]],
    test_by_domain: dict[int, dict[str, object]],
    normalized_full: dict[str, object],
    text_embeddings: np.ndarray,
    class_names: list[str],
    device: str,
    seed_dir: Path,
    norm_stats: NormalizationStats,
) -> dict[str, object]:
    strategy_dir = seed_dir / strategy
    (strategy_dir / "memory").mkdir(parents=True, exist_ok=False)
    _set_seed(seed)
    train_shape = np.asarray(train_by_domain[domain_order[0]]["x"]).shape
    input_channels = int(train_shape[1]) if len(train_shape) == 3 else 1
    model = ConvLSTMSignalEncoder(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        conv_dim=args.conv_dim,
        input_channels=input_channels,
    )
    memory = ReplayBuffer(capacity=0, version=0)
    matrix = np.full((len(domain_order), len(domain_order)), np.nan, dtype=np.float32)
    episode_rows: list[dict[str, object]] = []
    total_seen_train = 0

    for episode, domain in enumerate(domain_order):
        current_train = train_by_domain[domain]
        replay_ds = None
        if strategy != "no_replay" and memory.records:
            replay_ds = subset_by_sample_ids(normalized_full, memory.sample_ids())
        train_ds = current_train if replay_ds is None else merge_datasets(current_train, replay_ds)
        history = train_alignment(
            model,
            np.asarray(train_ds["x"], dtype=np.float32),
            np.asarray(train_ds["y"], dtype=np.int64),
            text_embeddings,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            temperature=args.temperature,
        )

        for eval_idx in range(episode + 1):
            eval_domain = domain_order[eval_idx]
            acc, macro_f1 = _evaluate(
                model,
                test_by_domain[eval_domain],
                text_embeddings,
                device,
                args,
            )
            matrix[episode, eval_idx] = acc
            episode_rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "episode": episode,
                    "trained_domain": domain,
                    "eval_domain": eval_domain,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "train_samples": len(np.asarray(current_train["y"])),
                    "replay_samples": 0 if replay_ds is None else len(np.asarray(replay_ds["y"])),
                    "memory_size_before_update": len(memory.records),
                    "last_train_loss": history[-1].loss,
                }
            )

        total_seen_train += len(np.asarray(current_train["y"]))
        if strategy != "no_replay":
            probs, _, _ = fuzzy_semantic_predict(
                model,
                np.asarray(current_train["x"], dtype=np.float32),
                text_embeddings,
                device=device,
                temperature=args.predict_temperature,
                batch_size=args.batch_size,
            )
            preds = probs.argmax(axis=1)
            _, current_candidates = select_replay_samples(
                current_train,
                probs,
                preds,
                n=len(np.asarray(current_train["y"])),
                seed=seed + episode,
                prototype_version=f"seed{seed}-d{domain}",
            )
            capacity = min(args.memory_max, max(1, int(round(total_seen_train * args.memory_ratio))))
            memory = update_persistent_memory(
                memory,
                current_candidates,
                capacity=capacity,
                episode=episode,
                seed=seed,
                strategy=strategy,
            )
            memory.save(strategy_dir / "memory" / f"memory_after_episode_{episode}.json")

    assert_seen_only_matrix(matrix)
    _write_accuracy_matrix(strategy_dir / "accuracy_matrix.csv", domain_order, matrix)
    _write_csv(strategy_dir / "episode_metrics.csv", episode_rows)
    final_predictions = _final_prediction_rows(
        model,
        test_by_domain,
        domain_order,
        text_embeddings,
        class_names,
        device,
        args,
    )
    _write_csv(strategy_dir / "final_predictions.csv", final_predictions)
    _write_csv(
        strategy_dir / "final_confusion_matrix.csv",
        _confusion_matrix_rows(final_predictions, class_names),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "text_embeddings": text_embeddings,
            "normalization": norm_stats.to_dict(),
            "domain_order": domain_order,
            "seed": seed,
            "strategy": strategy,
            "dataset": args.dataset,
            "input_channels": input_channels,
        },
        strategy_dir / "final_model.pt",
    )
    return compute_sequential_metrics(strategy, domain_order, matrix, seed)


def _evaluate(
    model: ConvLSTMSignalEncoder,
    dataset: dict[str, object],
    text_embeddings: np.ndarray,
    device: str,
    args: argparse.Namespace,
) -> tuple[float, float]:
    probs, _, _ = fuzzy_semantic_predict(
        model,
        np.asarray(dataset["x"], dtype=np.float32),
        text_embeddings,
        device=device,
        temperature=args.predict_temperature,
        batch_size=args.batch_size,
    )
    y = np.asarray(dataset["y"], dtype=np.int64)
    pred = probs.argmax(axis=1)
    return compute_accuracy(y, pred), compute_macro_f1(y, pred)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _final_prediction_rows(
    model: ConvLSTMSignalEncoder,
    test_by_domain: dict[int, dict[str, object]],
    domain_order: list[int],
    text_embeddings: np.ndarray,
    class_names: list[str],
    device: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain in domain_order:
        dataset = test_by_domain[domain]
        probs, _, _ = fuzzy_semantic_predict(
            model,
            np.asarray(dataset["x"], dtype=np.float32),
            text_embeddings,
            device=device,
            temperature=args.predict_temperature,
            batch_size=args.batch_size,
        )
        labels = np.asarray(dataset["y"], dtype=np.int64)
        predictions = probs.argmax(axis=1)
        sorted_probs = np.sort(probs, axis=1)
        confidence = sorted_probs[:, -1]
        margin = (
            sorted_probs[:, -1]
            if sorted_probs.shape[1] == 1
            else sorted_probs[:, -1] - sorted_probs[:, -2]
        )
        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
        condition_names = list(dataset.get("condition_name", [f"domain_{domain}"] * len(labels)))
        bearing_ids = list(dataset.get("bearing_id", [""] * len(labels)))
        source_record_ids = list(dataset.get("source_record_id", dataset["file_id"]))
        speed_rpm = np.asarray(dataset.get("speed_rpm", np.full(len(labels), np.nan)))
        torque_nm = np.asarray(dataset.get("torque_nm", np.full(len(labels), np.nan)))
        radial_force_n = np.asarray(dataset.get("radial_force_n", np.full(len(labels), np.nan)))
        for index, (label, prediction) in enumerate(zip(labels, predictions)):
            row: dict[str, object] = {
                "sample_id": int(np.asarray(dataset["sample_id"])[index]),
                "domain_id": domain,
                "condition_name": condition_names[index],
                "file_id": list(dataset["file_id"])[index],
                "source_record_id": source_record_ids[index],
                "bearing_id": bearing_ids[index],
                "window_start": int(np.asarray(dataset["window_start"])[index]),
                "speed_rpm": float(speed_rpm[index]),
                "torque_nm": float(torque_nm[index]),
                "radial_force_n": float(radial_force_n[index]),
                "true_label": int(label),
                "true_class": class_names[int(label)],
                "predicted_label": int(prediction),
                "predicted_class": class_names[int(prediction)],
                "correct": int(label) == int(prediction),
                "confidence": float(confidence[index]),
                "top1_top2_margin": float(margin[index]),
                "entropy": float(entropy[index]),
            }
            for class_index, probability in enumerate(probs[index]):
                row[f"p_C{class_index:02d}"] = float(probability)
            rows.append(row)
    return rows


def _confusion_matrix_rows(
    predictions: list[dict[str, object]],
    class_names: list[str],
) -> list[dict[str, object]]:
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for row in predictions:
        matrix[int(row["true_label"]), int(row["predicted_label"])] += 1
    rows: list[dict[str, object]] = []
    for true_index, true_class in enumerate(class_names):
        item: dict[str, object] = {
            "true_label": true_index,
            "true_class": true_class,
            "support": int(matrix[true_index].sum()),
        }
        for predicted_index in range(len(class_names)):
            item[f"pred_C{predicted_index:02d}"] = int(matrix[true_index, predicted_index])
        rows.append(item)
    return rows


def _split_counts(
    train: dict[int, dict[str, object]],
    val: dict[int, dict[str, object]],
    test: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain in sorted(train):
        rows.append(
            {
                "domain": domain,
                "train": len(np.asarray(train[domain]["y"])),
                "val": len(np.asarray(val[domain]["y"])),
                "test": len(np.asarray(test[domain]["y"])),
            }
        )
    return rows


def _aggregate_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    strategies = sorted(set(str(row["strategy"]) for row in rows))
    for strategy in strategies:
        subset = [row for row in rows if row["strategy"] == strategy]
        result: dict[str, object] = {"strategy": strategy, "seeds": len(subset)}
        for metric in ["ACC", "LA", "FM"]:
            values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(result)
    return output


def _write_accuracy_matrix(path: Path, domain_order: list[int], matrix: np.ndarray) -> None:
    rows: list[dict[str, object]] = []
    for episode in range(len(domain_order)):
        row: dict[str, object] = {"episode": episode, "trained_domain": domain_order[episode]}
        for idx, domain in enumerate(domain_order):
            value = matrix[episode, idx]
            row[f"D{domain}"] = "" if not np.isfinite(value) else float(value)
        rows.append(row)
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer.")
    return parsed


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


if __name__ == "__main__":
    raise SystemExit(main())
