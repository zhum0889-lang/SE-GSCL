"""Train a leakage-free continuous semantic prompt for frozen Qwen."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.llm import (  # noqa: E402
    LowRankContinuousPromptAdapter,
    build_continuous_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-prompt-tokens", type=int, default=4)
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}


def _stratified_indices(
    labels: np.ndarray,
    domains: np.ndarray,
    limit: int,
    seed: int,
) -> np.ndarray:
    if limit <= 0 or limit >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (domain, label) in enumerate(zip(domains, labels)):
        groups.setdefault((int(domain), int(label)), []).append(index)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def _subset(
    arrays: dict[str, np.ndarray],
    limit: int,
    seed: int,
) -> dict[str, np.ndarray]:
    indices = _stratified_indices(
        np.asarray(arrays["labels"]),
        np.asarray(arrays["domains"]),
        limit,
        seed,
    )
    return {
        key: value[indices]
        if len(value) == len(arrays["labels"])
        else value
        for key, value in arrays.items()
    }


def _instruction(class_names: list[str]) -> str:
    labels = ", ".join(class_names)
    return (
        "Continuous semantic tokens encoding one bearing vibration sample "
        "precede this instruction. Diagnose the sample using those tokens. "
        f"Reply with exactly one label from: {labels}.\nDiagnosis:"
    )


def _target_ids(
    tokenizer,
    class_names: list[str],
) -> list[torch.Tensor]:
    rows = []
    for name in class_names:
        ids = tokenizer.encode(
            " " + name,
            add_special_tokens=False,
        )
        ids.append(tokenizer.eos_token_id)
        rows.append(torch.tensor(ids, dtype=torch.long))
    return rows


def _training_batch(
    contexts: torch.Tensor,
    labels: torch.Tensor,
    adapter: LowRankContinuousPromptAdapter,
    model,
    instruction_ids: torch.Tensor,
    target_ids: list[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = contexts.device
    batch_size = len(labels)
    targets = [target_ids[int(value)].to(device) for value in labels]
    maximum = max(len(value) for value in targets)
    text_length = len(instruction_ids) + maximum
    input_ids = torch.full(
        (batch_size, text_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros(
        (batch_size, text_length),
        dtype=torch.long,
        device=device,
    )
    lm_labels = torch.full(
        (batch_size, text_length),
        -100,
        dtype=torch.long,
        device=device,
    )
    prefix_length = len(instruction_ids)
    for index, target in enumerate(targets):
        input_ids[index, :prefix_length] = instruction_ids
        input_ids[index, prefix_length : prefix_length + len(target)] = target
        attention[index, : prefix_length + len(target)] = 1
        lm_labels[index, prefix_length : prefix_length + len(target)] = target
    text_embeddings = model.get_input_embeddings()(input_ids)
    prompt_embeddings = adapter(contexts).to(text_embeddings.dtype)
    inputs_embeds = torch.cat([prompt_embeddings, text_embeddings], dim=1)
    prompt_attention = torch.ones(
        (batch_size, adapter.num_prompt_tokens),
        dtype=attention.dtype,
        device=device,
    )
    attention = torch.cat([prompt_attention, attention], dim=1)
    prompt_labels = torch.full(
        (batch_size, adapter.num_prompt_tokens),
        -100,
        dtype=lm_labels.dtype,
        device=device,
    )
    lm_labels = torch.cat([prompt_labels, lm_labels], dim=1)
    return inputs_embeds, attention, lm_labels


def _validation_loss(
    loader: DataLoader,
    adapter: LowRankContinuousPromptAdapter,
    model,
    instruction_ids: torch.Tensor,
    target_ids: list[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> float:
    adapter.eval()
    model.eval()
    total = 0.0
    samples = 0
    for contexts, labels, _ in loader:
        contexts = contexts.to(device)
        labels = labels.to(device)
        inputs_embeds, attention, lm_labels = _training_batch(
            contexts,
            labels,
            adapter,
            model,
            instruction_ids,
            target_ids,
            pad_token_id,
        )
        with torch.inference_mode():
            loss = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention,
                labels=lm_labels,
                use_cache=False,
            ).loss
        total += float(loss.cpu()) * len(labels)
        samples += len(labels)
    return total / max(1, samples)


def _parse_label(text: str, class_names: list[str]) -> int:
    matches = [
        index
        for index, name in enumerate(class_names)
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE)
    ]
    return matches[0] if len(matches) == 1 else -1


@torch.inference_mode()
def _generate_predictions(
    context: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    sample_ids: np.ndarray,
    adapter: LowRankContinuousPromptAdapter,
    model,
    tokenizer,
    instruction_ids: torch.Tensor,
    class_names: list[str],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapter.eval()
    model.eval()
    rows: list[dict[str, Any]] = []
    predictions: list[int] = []
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        batch_context = torch.from_numpy(context[start:stop]).to(device)
        prompt_embeddings = adapter(batch_context)
        instruction_batch = instruction_ids[None, :].expand(
            stop - start,
            -1,
        )
        text_embeddings = model.get_input_embeddings()(instruction_batch)
        prompt_embeddings = prompt_embeddings.to(text_embeddings.dtype)
        inputs_embeds = torch.cat(
            [prompt_embeddings, text_embeddings],
            dim=1,
        )
        attention = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=device,
        )
        generated = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for offset, text in enumerate(texts):
            index = start + offset
            prediction = _parse_label(text, class_names)
            predictions.append(prediction)
            rows.append(
                {
                    "sample_id": int(sample_ids[index]),
                    "domain_id": int(domains[index]),
                    "ground_truth_class_id": int(labels[index]),
                    "ground_truth_class_name": class_names[int(labels[index])],
                    "generated_text": text,
                    "predicted_class_id": prediction,
                    "predicted_class_name": (
                        class_names[prediction] if prediction >= 0 else None
                    ),
                    "valid_label": prediction >= 0,
                    "is_correct": prediction == int(labels[index]),
                }
            )
    predicted = np.asarray(predictions, dtype=np.int64)
    metrics: dict[str, Any] = {
        "samples": int(len(labels)),
        "valid_label_rate": float(np.mean(predicted >= 0)),
        "accuracy": float(np.mean(predicted == labels)),
        "domain_metrics": {},
    }
    for domain in sorted(int(value) for value in np.unique(domains)):
        mask = domains == domain
        metrics["domain_metrics"][str(domain)] = {
            "samples": int(mask.sum()),
            "valid_label_rate": float(np.mean(predicted[mask] >= 0)),
            "accuracy": float(np.mean(predicted[mask] == labels[mask])),
        }
    return rows, metrics


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    p2_dir = Path(args.p2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(
        (p2_dir / "p2_report.json").read_text(encoding="utf-8")
    )
    train_arrays = _subset(
        _load_npz(p2_dir / "p2_prompt_train.npz"),
        args.max_train_samples,
        args.seed,
    )
    validation_arrays = _subset(
        _load_npz(p2_dir / "p2_prompt_validation.npz"),
        args.max_validation_samples,
        args.seed + 1,
    )
    test_arrays = _subset(
        _load_npz(p2_dir / "p2_outputs.npz"),
        args.max_test_samples,
        args.seed + 2,
    )
    train_context = build_continuous_context(train_arrays)
    validation_context = build_continuous_context(validation_arrays)
    test_context = build_continuous_context(test_arrays)
    context_mean = train_context.mean(axis=0, keepdims=True)
    context_std = train_context.std(axis=0, keepdims=True)
    context_std = np.maximum(context_std, 1e-5)
    train_context = (train_context - context_mean) / context_std
    validation_context = (validation_context - context_mean) / context_std
    test_context = (test_context - context_mean) / context_std

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for P3.1.") from exc
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
    }
    if args.device.startswith("cuda"):
        transformers_major = int(
            transformers.__version__.split(".", maxsplit=1)[0]
        )
        dtype_key = "dtype" if transformers_major >= 5 else "torch_dtype"
        model_kwargs[dtype_key] = dtype_map[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    device = torch.device(args.device)
    model.to(device).requires_grad_(False)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    class_names = [str(value) for value in report["class_names"]]
    instruction_ids = torch.tensor(
        tokenizer.encode(
            _instruction(class_names),
            add_special_tokens=True,
        ),
        dtype=torch.long,
        device=device,
    )
    targets = _target_ids(tokenizer, class_names)
    adapter = LowRankContinuousPromptAdapter(
        input_dim=train_context.shape[1],
        hidden_size=int(model.config.hidden_size),
        num_prompt_tokens=args.num_prompt_tokens,
        rank=args.adapter_rank,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(train_context.astype(np.float32)),
        torch.from_numpy(train_arrays["labels"].astype(np.int64)),
        torch.from_numpy(train_arrays["domains"].astype(np.int64)),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_context.astype(np.float32)),
        torch.from_numpy(validation_arrays["labels"].astype(np.int64)),
        torch.from_numpy(validation_arrays["domains"].astype(np.int64)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    history = []
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    stale = 0
    for epoch in range(args.epochs):
        adapter.train()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        samples = 0
        for step, (contexts, labels, _) in enumerate(train_loader):
            contexts = contexts.to(device)
            labels = labels.to(device)
            inputs_embeds, attention, lm_labels = _training_batch(
                contexts,
                labels,
                adapter,
                model,
                instruction_ids,
                targets,
                tokenizer.pad_token_id,
            )
            loss = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention,
                labels=lm_labels,
                use_cache=False,
            ).loss
            (loss / args.gradient_accumulation).backward()
            if (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu()) * len(labels)
            samples += len(labels)
        validation_loss = _validation_loss(
            validation_loader,
            adapter,
            model,
            instruction_ids,
            targets,
            tokenizer.pad_token_id,
            device,
        )
        row = {
            "epoch": epoch,
            "training_loss": total / max(1, samples),
            "validation_loss": validation_loss,
        }
        history.append(row)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Continuous prompt training produced no checkpoint.")
    adapter.load_state_dict(best_state)
    model.config.use_cache = True
    predictions, prompt_metrics = _generate_predictions(
        test_context.astype(np.float32),
        test_arrays["labels"].astype(np.int64),
        test_arrays["domains"].astype(np.int64),
        test_arrays["sample_ids"].astype(np.int64),
        adapter,
        model,
        tokenizer,
        instruction_ids,
        class_names,
        args.batch_size,
        args.max_new_tokens,
        device,
    )
    fused_predictions = test_arrays["fused_probabilities"].argmax(axis=1)
    fused_metrics = {
        "accuracy": float(
            np.mean(fused_predictions == test_arrays["labels"])
        ),
        "domain_accuracy": {
            str(domain): float(
                np.mean(
                    fused_predictions[test_arrays["domains"] == domain]
                    == test_arrays["labels"][
                        test_arrays["domains"] == domain
                    ]
                )
            )
            for domain in sorted(
                int(value) for value in np.unique(test_arrays["domains"])
            )
        },
    }
    checkpoint = {
        "adapter_state_dict": {
            key: value.detach().cpu()
            for key, value in adapter.state_dict().items()
        },
        "adapter_config": {
            "input_dim": adapter.input_dim,
            "hidden_size": adapter.hidden_size,
            "num_prompt_tokens": adapter.num_prompt_tokens,
            "rank": adapter.rank,
        },
        "context_mean": torch.from_numpy(context_mean),
        "context_std": torch.from_numpy(context_std),
        "class_names": class_names,
        "instruction": _instruction(class_names),
    }
    torch.save(checkpoint, output_dir / "continuous_prompt_adapter.pt")
    with (output_dir / "p31_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "status": "ok",
        "stage": "P3.1 direct continuous semantic prompt classification",
        "model": str(args.model),
        "p2_dir": str(p2_dir.resolve()),
        "qwen_frozen": True,
        "continuous_vector_prompt_enabled": True,
        "input_contract": {
            "fuzzy_semantic_dim": int(
                train_arrays["fuzzy_symptom_embeddings"].shape[1]
            ),
            "context_dim": int(train_context.shape[1]),
            "num_prompt_tokens": args.num_prompt_tokens,
            "adapter_rank": args.adapter_rank,
        },
        "data_protocol": {
            "adapter_training_domains": sorted(
                int(value) for value in np.unique(train_arrays["domains"])
            ),
            "validation_domains": sorted(
                int(value)
                for value in np.unique(validation_arrays["domains"])
            ),
            "test_domains": sorted(
                int(value) for value in np.unique(test_arrays["domains"])
            ),
            "train_samples": int(len(train_context)),
            "validation_samples": int(len(validation_context)),
            "test_samples": int(len(test_context)),
            "test_labels_used_for_training": False,
        },
        "training": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
        },
        "continuous_prompt_metrics": prompt_metrics,
        "upstream_fused_baseline": fused_metrics,
        "note": (
            "This stage isolates direct-vector label generation. "
            "Auditable explanations remain handled by the validated P3.0.2 "
            "semantic controller."
        ),
    }
    (output_dir / "p31_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
