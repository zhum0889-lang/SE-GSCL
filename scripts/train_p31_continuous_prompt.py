"""Train leakage-free continuous semantic prompts for a pretrained LLM."""

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
    parser.add_argument("--auxiliary-weight", type=float, default=0.5)
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
    parser.add_argument(
        "--init-prompt-checkpoint",
        help=(
            "Optional frozen-LLM continuous-prompt checkpoint used to "
            "initialize progressive LoRA adaptation."
        ),
    )
    parser.add_argument(
        "--context-mode",
        choices=(
            "full",
            "no_condition",
            "no_fuzzy_identity",
            "fault_identity_only",
        ),
        default="full",
    )
    parser.add_argument(
        "--llm-tuning",
        choices=("frozen", "lora"),
        default="frozen",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
        help="Comma-separated attention projections used only in LoRA mode.",
    )
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


def _instruction(
    class_names: list[str],
    class_semantic_summaries: dict[str, str] | None = None,
) -> str:
    labels = ", ".join(class_names)
    ontology = ""
    if class_semantic_summaries:
        rows = [
            f"- {name}: {class_semantic_summaries[name]}"
            for name in class_names
            if class_semantic_summaries.get(name)
        ]
        if rows:
            ontology = "\nFault ontology:\n" + "\n".join(rows) + "\n"
    return (
        "Continuous semantic tokens for one bearing vibration sample precede "
        "this instruction. They preserve a fuzzy mixture over multiple fault "
        "identity descriptions, local physical symptoms, posterior evidence, "
        "reliability, and optional observable operating context. Treat them "
        "as graded evidence rather than as a hard upstream label."
        f"{ontology}"
        "Use operating context only to interpret how a signature shifts; do "
        "not treat a condition change itself as a fault. "
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    prompt_embeddings = adapter(contexts)
    inputs_embeds = torch.cat(
        [prompt_embeddings.to(text_embeddings.dtype), text_embeddings],
        dim=1,
    )
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
    return inputs_embeds, attention, lm_labels, prompt_embeddings


def _class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(
            f"Training split is missing fault classes: {missing}."
        )
    weights = len(labels) / (num_classes * counts.astype(np.float64))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _validation_loss(
    loader: DataLoader,
    adapter: LowRankContinuousPromptAdapter,
    model,
    instruction_ids: torch.Tensor,
    target_ids: list[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
    class_weights: torch.Tensor,
    auxiliary_weight: float,
) -> dict[str, float]:
    adapter.eval()
    model.eval()
    total = 0.0
    total_lm = 0.0
    total_auxiliary = 0.0
    samples = 0
    for contexts, labels, _ in loader:
        contexts = contexts.to(device)
        labels = labels.to(device)
        inputs_embeds, attention, lm_labels, prompt_embeddings = (
            _training_batch(
                contexts,
                labels,
                adapter,
                model,
                instruction_ids,
                target_ids,
                pad_token_id,
            )
        )
        with torch.inference_mode():
            lm_loss = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention,
                labels=lm_labels,
                use_cache=False,
            ).loss
            auxiliary_loss = torch.nn.functional.cross_entropy(
                adapter.classification_logits(prompt_embeddings),
                labels,
                weight=class_weights,
            )
            loss = lm_loss + auxiliary_weight * auxiliary_loss
        total += float(loss.cpu()) * len(labels)
        total_lm += float(lm_loss.cpu()) * len(labels)
        total_auxiliary += float(auxiliary_loss.cpu()) * len(labels)
        samples += len(labels)
    denominator = max(1, samples)
    return {
        "total": total / denominator,
        "language_model": total_lm / denominator,
        "semantic_classification": total_auxiliary / denominator,
    }


def _parse_label(text: str, class_names: list[str]) -> int:
    matches = [
        index
        for index, name in enumerate(class_names)
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE)
    ]
    return matches[0] if len(matches) == 1 else -1


def _classification_metrics(
    predicted: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(labels, predicted):
        if prediction >= 0:
            confusion[int(target), int(prediction)] += 1
    class_recalls = {
        class_names[class_id]: float(
            np.mean(predicted[labels == class_id] == class_id)
        )
        for class_id in range(num_classes)
    }
    metrics: dict[str, Any] = {
        "samples": int(len(labels)),
        "valid_label_rate": float(np.mean(predicted >= 0)),
        "accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy": float(np.mean(list(class_recalls.values()))),
        "per_class_recall": class_recalls,
        "prediction_distribution": {
            class_names[class_id]: int(np.sum(predicted == class_id))
            for class_id in range(num_classes)
        },
        "invalid_predictions": int(np.sum(predicted < 0)),
        "confusion_matrix": confusion.tolist(),
        "domain_metrics": {},
    }
    for domain in sorted(int(value) for value in np.unique(domains)):
        mask = domains == domain
        domain_recalls = [
            float(np.mean(predicted[mask & (labels == class_id)] == class_id))
            for class_id in range(num_classes)
            if np.any(mask & (labels == class_id))
        ]
        metrics["domain_metrics"][str(domain)] = {
            "samples": int(mask.sum()),
            "valid_label_rate": float(np.mean(predicted[mask] >= 0)),
            "accuracy": float(np.mean(predicted[mask] == labels[mask])),
            "balanced_accuracy": float(np.mean(domain_recalls)),
        }
    return metrics


def _paired_comparison(
    generated: np.ndarray,
    upstream: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    generated_correct = generated == labels
    upstream_correct = upstream == labels
    llm_only = int(np.sum(generated_correct & ~upstream_correct))
    upstream_only = int(np.sum(~generated_correct & upstream_correct))
    discordant = llm_only + upstream_only
    try:
        from scipy.stats import binomtest

        mcnemar_p = float(
            binomtest(
                min(llm_only, upstream_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        ) if discordant else 1.0
    except ImportError:
        mcnemar_p = None
    return {
        "samples": int(len(labels)),
        "prediction_agreement_rate": float(
            np.mean(generated == upstream)
        ),
        "both_correct": int(np.sum(generated_correct & upstream_correct)),
        "llm_only_correct": llm_only,
        "qwen_only_correct": llm_only,
        "upstream_only_correct": upstream_only,
        "both_wrong": int(np.sum(~generated_correct & ~upstream_correct)),
        "correction_rate": float(llm_only / max(1, len(labels))),
        "corruption_rate": float(upstream_only / max(1, len(labels))),
        "net_correction_rate": float(
            (llm_only - upstream_only) / max(1, len(labels))
        ),
        "mcnemar_exact_p_value": mcnemar_p,
        "qwen_minus_upstream_accuracy": float(
            np.mean(generated_correct) - np.mean(upstream_correct)
        ),
    }


def _probability_baseline(
    probabilities: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = probabilities.argmax(axis=1)
    metrics = _classification_metrics(
        predictions,
        labels,
        domains,
        class_names,
    )
    order = np.argsort(-probabilities, axis=1)
    metrics["candidate_coverage"] = {
        f"top_{k}": float(
            np.mean(np.any(order[:, :k] == labels[:, None], axis=1))
        )
        for k in range(1, probabilities.shape[1] + 1)
    }
    return metrics


def _trainable_state(module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(
    module,
    state: dict[str, torch.Tensor],
) -> None:
    parameters = dict(module.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"Trainable checkpoint parameters disappeared: {missing}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.inference_mode()
def _auxiliary_predictions(
    context: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    adapter: LowRankContinuousPromptAdapter,
    class_names: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    adapter.eval()
    predictions = []
    for start in range(0, len(labels), batch_size):
        batch = torch.from_numpy(context[start : start + batch_size]).to(
            device
        )
        prompt_tokens = adapter(batch)
        predictions.append(
            adapter.classification_logits(prompt_tokens)
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
    predicted = np.concatenate(predictions).astype(np.int64)
    return _classification_metrics(
        predicted,
        labels,
        domains,
        class_names,
    )


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
    metrics = _classification_metrics(
        predicted,
        labels,
        domains,
        class_names,
    )
    return rows, metrics


def main() -> int:
    args = parse_args()
    if args.auxiliary_weight < 0:
        raise ValueError("--auxiliary-weight must be non-negative.")
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
    train_context = build_continuous_context(
        train_arrays,
        mode=args.context_mode,
    )
    validation_context = build_continuous_context(
        validation_arrays,
        mode=args.context_mode,
    )
    test_context = build_continuous_context(
        test_arrays,
        mode=args.context_mode,
    )
    context_mean = np.zeros(
        (1, train_context.shape[1]),
        dtype=np.float32,
    )
    context_std = np.ones(
        (1, train_context.shape[1]),
        dtype=np.float32,
    )

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
    lora_targets = [
        value.strip()
        for value in args.lora_target_modules.split(",")
        if value.strip()
    ]
    if args.llm_tuning == "lora":
        if not lora_targets:
            raise ValueError("LoRA mode requires at least one target module.")
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "LoRA mode requires PEFT. Install requirements.txt first."
            ) from exc
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=lora_targets,
                bias="none",
            ),
        )
    # In frozen mode gradients traverse the LLM to reach continuous prompt
    # tokens, while every pretrained parameter remains fixed. LoRA mode updates
    # only low-rank attention adapters and never performs full-model training.
    model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    class_names = [str(value) for value in report["class_names"]]
    class_semantic_summaries = {
        str(key): str(value)
        for key, value in report.get("class_semantic_summaries", {}).items()
    }
    instruction_text = _instruction(
        class_names,
        class_semantic_summaries,
    )
    instruction_ids = torch.tensor(
        tokenizer.encode(
            instruction_text,
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
        num_classes=len(class_names),
    ).to(device)
    initialization_source = None
    if args.init_prompt_checkpoint:
        initialization_path = Path(args.init_prompt_checkpoint)
        if not initialization_path.is_file():
            raise FileNotFoundError(
                f"Missing prompt initialization checkpoint: {initialization_path}"
            )
        initialization = _load_checkpoint(initialization_path)
        initialization_config = dict(initialization["adapter_config"])
        expected = {
            "input_dim": train_context.shape[1],
            "hidden_size": int(model.config.hidden_size),
            "num_prompt_tokens": args.num_prompt_tokens,
            "rank": args.adapter_rank,
            "num_classes": len(class_names),
            "context_mode": args.context_mode,
        }
        mismatches = {
            key: (initialization_config.get(key), value)
            for key, value in expected.items()
            if initialization_config.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Prompt initialization is incompatible with the requested "
                f"adapter: {mismatches}"
            )
        if list(initialization["class_names"]) != class_names:
            raise ValueError("Prompt initialization class order does not match P2.")
        adapter.load_state_dict(initialization["adapter_state_dict"], strict=True)
        initialization_source = str(initialization_path.resolve())
    class_weights = _class_weights(
        train_arrays["labels"],
        len(class_names),
        device,
    )
    model_trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimized_parameters = [
        *adapter.parameters(),
        *model_trainable_parameters,
    ]
    optimizer = torch.optim.AdamW(
        optimized_parameters,
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
    best_llm_state = None
    stale = 0
    for epoch in range(args.epochs):
        adapter.train()
        model.train(args.llm_tuning == "lora")
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        total_lm = 0.0
        total_auxiliary = 0.0
        samples = 0
        for step, (contexts, labels, _) in enumerate(train_loader):
            contexts = contexts.to(device)
            labels = labels.to(device)
            (
                inputs_embeds,
                attention,
                lm_labels,
                prompt_embeddings,
            ) = _training_batch(
                contexts,
                labels,
                adapter,
                model,
                instruction_ids,
                targets,
                tokenizer.pad_token_id,
            )
            # Teacher-forced language-model loss trains the adapter to make the
            # frozen LLM emit the class name from continuous prompt tokens.
            lm_loss = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention,
                labels=lm_labels,
                use_cache=False,
            ).loss
            # The auxiliary probe stabilizes class separation in prompt space;
            # it is not the final diagnostic output reported for P3.1.
            auxiliary_loss = torch.nn.functional.cross_entropy(
                adapter.classification_logits(prompt_embeddings),
                labels,
                weight=class_weights,
            )
            loss = lm_loss + args.auxiliary_weight * auxiliary_loss
            (loss / args.gradient_accumulation).backward()
            if (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu()) * len(labels)
            total_lm += float(lm_loss.detach().cpu()) * len(labels)
            total_auxiliary += (
                float(auxiliary_loss.detach().cpu()) * len(labels)
            )
            samples += len(labels)
        # Select the adapter checkpoint using held-out source-condition data;
        # labels from unseen test conditions remain unavailable during training.
        validation_losses = _validation_loss(
            validation_loader,
            adapter,
            model,
            instruction_ids,
            targets,
            tokenizer.pad_token_id,
            device,
            class_weights,
            args.auxiliary_weight,
        )
        row = {
            "epoch": epoch,
            "training_loss": total / max(1, samples),
            "training_language_model_loss": (
                total_lm / max(1, samples)
            ),
            "training_semantic_classification_loss": (
                total_auxiliary / max(1, samples)
            ),
            "validation_loss": validation_losses["total"],
            "validation_language_model_loss": (
                validation_losses["language_model"]
            ),
            "validation_semantic_classification_loss": (
                validation_losses["semantic_classification"]
            ),
        }
        history.append(row)
        if validation_losses["total"] < best_loss - 1e-5:
            best_loss = validation_losses["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())
            best_llm_state = _trainable_state(model)
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Continuous prompt training produced no checkpoint.")
    adapter.load_state_dict(best_state)
    if best_llm_state:
        _restore_trainable_state(model, best_llm_state)
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
    auxiliary_metrics = _auxiliary_predictions(
        test_context.astype(np.float32),
        test_arrays["labels"].astype(np.int64),
        test_arrays["domains"].astype(np.int64),
        adapter,
        class_names,
        args.batch_size,
        device,
    )
    fused_predictions = test_arrays["fused_probabilities"].argmax(axis=1)
    generated_predictions = np.asarray(
        [row["predicted_class_id"] for row in predictions],
        dtype=np.int64,
    )
    for row, upstream_prediction in zip(predictions, fused_predictions):
        upstream_id = int(upstream_prediction)
        row["upstream_fused_class_id"] = upstream_id
        row["upstream_fused_class_name"] = class_names[upstream_id]
        row["qwen_upstream_agreement"] = (
            row["predicted_class_id"] == upstream_id
        )
    semantic_baselines = {
        "global_fault_identity": _probability_baseline(
            test_arrays["global_probabilities"],
            test_arrays["labels"].astype(np.int64),
            test_arrays["domains"].astype(np.int64),
            class_names,
        ),
        "local_symptom": _probability_baseline(
            test_arrays["local_probabilities"],
            test_arrays["labels"].astype(np.int64),
            test_arrays["domains"].astype(np.int64),
            class_names,
        ),
        "hierarchical_fusion": _probability_baseline(
            test_arrays["fused_probabilities"],
            test_arrays["labels"].astype(np.int64),
            test_arrays["domains"].astype(np.int64),
            class_names,
        ),
    }
    fused_metrics = semantic_baselines["hierarchical_fusion"]
    paired_metrics = _paired_comparison(
        generated_predictions,
        fused_predictions,
        test_arrays["labels"].astype(np.int64),
    )
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
            "num_classes": adapter.num_classes,
            "context_mode": args.context_mode,
            "llm_tuning": args.llm_tuning,
        },
        "context_mean": torch.from_numpy(context_mean),
        "context_std": torch.from_numpy(context_std),
        "class_names": class_names,
        "instruction": instruction_text,
    }
    torch.save(checkpoint, output_dir / "continuous_prompt_adapter.pt")
    if args.llm_tuning == "lora":
        model.save_pretrained(output_dir / "llm_lora_adapter")
    with (output_dir / "p31_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "status": "ok",
        "stage": (
            "P3.1.1 class-balanced continuous semantic prompt "
            "classification"
        ),
        "model": str(args.model),
        "p2_dir": str(p2_dir.resolve()),
        "qwen_frozen": args.llm_tuning == "frozen",
        "llm_frozen": args.llm_tuning == "frozen",
        "llm_tuning": args.llm_tuning,
        "continuous_vector_prompt_enabled": True,
        "input_contract": {
            "fuzzy_semantic_dim": int(
                train_arrays["fuzzy_symptom_embeddings"].shape[1]
            ),
            "context_dim": int(train_context.shape[1]),
            "num_prompt_tokens": args.num_prompt_tokens,
            "adapter_rank": args.adapter_rank,
            "context_mode": args.context_mode,
            "condition_context_enabled": args.context_mode == "full",
            "fuzzy_identity_enabled": (
                "fuzzy_identity_embeddings" in train_arrays
            ),
            "description_posterior_enabled": (
                "identity_description_probabilities" in train_arrays
            ),
            "ontology_guidance_enabled": bool(class_semantic_summaries),
            "normalization": (
                "sample-wise L2 normalization for fuzzy semantics; "
                "posterior and reliability values remain in [0,1]"
            ),
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
            "seed": args.seed,
            "initialization_source": initialization_source,
            "objective": (
                "language_model_loss + auxiliary_weight * "
                "class_balanced_semantic_classification_loss"
            ),
            "auxiliary_weight": args.auxiliary_weight,
            "class_weights": [
                float(value)
                for value in class_weights.detach().cpu().tolist()
            ],
            "trainable_parameters": {
                "continuous_prompt_adapter": int(
                    sum(parameter.numel() for parameter in adapter.parameters())
                ),
                "llm_adapter": int(
                    sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                ),
            },
            "lora": (
                {
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": lora_targets,
                }
                if args.llm_tuning == "lora"
                else None
            ),
            "history": history,
        },
        "continuous_prompt_metrics": prompt_metrics,
        "training_only_auxiliary_probe_metrics": auxiliary_metrics,
        "upstream_semantic_baselines": semantic_baselines,
        "upstream_fused_baseline": fused_metrics,
        "qwen_upstream_paired_comparison": paired_metrics,
        "llm_upstream_paired_comparison": paired_metrics,
        "note": (
            "This stage isolates direct-vector label generation. "
            "The auxiliary classifier regularizes prompt tokens during "
            "training and is not used for the reported LLM diagnosis. "
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
