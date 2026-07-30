"""Generate evidence-grounded explanations from continuous prompt tokens."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.llm import (  # noqa: E402
    LowRankContinuousPromptAdapter,
    apply_diagnosis_locked_control,
    apply_semantic_control,
    build_continuous_context,
    build_continuous_diagnostic_messages,
    evaluate_llm_outputs,
    parse_diagnostic_json,
    select_evaluation_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-dir", required=True)
    parser.add_argument("--p31-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--unlock-diagnosis",
        action="store_true",
        help=(
            "Ablation: let the explanation pass rediagnose instead of "
            "preserving the P3.1 direct continuous-token label."
        ),
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sample_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["domain_id"]), int(row["sample_id"])


def _expand_candidates(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    index: int,
    class_names: Sequence[str],
) -> dict[str, Any]:
    packet = dict(row)
    fused = arrays["fused_probabilities"][index]
    global_probabilities = arrays["global_probabilities"][index]
    local_probabilities = arrays["local_probabilities"][index]
    order = np.argsort(-fused)
    packet["top_candidates"] = [
        {
            "class_id": int(class_id),
            "class_name": str(class_names[int(class_id)]),
            "probability": float(fused[int(class_id)]),
            "global_probability": float(
                global_probabilities[int(class_id)]
            ),
            "local_probability": float(
                local_probabilities[int(class_id)]
            ),
        }
        for class_id in order
    ]
    return packet


def _continuous_packet(
    row: dict[str, Any],
    direct_prediction: dict[str, Any],
) -> dict[str, Any]:
    packet = dict(row)
    class_id = int(direct_prediction["predicted_class_id"])
    if class_id < 0:
        raise ValueError("P3.1 direct prediction must be a valid class.")
    candidates = {
        int(candidate["class_id"]): candidate
        for candidate in packet["top_candidates"]
    }
    candidate = candidates[class_id]
    packet["p2_predicted_class_id"] = int(packet["predicted_class_id"])
    packet["p2_predicted_class_name"] = str(
        packet["predicted_class_name"]
    )
    packet["predicted_class_id"] = class_id
    packet["predicted_class_name"] = str(
        direct_prediction["predicted_class_name"]
    )
    packet["confidence"] = float(candidate["probability"])
    return packet


def _assemble_continuous_inputs(
    prompt_embeddings: torch.Tensor,
    token_rows: Sequence[torch.Tensor],
    embedding_layer,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad text while keeping semantic tokens immediately before it."""

    if prompt_embeddings.ndim != 3:
        raise ValueError("prompt_embeddings must have shape [B,Q,H].")
    if len(token_rows) != len(prompt_embeddings):
        raise ValueError("token_rows must match prompt batch size.")
    maximum_text = max(len(row) for row in token_rows)
    batch_size, prompt_length, hidden_size = prompt_embeddings.shape
    total_length = maximum_text + prompt_length
    device = prompt_embeddings.device
    pad_ids = torch.full(
        (batch_size, total_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    inputs = embedding_layer(pad_ids)
    if inputs.shape[-1] != hidden_size:
        raise ValueError("Prompt and text embedding sizes must match.")
    attention = torch.zeros(
        (batch_size, total_length),
        dtype=torch.long,
        device=device,
    )
    for index, token_ids in enumerate(token_rows):
        token_ids = token_ids.to(device)
        start = maximum_text - len(token_ids)
        prompt = prompt_embeddings[index].to(inputs.dtype)
        text = embedding_layer(token_ids)
        inputs[index, start : start + prompt_length] = prompt
        inputs[
            index,
            start + prompt_length : start + prompt_length + len(token_ids),
        ] = text
        attention[
            index,
            start : start + prompt_length + len(token_ids),
        ] = 1
    return inputs, attention


def _preservation_metrics(
    records: Sequence[dict[str, Any]],
) -> dict[str, float | int]:
    valid = [
        record
        for record in records
        if isinstance(record.get("parsed_output"), dict)
    ]
    preserved = sum(
        str(record["parsed_output"].get("diagnosis", ""))
        == str(record["direct_prediction"]["predicted_class_name"])
        for record in valid
    )
    direct_correct = sum(
        str(record["direct_prediction"]["predicted_class_name"])
        == str(record["packet"]["ground_truth_class_name"])
        for record in records
    )
    p2_correct = sum(
        str(record["packet"]["p2_predicted_class_name"])
        == str(record["packet"]["ground_truth_class_name"])
        for record in records
    )
    return {
        "samples": len(records),
        "direct_prompt_accuracy": direct_correct / max(1, len(records)),
        "p2_fused_accuracy": p2_correct / max(1, len(records)),
        "valid_explanation_outputs": len(valid),
        "diagnosis_preservation_rate": preserved / max(1, len(records)),
        "valid_output_diagnosis_preservation_rate": (
            preserved / max(1, len(valid))
        ),
    }


def _failure_audit(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[int]]:
    return {
        "unparseable_sample_ids": [
            int(record["sample_id"])
            for record in records
            if not isinstance(record.get("parsed_output"), dict)
        ],
        "raw_diagnosis_drift_sample_ids": [
            int(record["sample_id"])
            for record in records
            if isinstance(record.get("parsed_output"), dict)
            and str(record["parsed_output"].get("diagnosis", ""))
            != str(record["direct_prediction"]["predicted_class_name"])
        ],
        "semantic_control_repaired_sample_ids": [
            int(record["sample_id"])
            for record in records
            if record["controlled_output"]["semantic_control_repairs"]
        ],
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    p2_dir = Path(args.p2_dir)
    p31_dir = Path(args.p31_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p2_report = json.loads(
        (p2_dir / "p2_report.json").read_text(encoding="utf-8")
    )
    p31_report = json.loads(
        (p31_dir / "p31_report.json").read_text(encoding="utf-8")
    )
    source_rows = _load_rows(p2_dir / "evaluation_predictions.jsonl")
    rows = select_evaluation_rows(source_rows, args.max_samples)
    arrays = _load_npz(p2_dir / "p2_outputs.npz")
    context = build_continuous_context(arrays)
    array_lookup = {
        (int(domain), int(sample_id)): index
        for index, (domain, sample_id) in enumerate(
            zip(arrays["domains"], arrays["sample_ids"])
        )
    }
    direct_rows = _load_rows(p31_dir / "p31_predictions.jsonl")
    direct_lookup = {_sample_key(row): row for row in direct_rows}

    checkpoint = _load_checkpoint(
        p31_dir / "continuous_prompt_adapter.pt"
    )
    class_names = [str(value) for value in checkpoint["class_names"]]
    adapter_config = dict(checkpoint["adapter_config"])

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for P3.2.") from exc
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
    model.to(device).eval().requires_grad_(False)
    adapter = LowRankContinuousPromptAdapter(
        input_dim=int(adapter_config["input_dim"]),
        hidden_size=int(adapter_config["hidden_size"]),
        num_prompt_tokens=int(adapter_config["num_prompt_tokens"]),
        rank=int(adapter_config["rank"]),
        num_classes=int(adapter_config["num_classes"]),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval().requires_grad_(False)
    context_mean = checkpoint["context_mean"].numpy()
    context_std = checkpoint["context_std"].numpy()
    context = (context - context_mean) / np.maximum(context_std, 1e-6)

    records: list[dict[str, Any]] = []
    repair_counts: Counter[str] = Counter()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch_packets = []
        batch_direct = []
        batch_context = []
        token_rows = []
        for row in batch_rows:
            key = _sample_key(row)
            if key not in array_lookup or key not in direct_lookup:
                raise KeyError(f"Missing P2/P3.1 sample mapping for {key}.")
            index = array_lookup[key]
            expanded = _expand_candidates(
                row,
                arrays,
                index,
                class_names,
            )
            direct = direct_lookup[key]
            packet = _continuous_packet(expanded, direct)
            messages = build_continuous_diagnostic_messages(
                packet,
                class_names,
                locked_diagnosis=(
                    None
                    if args.unlock_diagnosis
                    else str(direct["predicted_class_name"])
                ),
            )
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            token_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_input_tokens,
            )
            batch_packets.append(packet)
            batch_direct.append(direct)
            batch_context.append(context[index])
            token_rows.append(torch.tensor(token_ids, dtype=torch.long))

        with torch.inference_mode():
            semantic_tokens = adapter(
                torch.from_numpy(
                    np.asarray(batch_context, dtype=np.float32)
                ).to(device)
            )
            inputs, attention = _assemble_continuous_inputs(
                semantic_tokens,
                token_rows,
                model.get_input_embeddings(),
                tokenizer.pad_token_id,
            )
            generated = model.generate(
                inputs_embeds=inputs,
                attention_mask=attention,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        for packet, direct, response in zip(
            batch_packets,
            batch_direct,
            responses,
        ):
            parsed = parse_diagnostic_json(response)
            controlled = (
                apply_semantic_control(packet, parsed)
                if args.unlock_diagnosis
                else apply_diagnosis_locked_control(
                    packet,
                    parsed,
                    str(direct["predicted_class_name"]),
                )
            )
            repair_counts.update(controlled["semantic_control_repairs"])
            records.append(
                {
                    "sample_id": int(packet["sample_id"]),
                    "domain_id": int(packet["domain_id"]),
                    "packet": packet,
                    "direct_prediction": direct,
                    "llm_raw_output": response,
                    "parsed_output": parsed,
                    "controlled_output": controlled,
                }
            )

    raw_metrics = evaluate_llm_outputs(records)
    controlled_records = [
        {**record, "parsed_output": record["controlled_output"]}
        for record in records
    ]
    controlled_metrics = evaluate_llm_outputs(controlled_records)
    repaired_samples = sum(
        bool(record["controlled_output"]["semantic_control_repairs"])
        for record in records
    )
    preservation = _preservation_metrics(records)
    with (output_dir / "p32_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "ok",
        "stage": (
            "P3.2.0 unlocked continuous-token explanation ablation"
            if args.unlock_diagnosis
            else "P3.2.1 diagnosis-locked continuous-token explanation"
        ),
        "model": str(args.model),
        "p2_dir": str(p2_dir.resolve()),
        "p31_dir": str(p31_dir.resolve()),
        "p2_stage": p2_report["stage"],
        "p31_stage": p31_report["stage"],
        "qwen_frozen": True,
        "continuous_vector_prompt_enabled": True,
        "text_prompt_exposes_upstream_top1_or_probabilities": False,
        "text_prompt_exposes_ground_truth": False,
        "text_prompt_exposes_stage1_qwen_diagnosis": (
            not args.unlock_diagnosis
        ),
        "diagnosis_lock_source": (
            None
            if args.unlock_diagnosis
            else "P3.1.1 frozen-Qwen direct continuous-token prediction"
        ),
        "selection": {
            "method": "condition-balanced uncertainty stratification",
            "available_samples": len(source_rows),
            "selected_samples": len(rows),
        },
        "generation": {
            "do_sample": False,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "diagnosis_preservation": preservation,
        "failure_audit": _failure_audit(records),
        "raw_metrics": raw_metrics,
        "controlled_metrics": {
            **controlled_metrics,
            "semantic_control_repair_rate": (
                repaired_samples / max(1, len(records))
            ),
        },
        "semantic_control": {
            "repair_counts": dict(sorted(repair_counts.items())),
        },
        "note": (
            "Ground-truth labels are used only after generation. The textual "
            "prompt supplies ontology and physical evidence"
            + (
                ". The unlocked ablation does not expose the preceding Qwen "
                "diagnosis"
                if args.unlock_diagnosis
                else ", together with the preceding Qwen continuous-token "
                "diagnosis"
            )
            + ", and never exposes P2 labels, candidate probabilities, or "
            "ground truth."
        ),
    }
    (output_dir / "p32_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
