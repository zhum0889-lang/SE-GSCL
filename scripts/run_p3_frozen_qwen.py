"""Run the P3.0 frozen-Qwen structured semantic prompt baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.llm import (  # noqa: E402
    build_diagnostic_messages,
    evaluate_llm_outputs,
    parse_diagnostic_json,
    select_evaluation_rows,
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _public_packet(row: dict[str, Any]) -> dict[str, Any]:
    blocked = {
        "ground_truth_class_id",
        "ground_truth_class_name",
        "is_correct",
        "physical_symptom_targets",
    }
    return {key: value for key, value in row.items() if key not in blocked}


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    p2_dir = Path(args.p2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(
        (p2_dir / "p2_report.json").read_text(encoding="utf-8")
    )
    source_rows = _load_rows(p2_dir / "evaluation_predictions.jsonl")
    rows = select_evaluation_rows(source_rows, args.max_samples)

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for P3.") from exc
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "left"
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
    model.to(torch.device(args.device)).eval().requires_grad_(False)

    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        conversations = [
            build_diagnostic_messages(_public_packet(row))
            for row in batch_rows
        ]
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in conversations
        ]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(model.device) for key, value in encoded.items()
        }
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = int(encoded["input_ids"].shape[1])
        responses = tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
        )
        for row, response in zip(batch_rows, responses):
            records.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "domain_id": int(row["domain_id"]),
                    "packet": row,
                    "llm_raw_output": response,
                    "parsed_output": parse_diagnostic_json(response),
                }
            )
    metrics = evaluate_llm_outputs(records)
    with (output_dir / "p3_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "ok",
        "stage": "P3.0 frozen-Qwen structured semantic prompt baseline",
        "model": str(args.model),
        "p2_dir": str(p2_dir.resolve()),
        "p2_stage": report["stage"],
        "continuous_vector_prompt_enabled": False,
        "qwen_frozen": True,
        "generation": {
            "do_sample": False,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "selection": {
            "method": "condition-balanced uncertainty stratification",
            "available_samples": len(source_rows),
            "selected_samples": len(rows),
        },
        "metrics": metrics,
        "next_stage": (
            "Train a continuous semantic prompt adapter and compare it with "
            "this structured-text baseline."
        ),
    }
    (output_dir / "p3_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
