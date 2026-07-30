#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_ROOT="${QWEN_ROOT:-/mnt/workspace/fdllm/models/Qwen}"
QWEN_PATH="${QWEN_PATH:-}"
P2_DIR="${P2_DIR:-}"
P31_DIR="${P31_DIR:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
RUN_LABEL="${RUN_LABEL:-smoke}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "$P2_DIR" ]]; then
  report_path="$(
    find results/cloud_p22_semantic_guard -type f \
      -path '*_full*/p2_report.json' 2>/dev/null \
      | sort \
      | tail -n 1 || true
  )"
  if [[ -n "$report_path" ]]; then
    P2_DIR="$(dirname "$report_path")"
  fi
fi
if [[ -z "$P2_DIR" || ! -f "$P2_DIR/p2_outputs.npz" ]]; then
  echo "Set P2_DIR to a complete P2.2 export." >&2
  exit 2
fi

if [[ -z "$P31_DIR" ]]; then
  report_path="$(
    find results/cloud_p31_continuous_prompt -type f \
      -path '*_full_robust_seed42/p31_report.json' 2>/dev/null \
      | sort \
      | tail -n 1 || true
  )"
  if [[ -n "$report_path" ]]; then
    P31_DIR="$(dirname "$report_path")"
  fi
fi
if [[ -z "$P31_DIR" \
   || ! -f "$P31_DIR/continuous_prompt_adapter.pt" \
   || ! -f "$P31_DIR/p31_predictions.jsonl" ]]; then
  echo "Set P31_DIR to the predefined seed-42 P3.1.1 run." >&2
  exit 3
fi

if [[ -z "$QWEN_PATH" ]]; then
  config_path="$(
    find "$QWEN_ROOT" -maxdepth 3 -type f -name config.json 2>/dev/null \
      | grep -i 'qwen.*7b' \
      | head -n 1 || true
  )"
  if [[ -n "$config_path" ]]; then
    QWEN_PATH="$(dirname "$config_path")"
  fi
fi
if [[ -z "$QWEN_PATH" || ! -f "$QWEN_PATH/config.json" ]]; then
  echo "Set QWEN_PATH to the local Qwen2.5-7B directory." >&2
  exit 4
fi

if ! "$PYTHON_BIN" - <<'PY'
import numpy
import torch
import transformers
PY
then
  echo "Missing P3.2 dependencies for Python: $PYTHON_BIN" >&2
  exit 5
fi

"$PYTHON_BIN" -m unittest \
  tests.test_p31_continuous_prompt \
  tests.test_p3_prompting \
  tests.test_p32_continuous_explanations -q

run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="results/cloud_p32_continuous_explanations/${run_id}_${RUN_LABEL}"
"$PYTHON_BIN" scripts/run_p32_continuous_explanations.py \
  --p2-dir "$P2_DIR" \
  --p31-dir "$P31_DIR" \
  --model "$QWEN_PATH" \
  --output-dir "$output_dir" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --local-files-only

echo "RETURN THIS FILE: $ROOT/$output_dir/p32_report.json"
echo "SAMPLE OUTPUTS: $ROOT/$output_dir/p32_predictions.jsonl"
