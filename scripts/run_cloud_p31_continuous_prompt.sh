#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_ROOT="${QWEN_ROOT:-/mnt/workspace/fdllm/models/Qwen}"
QWEN_PATH="${QWEN_PATH:-}"
P2_DIR="${P2_DIR:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"
AUXILIARY_WEIGHT="${AUXILIARY_WEIGHT:-0.5}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-32}"
MAX_VALIDATION_SAMPLES="${MAX_VALIDATION_SAMPLES:-16}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-32}"
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
      | while read -r candidate; do
          directory="$(dirname "$candidate")"
          if [[ -f "$directory/p2_prompt_train.npz" \
             && -f "$directory/p2_prompt_validation.npz" ]]; then
            echo "$candidate"
          fi
        done \
      | sort \
      | tail -n 1 || true
  )"
  if [[ -n "$report_path" ]]; then
    P2_DIR="$(dirname "$report_path")"
  fi
fi
if [[ -z "$P2_DIR" \
   || ! -f "$P2_DIR/p2_prompt_train.npz" \
   || ! -f "$P2_DIR/p2_prompt_validation.npz" ]]; then
  echo "No P3.1-ready P2.2 result found." >&2
  echo "Rerun the current P2.2 full stage or set P2_DIR explicitly." >&2
  exit 2
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
  exit 3
fi

if ! "$PYTHON_BIN" - <<'PY'
import numpy
import torch
import transformers
PY
then
  echo "Missing P3.1 dependencies for Python: $PYTHON_BIN" >&2
  echo "Install them with: $PYTHON_BIN -m pip install -r requirements.txt" >&2
  exit 4
fi

"$PYTHON_BIN" -m unittest tests.test_p31_continuous_prompt -q

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/cloud_p31_continuous_prompt/${RUN_ID}_${RUN_LABEL}"
"$PYTHON_BIN" scripts/train_p31_continuous_prompt.py \
  --p2-dir "$P2_DIR" \
  --model "$QWEN_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --auxiliary-weight "$AUXILIARY_WEIGHT" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --max-validation-samples "$MAX_VALIDATION_SAMPLES" \
  --max-test-samples "$MAX_TEST_SAMPLES" \
  --gradient-checkpointing \
  --local-files-only

echo "RETURN THIS FILE: $ROOT/$OUTPUT_DIR/p31_report.json"
echo "PREDICTIONS: $ROOT/$OUTPUT_DIR/p31_predictions.jsonl"
echo "ADAPTER: $ROOT/$OUTPUT_DIR/continuous_prompt_adapter.pt"
