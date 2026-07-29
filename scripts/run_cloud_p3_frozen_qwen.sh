#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_ROOT="${QWEN_ROOT:-/mnt/workspace/fdllm/models/Qwen}"
QWEN_PATH="${QWEN_PATH:-}"
P2_DIR="${P2_DIR:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
if [[ -z "$P2_DIR" || ! -f "$P2_DIR/evaluation_predictions.jsonl" ]]; then
  echo "No complete P2.2 result found. Set P2_DIR explicitly." >&2
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
import torch
import transformers
PY
then
  echo "Missing P3 dependencies for Python: $PYTHON_BIN" >&2
  echo "Install them with: $PYTHON_BIN -m pip install -r requirements.txt" >&2
  exit 4
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/cloud_p3_frozen_qwen/${RUN_ID}"
"$PYTHON_BIN" scripts/run_p3_frozen_qwen.py \
  --p2-dir "$P2_DIR" \
  --model "$QWEN_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --local-files-only

echo "RETURN THIS FILE: $ROOT/$OUTPUT_DIR/p3_report.json"
echo "SAMPLE OUTPUTS: $ROOT/$OUTPUT_DIR/p3_predictions.jsonl"
