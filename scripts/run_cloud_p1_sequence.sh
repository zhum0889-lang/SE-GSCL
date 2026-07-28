#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CWRU_ROOT="${CWRU_ROOT:-/mnt/workspace/fdllm/data/CWRU}"
QWEN_ROOT="${QWEN_ROOT:-/mnt/workspace/fdllm/models/Qwen}"
QWEN_PATH="${QWEN_PATH:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
INITIAL_EPOCHS="${INITIAL_EPOCHS:-10}"
CONTINUAL_EPOCHS="${CONTINUAL_EPOCHS:-10}"
STRATEGIES="${STRATEGIES:-sequential,balanced_replay,full}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if ! "$PYTHON_BIN" - <<'PY'
import numpy
import scipy
import sklearn
import torch
import transformers
PY
then
  echo "Missing dependencies. Run: pip install -r requirements.txt" >&2
  exit 4
fi

if [[ ! -d "$CWRU_ROOT" ]]; then
  echo "CWRU directory not found: $CWRU_ROOT" >&2
  exit 2
fi

SEMANTIC_CACHE="results/semantic_cache/qwen25_7b_bearing4_v1"
if [[ ! -f "$SEMANTIC_CACHE/text_embeddings.npz" ]]; then
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
  "$PYTHON_BIN" scripts/cache_text_embeddings.py \
    --model "$QWEN_PATH" \
    --ontology configs/semantics/bearing_faults_4.json \
    --output-dir "$SEMANTIC_CACHE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch-size 2 \
    --max-length 128 \
    --local-files-only
fi

if ! "$PYTHON_BIN" -m unittest discover -s tests -q; then
  echo "Unit tests failed; sequence experiment stopped." >&2
  exit 5
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="results/cloud_p1_sequence/${RUN_ID}_seed${SEED}"
mkdir -p "$RUN_ROOT"

IFS=',' read -ra STRATEGY_LIST <<< "$STRATEGIES"
for strategy in "${STRATEGY_LIST[@]}"; do
  strategy="$(echo "$strategy" | xargs)"
  echo "Running strategy: $strategy"
  "$PYTHON_BIN" scripts/train_p1_global.py \
    --dataset cwru4 \
    --data-root "$CWRU_ROOT" \
    --text-cache "$SEMANTIC_CACHE" \
    --output-dir "$RUN_ROOT/$strategy" \
    --domains 0,1,2,3 \
    --strategy "$strategy" \
    --window-size 1024 \
    --step-size 1024 \
    --max-windows-per-file 24 \
    --semantic-dim 256 \
    --num-tokens 32 \
    --batch-size 64 \
    --initial-epochs "$INITIAL_EPOCHS" \
    --continual-epochs "$CONTINUAL_EPOCHS" \
    --replay-per-class 4 \
    --learning-rate 0.001 \
    --lambda-cc 0.1 \
    --lambda-dec 0.001 \
    --lambda-rel 1.0 \
    --seed "$SEED" \
    --device "$DEVICE" \
    > "$RUN_ROOT/${strategy}.log"
done

"$PYTHON_BIN" scripts/summarize_p1_sequence.py --root "$RUN_ROOT"
echo "RETURN THIS FILE: $ROOT/$RUN_ROOT/comparison.json"
