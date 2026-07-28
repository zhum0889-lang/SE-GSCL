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
INITIAL_EPOCHS="${INITIAL_EPOCHS:-2}"
CONTINUAL_EPOCHS="${CONTINUAL_EPOCHS:-2}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -d "$CWRU_ROOT" ]]; then
  echo "CWRU directory not found: $CWRU_ROOT" >&2
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
  echo "Qwen2.5-7B config.json was not found." >&2
  echo "Set QWEN_PATH to the local model directory and run again." >&2
  exit 3
fi

"$PYTHON_BIN" - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

if [[ "$DEVICE" == cuda* ]]; then
  "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("DEVICE requests CUDA, but torch.cuda.is_available() is false.")
PY
fi

echo "[1/4] Running unit tests"
"$PYTHON_BIN" -m unittest discover -s tests -v

echo "[2/4] Auditing CWRU records"
"$PYTHON_BIN" scripts/prepare_dataset.py \
  --dataset cwru4 \
  --data-root "$CWRU_ROOT" \
  --output-dir results/data_audit/cwru4_cloud

SEMANTIC_CACHE="results/semantic_cache/qwen25_7b_bearing4_v1"
if [[ ! -f "$SEMANTIC_CACHE/text_embeddings.npz" ]]; then
  echo "[3/4] Building frozen Qwen semantic cache"
  "$PYTHON_BIN" scripts/cache_text_embeddings.py \
    --model "$QWEN_PATH" \
    --ontology configs/semantics/bearing_faults_4.json \
    --output-dir "$SEMANTIC_CACHE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch-size 2 \
    --max-length 128 \
    --local-files-only
else
  echo "[3/4] Reusing semantic cache: $SEMANTIC_CACHE"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/cloud_p1/cwru4_d0_d1_qwen7b_seed${SEED}_${RUN_ID}"
echo "[4/4] Running two-domain P1 smoke"
"$PYTHON_BIN" scripts/train_p1_global.py \
  --dataset cwru4 \
  --data-root "$CWRU_ROOT" \
  --text-cache "$SEMANTIC_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --domains 0,1 \
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
  --device "$DEVICE"

echo "Cloud P1 smoke completed."
echo "Result: $ROOT/$OUTPUT_DIR/p1_report.json"
