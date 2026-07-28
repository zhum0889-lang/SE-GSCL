#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CWRU_ROOT="${CWRU_ROOT:-/mnt/workspace/fdllm/data/CWRU}"
QWEN_ROOT="${QWEN_ROOT:-/mnt/workspace/fdllm/models/Qwen}"
QWEN_PATH="${QWEN_PATH:-}"
P1_DIR="${P1_DIR:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
ADAPTER_EPOCHS="${ADAPTER_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TOP_TOKENS="${TOP_TOKENS:-4}"
LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.1}"
LOCAL_WEIGHT="${LOCAL_WEIGHT:-0.3}"
LEARNABLE_SYMPTOM_WEIGHTS="${LEARNABLE_SYMPTOM_WEIGHTS:-0}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ -z "$P1_DIR" ]]; then
  projector_path="$(
    find results/cloud_p1_sequence -type f \
      -path '*/full/projected_text_bank.pt' 2>/dev/null \
      | sort \
      | tail -n 1 || true
  )"
  if [[ -n "$projector_path" ]]; then
    P1_DIR="$(dirname "$projector_path")"
  fi
fi
if [[ -z "$P1_DIR" || ! -f "$P1_DIR/projected_text_bank.pt" ]]; then
  echo "No P2-ready P1 run found." >&2
  echo "Run the current P1 full strategy first, or set P1_DIR explicitly." >&2
  exit 2
fi

GLOBAL_CACHE="results/semantic_cache/qwen25_7b_bearing4_v1"
SYMPTOM_CACHE="results/semantic_cache/qwen25_7b_bearing4_symptoms_v1"
if [[ ! -f "$SYMPTOM_CACHE/symptom_embeddings.npz" ]]; then
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
  "$PYTHON_BIN" scripts/cache_symptom_embeddings.py \
    --model "$QWEN_PATH" \
    --ontology configs/semantics/bearing_faults_4.json \
    --output-dir "$SYMPTOM_CACHE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch-size 4 \
    --max-length 192 \
    --local-files-only
fi

if [[ ! -f "$GLOBAL_CACHE/text_embeddings.npz" ]]; then
  echo "Global text cache not found: $GLOBAL_CACHE" >&2
  exit 4
fi

"$PYTHON_BIN" -m unittest discover -s tests -q

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/cloud_p2_local/${RUN_ID}"
EXTRA_ARGS=()
if [[ "$LEARNABLE_SYMPTOM_WEIGHTS" == "1" ]]; then
  EXTRA_ARGS+=(--learnable-symptom-weights)
fi
"$PYTHON_BIN" scripts/evaluate_p2_local.py \
  --data-root "$CWRU_ROOT" \
  --global-text-cache "$GLOBAL_CACHE" \
  --symptom-text-cache "$SYMPTOM_CACHE" \
  --p1-dir "$P1_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --adapter-epochs "$ADAPTER_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --top-tokens "$TOP_TOKENS" \
  --local-temperature "$LOCAL_TEMPERATURE" \
  --local-weight "$LOCAL_WEIGHT" \
  --top-k 3 \
  --top-symptoms 4 \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" scripts/visualize_p2_results.py \
  --root "$OUTPUT_DIR" \
  --formats png,pdf \
  --dpi 300

echo "RETURN THIS FILE: $ROOT/$OUTPUT_DIR/p2_report.json"
echo "SAMPLE PACKETS: $ROOT/$OUTPUT_DIR/evaluation_predictions.jsonl"
echo "FIGURES: $ROOT/$OUTPUT_DIR/figures"
