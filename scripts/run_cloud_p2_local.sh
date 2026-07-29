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
PHYSICS_GUIDED="${PHYSICS_GUIDED:-0}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-1.0}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.1}"
SEMANTIC_GUARD="${SEMANTIC_GUARD:-0}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.2}"
RESIDUAL_LR_MULTIPLIER="${RESIDUAL_LR_MULTIPLIER:-5.0}"
RANKING_WEIGHT="${RANKING_WEIGHT:-0.5}"
RANKING_TEMPERATURE="${RANKING_TEMPERATURE:-0.2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
ADAPTIVE_FUSION="${ADAPTIVE_FUSION:-0}"
RUN_LABEL="${RUN_LABEL:-}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if ! "$PYTHON_BIN" - <<'PY'
import matplotlib
import numpy
import scipy
import sklearn
import torch
import transformers
PY
then
  echo "Missing P2 dependencies for Python: $PYTHON_BIN" >&2
  echo "Install them with: $PYTHON_BIN -m pip install -r requirements.txt" >&2
  exit 5
fi

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
SYMPTOM_CACHE="results/semantic_cache/qwen25_7b_bearing4_symptoms_physics_v1"
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
EXTRA_ARGS=()
if [[ "$LEARNABLE_SYMPTOM_WEIGHTS" == "1" ]]; then
  EXTRA_ARGS+=(--learnable-symptom-weights)
fi
if [[ "$PHYSICS_GUIDED" == "1" ]]; then
  EXTRA_ARGS+=(
    --physics-guided
    --physics-weight "$PHYSICS_WEIGHT"
    --anchor-weight "$ANCHOR_WEIGHT"
  )
fi
if [[ "$SEMANTIC_GUARD" == "1" ]]; then
  EXTRA_ARGS+=(
    --semantic-guard
    --residual-scale "$RESIDUAL_SCALE"
    --residual-lr-multiplier "$RESIDUAL_LR_MULTIPLIER"
    --ranking-weight "$RANKING_WEIGHT"
    --ranking-temperature "$RANKING_TEMPERATURE"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  )
fi
if [[ "$ADAPTIVE_FUSION" == "1" ]]; then
  EXTRA_ARGS+=(--adaptive-fusion)
fi
STAGE_DIR="cloud_p2_local"
if [[ "$PHYSICS_GUIDED" == "1" ]]; then
  STAGE_DIR="cloud_p21_physics"
fi
if [[ "$SEMANTIC_GUARD" == "1" ]]; then
  STAGE_DIR="cloud_p22_semantic_guard"
fi
OUTPUT_DIR="results/${STAGE_DIR}/${RUN_ID}"
if [[ -n "$RUN_LABEL" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}_${RUN_LABEL}"
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
