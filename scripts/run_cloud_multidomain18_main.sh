#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/fdllm/data/MultiDomainBearing}"
TEXT_CACHE="${TEXT_CACHE:-$ROOT/results/semantic_cache/qwen25_7b_bearing4_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/paper_matrix/p1}"
SEEDS="${SEEDS:-42,52,62}"
JOBS="${JOBS:-finetune,lwf_relation,experience_replay,se_gscl_full}"
DEVICE="${DEVICE:-cuda}"
FIGURE_FORMATS="${FIGURE_FORMATS:-png,pdf}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "MultiDomainBearing directory not found: $DATA_ROOT" >&2
  exit 2
fi
if [[ ! -f "$TEXT_CACHE/text_embeddings.npz" ]]; then
  echo "Frozen text cache not found: $TEXT_CACHE/text_embeddings.npz" >&2
  exit 3
fi

"$PYTHON_BIN" scripts/audit_multidomain_overlap.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT/multidomain8_disjoint18/protocol_audit" \
  --sampling-rate 8000 \
  --protocol disjoint18 \
  --fail-on-overlap

"$PYTHON_BIN" scripts/run_paper_p1_matrix.py \
  --dataset multidomain8_disjoint18 \
  --data-root "$DATA_ROOT" \
  --text-cache "$TEXT_CACHE" \
  --output-root "$OUTPUT_ROOT" \
  --seeds "$SEEDS" \
  --jobs "$JOBS" \
  --device "$DEVICE" \
  --execute \
  --visualize \
  --figure-formats "$FIGURE_FORMATS"

echo "REPORT ROOT: $OUTPUT_ROOT/multidomain8_disjoint18"
echo "FIGURES: $OUTPUT_ROOT/multidomain8_disjoint18/figures"
