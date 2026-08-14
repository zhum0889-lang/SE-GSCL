#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/fdllm/se_gscl_impl}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/fdllm/data/MultiDomainBearing}"
MODEL="${MODEL:-/mnt/workspace/fdllm/models/Qwen/Qwen2___5-7B-Instruct}"
DATASET="${DATASET:-multidomain8_disjoint18}"
P1_ROOT="${P1_ROOT:-$ROOT/results/paper_matrix/p1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/paper_matrix}"
SEEDS="${SEEDS:-42,52,62}"
RUN_EXPLANATIONS="${RUN_EXPLANATIONS:-1}"

cd "$ROOT"

for seed in ${SEEDS//,/ }; do
  report="$P1_ROOT/$DATASET/seed_${seed}/se_gscl_full/p1_report.json"
  if [[ ! -f "$report" ]]; then
    echo "Missing P1 checkpoint: $report" >&2
    exit 2
  fi
done

python scripts/run_paper_downstream_matrix.py \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --p1-root "$P1_ROOT" \
  --model "$MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --seeds "$SEEDS" \
  --stage p2 \
  --p2-jobs semantic_local_adaptive \
  --device cuda \
  --execute

python scripts/run_paper_downstream_matrix.py \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --p1-root "$P1_ROOT" \
  --model "$MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --seeds "$SEEDS" \
  --stage p3 \
  --p3-p2-job semantic_local_adaptive \
  --p3-jobs continuous_identity_only,continuous_no_condition,continuous_full,continuous_full_lora \
  --device cuda \
  --dtype bfloat16 \
  --local-files-only \
  --execute

python scripts/summarize_p3_llm_experiment.py \
  --root "$OUTPUT_ROOT/p3/$DATASET" \
  --output-dir "$OUTPUT_ROOT/summary/$DATASET/llm" \
  --formats png,pdf

if [[ "$RUN_EXPLANATIONS" == "1" ]]; then
  python scripts/run_paper_downstream_matrix.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --p1-root "$P1_ROOT" \
    --model "$MODEL" \
    --output-root "$OUTPUT_ROOT" \
    --seeds "$SEEDS" \
    --stage p3 \
    --p3-p2-job semantic_local_adaptive \
    --p3-jobs explanation_unlocked,explanation_locked \
    --device cuda \
    --dtype bfloat16 \
    --local-files-only \
    --execute
fi

echo "LLM experiment complete: $OUTPUT_ROOT/summary/$DATASET/llm"
