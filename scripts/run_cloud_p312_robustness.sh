#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 52 62}"
EPOCHS="${EPOCHS:-3}"
PATIENCE="${PATIENCE:-2}"
AUXILIARY_WEIGHT="${AUXILIARY_WEIGHT:-0.5}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VALIDATION_SAMPLES="${MAX_VALIDATION_SAMPLES:-0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"
RUN_PREFIX="${RUN_PREFIX:-robust}"

cd "$ROOT"
reports=()
for seed in $SEEDS; do
  label="${RUN_PREFIX}_seed${seed}"
  echo "=== P3.1.1 seed $seed ==="
  SEED="$seed" \
  EPOCHS="$EPOCHS" \
  PATIENCE="$PATIENCE" \
  AUXILIARY_WEIGHT="$AUXILIARY_WEIGHT" \
  MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
  MAX_VALIDATION_SAMPLES="$MAX_VALIDATION_SAMPLES" \
  MAX_TEST_SAMPLES="$MAX_TEST_SAMPLES" \
  RUN_LABEL="$label" \
    bash scripts/run_cloud_p31_continuous_prompt.sh
  report="$(
    find results/cloud_p31_continuous_prompt -type f \
      -path "*_${label}/p31_report.json" \
      | sort \
      | tail -n 1
  )"
  if [[ -z "$report" ]]; then
    echo "Missing report for seed $seed." >&2
    exit 5
  fi
  reports+=("$report")
done

run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="results/cloud_p312_robustness/${run_id}_${RUN_PREFIX}"
"$PYTHON_BIN" scripts/summarize_p31_robustness.py \
  --reports "${reports[@]}" \
  --output-dir "$output_dir"

echo "RETURN THIS FILE: $ROOT/$output_dir/p312_robustness_report.json"
echo "MARKDOWN SUMMARY: $ROOT/$output_dir/p312_robustness_summary.md"
