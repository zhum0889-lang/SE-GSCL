#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/3] Semantic guard without within-class ranking"
RUN_LABEL="no_rank" \
RANKING_WEIGHT=0 \
bash "$ROOT/scripts/run_cloud_p22_semantic_guard.sh"

echo "[2/3] Semantic guard with fixed fusion"
RUN_LABEL="fixed_fusion" \
ADAPTIVE_FUSION=0 \
LOCAL_WEIGHT=0.2 \
bash "$ROOT/scripts/run_cloud_p22_semantic_guard.sh"

echo "[3/3] Complete P2.2"
RUN_LABEL="full" \
bash "$ROOT/scripts/run_cloud_p22_semantic_guard.sh"
