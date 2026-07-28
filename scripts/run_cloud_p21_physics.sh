#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PHYSICS_GUIDED=1
export ADAPTER_EPOCHS="${ADAPTER_EPOCHS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-0.0005}"
export LOCAL_WEIGHT="${LOCAL_WEIGHT:-0.2}"
export PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-1.0}"
export ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.1}"

exec bash "$ROOT/scripts/run_cloud_p2_local.sh"
