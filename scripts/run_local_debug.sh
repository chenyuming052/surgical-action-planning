#!/bin/bash
set -euo pipefail

# Local debug training — single GPU, small batch, fp32, 2 epochs
#
# Usage:
#   ./scripts/run_local_debug.sh [run-name] [extra args...]
#   ./scripts/run_local_debug.sh debug_cholec --experiment configs/experiment/cholec_only.yaml
#   ./scripts/run_local_debug.sh debug_test --override loss.lambda_hazard=0.5

FEATURES_ROOT="${FEATURES_ROOT:-/yuming/data/surgcast/features}"
NPZ_ROOT="${NPZ_ROOT:-/yuming/data/surgcast/npz}"
REGISTRY="${REGISTRY:-data/registry.json}"

python scripts/train.py \
    --config-data configs/data/default.yaml \
    --config-model configs/model/default.yaml \
    --config-train configs/train/default.yaml configs/train/local_debug.yaml \
    --registry "$REGISTRY" \
    --features-root "$FEATURES_ROOT" \
    --npz-root "$NPZ_ROOT" \
    --run-name "${1:-debug_test}" \
    "${@:2}"
