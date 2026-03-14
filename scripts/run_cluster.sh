#!/bin/bash
set -euo pipefail

# Multi-GPU cluster training via torchrun
#
# Usage:
#   ./scripts/run_cluster.sh [run-name] [extra args...]
#   ./scripts/run_cluster.sh surgcast_full --experiment configs/experiment/full.yaml
#   NGPU=4 ./scripts/run_cluster.sh surgcast_full --stage full
#
# Environment variables:
#   NGPU            Number of GPUs (default: 2)
#   FEATURES_ROOT   Path to HDF5 features dir (default: /yuming/data/surgcast/features)
#   NPZ_ROOT        Path to NPZ labels dir (default: /yuming/data/surgcast/npz)
#   REGISTRY        Path to registry.json (default: data/registry.json)

NGPU="${NGPU:-2}"
FEATURES_ROOT="${FEATURES_ROOT:-/yuming/data/surgcast/features}"
NPZ_ROOT="${NPZ_ROOT:-/yuming/data/surgcast/npz}"
REGISTRY="${REGISTRY:-data/registry.json}"

torchrun --nproc_per_node="$NGPU" scripts/train.py \
    --config-data configs/data/default.yaml \
    --config-model configs/model/default.yaml \
    --config-train configs/train/default.yaml configs/train/cluster.yaml \
    --registry "$REGISTRY" \
    --features-root "$FEATURES_ROOT" \
    --npz-root "$NPZ_ROOT" \
    --run-name "${1:-surgcast_full}" \
    "${@:2}"
