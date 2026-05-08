#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${PBS_O_WORKDIR:-$PWD}}"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/outputs/navier_stokes/mean_correction/transolver/final/seed_42/resolved_config.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_DIR/outputs/navier_stokes/mean_correction/transolver/final/seed_42/latest.pt}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

IFS=' ' read -r -a EXTRA_ARGS <<< "$TRAIN_EXTRA_ARGS"

"$PYTHON" scripts/train.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}"
