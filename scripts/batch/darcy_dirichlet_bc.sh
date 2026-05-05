#!/usr/bin/env bash
set -euo pipefail

BUDGET="${BUDGET:-debug}"
CONSTRAINT="${CONSTRAINT:-dirichlet_ansatz}"
DEVICE="${DEVICE:-auto}"

for BACKBONE in Galerkin_Transformer GNOT Factformer ONO Transolver; do
    python scripts/train.py \
        --benchmark darcy \
        --backbone "$BACKBONE" \
        --constraint "$CONSTRAINT" \
        --budget "$BUDGET" \
        --device "$DEVICE"
done
