#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${PBS_O_WORKDIR:-$PWD}}"
cd "$PROJECT_DIR"

module load Python/3.10.8-GCCcore-12.2.0

VENV_PATH="${VENV_PATH:-$HOME/omni-hc-env}"
PYTHON="$VENV_PATH/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found or not executable: $PYTHON" >&2
    exit 2
fi

source "$VENV_PATH/bin/activate"
hash -r

DEVICE="${DEVICE:-auto}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/outputs/navier_stokes/mean_correction/transolver/final/seed_42/resolved_config.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_DIR/outputs/navier_stokes/mean_correction/transolver/final/seed_42/latest.pt}"
DATA_PATH="${DATA_PATH:-$HOME/omni_hc/data}"
OUT_ROOT="${OUT_ROOT:-$HOME/omni-hc-resume-results/${PBS_JOBID:-manual}}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

mkdir -p "$OUT_ROOT"

echo "Resume job started: $(date -Is)"
echo "Host: $(hostname)"
echo "Job ID: ${PBS_JOBID:-manual}"
echo "Project dir: $PROJECT_DIR"
echo "Output dir: $OUT_ROOT"
echo "Python: $PYTHON"
"$PYTHON" -c "import sys; print(sys.executable); print(sys.prefix)"
"$PYTHON" --version

if [ ! -f scripts/train.py ]; then
    echo "ERROR: scripts/train.py was not found in $PROJECT_DIR" >&2
    exit 2
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: resolved config does not exist: $CONFIG_PATH" >&2
    exit 2
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: checkpoint does not exist: $CHECKPOINT_PATH" >&2
    exit 2
fi

if [ ! -e "$DATA_PATH" ]; then
    echo "ERROR: DATA_PATH does not exist: $DATA_PATH" >&2
    echo "Set DATA_PATH to the symlink or data directory expected by the repo." >&2
    exit 2
fi

if [ -L "$DATA_PATH" ]; then
    echo "Data symlink: $DATA_PATH -> $(readlink "$DATA_PATH")"
else
    echo "Data path: $DATA_PATH"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

IFS=' ' read -r -a EXTRA_ARGS <<< "$TRAIN_EXTRA_ARGS"

"$PYTHON" scripts/train.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT_ROOT/resume-navier-stokes-transolver-seed42.log"

touch "$OUT_ROOT/RESUME_TRAINING_FINISHED"
echo "Resume job finished: $(date -Is)"
echo "Resume log and sentinel: $OUT_ROOT"
