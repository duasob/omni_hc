#!/usr/bin/env bash
set -euo pipefail

# Shared runner for the CPU and GPU PBS smoke-test wrappers.
#
# Submit from the training repository root:
#   qsub examples/omni_hc_smoke_test_cpu.pbs
#   qsub examples/omni_hc_smoke_test_gpu.pbs
#
# Useful overrides:
#   qsub -v DEVICE=cpu examples/omni_hc_smoke_test_cpu.pbs
#   qsub -v RESULT_DIRS=outputs examples/omni_hc_smoke_test_cpu.pbs
#   qsub -v TRAIN_EXTRA_ARGS="--max_epochs 1" examples/omni_hc_smoke_test_cpu.pbs
#
# To test multiple backbones, either edit SMOKE_BACKBONES below or export it
# before submission with `qsub -V`.

module load Python/3.10.8-GCCcore-12.2.0

VENV_PATH="${VENV_PATH:-$HOME/omni-hc-env}"
PYTHON="$VENV_PATH/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found or not executable: $PYTHON" >&2
    exit 2
fi

source "$VENV_PATH/bin/activate"
hash -r

echo "Python: $PYTHON"
"$PYTHON" -c "import sys; print(sys.executable); print(sys.prefix)"

PROJECT_DIR="${PROJECT_DIR:-${PBS_O_WORKDIR:-$PWD}}"
cd "$PROJECT_DIR"

BUDGET="${BUDGET:-smoke}"
CONSTRAINT="${CONSTRAINT:-dirichlet_ansatz_zero}"
DEVICE="${DEVICE:-auto}"
SMOKE_BACKBONES="${SMOKE_BACKBONES:-FNO}"
BENCHMARK="${BENCHMARK:-darcy}"
DATA_PATH="${DATA_PATH:-$HOME/omni_hc/data}"
RESULT_DIRS="${RESULT_DIRS:-outputs results runs checkpoints wandb logs}"
OUT_ROOT="${OUT_ROOT:-$HOME/omni-hc-smoke-results/${PBS_JOBID:-manual}}"
REQUIRE_RESULT_ARTIFACT="${REQUIRE_RESULT_ARTIFACT:-1}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

mkdir -p "$OUT_ROOT"

echo "Job started: $(date -Is)"
echo "Host: $(hostname)"
echo "Job ID: ${PBS_JOBID:-manual}"
echo "Project dir: $PROJECT_DIR"
echo "Output dir: $OUT_ROOT"
echo "Python: $(command -v "$PYTHON")"
"$PYTHON" --version

if [ ! -f scripts/train.py ]; then
    echo "ERROR: scripts/train.py was not found in $PROJECT_DIR" >&2
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

for BACKBONE in $SMOKE_BACKBONES; do
    [ -n "$BACKBONE" ] || continue

    echo "Smoke test started for backbone=$BACKBONE at $(date -Is)"
    marker="$OUT_ROOT/start-${BACKBONE}-$(date +%s)"
    touch "$marker"

    "$PYTHON" scripts/train.py \
        --benchmark "$BENCHMARK" \
        --backbone "$BACKBONE" \
        --constraint "$CONSTRAINT" \
        --budget "$BUDGET" \
        --device "$DEVICE" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$OUT_ROOT/train-${BACKBONE}.log"

    artifact_count=0
    for dir in $RESULT_DIRS; do
        if [ -d "$dir" ]; then
            count="$(find "$dir" -type f -newer "$marker" | wc -l | tr -d ' ')"
            artifact_count=$((artifact_count + count))
        fi
    done

    if [ "$REQUIRE_RESULT_ARTIFACT" = "1" ] && [ "$artifact_count" -eq 0 ]; then
        echo "ERROR: training finished but no new result artifacts were found." >&2
        echo "Checked RESULT_DIRS='$RESULT_DIRS' relative to $PROJECT_DIR." >&2
        echo "Set RESULT_DIRS to the directory your repo writes to, or set REQUIRE_RESULT_ARTIFACT=0." >&2
        exit 3
    fi

    echo "Smoke test passed for backbone=$BACKBONE; new artifacts found: $artifact_count"
done

touch "$OUT_ROOT/SMOKE_TEST_PASSED"
echo "Job finished: $(date -Is)"
echo "Smoke-test logs and sentinel: $OUT_ROOT"
