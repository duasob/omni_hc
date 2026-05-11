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
#   qsub -v BUDGET=debug examples/omni_hc_smoke_test_cpu.pbs
#   qsub -v BENCHMARK=navier_stokes,CONSTRAINT=mean_correction,SMOKE_BACKBONES="Galerkin_Transformer Transolver Factformer ONO GNOT" examples/omni_hc_smoke_test_gpu.pbs
#   qsub -v PARALLEL_BACKBONES=1,GPU_IDS="0 1 2",SMOKE_BACKBONES="Factformer ONO GNOT" scripts/hpc/gpu_parallel.pbs
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

BUDGET="${BUDGET:-final}"
CONSTRAINT="${CONSTRAINT:-mean_correction}"
DEVICE="${DEVICE:-auto}"
SMOKE_BACKBONES="${SMOKE_BACKBONES:-GNOT}"
BENCHMARK="${BENCHMARK:-navier_stokes}"
DATA_PATH="${DATA_PATH:-$HOME/omni_hc/data}"
RESULT_DIRS="${RESULT_DIRS:-outputs results runs checkpoints wandb logs}"
OUT_ROOT="${OUT_ROOT:-$HOME/omni-hc-smoke-results/${PBS_JOBID:-manual}}"
REQUIRE_RESULT_ARTIFACT="${REQUIRE_RESULT_ARTIFACT:-1}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
LATENT_MODULES="${LATENT_MODULES:-Galerkin_Transformer=blocks.-1.ln_3 Transolver=blocks.-1.ln_3 Factformer=blocks.-1.ln_3 ONO=blocks.-1.ln_3 GNOT=blocks.-1}"
PARALLEL_BACKBONES="${PARALLEL_BACKBONES:-0}"
GPU_IDS="${GPU_IDS:-}"
DRY_RUN="${DRY_RUN:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

mkdir -p "$OUT_ROOT"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

echo "Job started: $(timestamp)"
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

EXTRA_ARGS=()
if [ -n "$TRAIN_EXTRA_ARGS" ]; then
    IFS=' ' read -r -a EXTRA_ARGS <<< "$TRAIN_EXTRA_ARGS"
fi

latent_module_for_backbone() {
    local backbone="$1"
    local entry name module
    for entry in $LATENT_MODULES; do
        name="${entry%%=*}"
        module="${entry#*=}"
        if [ "$name" = "$backbone" ] && [ "$module" != "$entry" ]; then
            printf '%s\n' "$module"
            return 0
        fi
    done
    return 1
}

if [ "$PARALLEL_BACKBONES" = "1" ] && [ -z "$GPU_IDS" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        GPU_IDS="${CUDA_VISIBLE_DEVICES//,/ }"
    else
        GPU_IDS="0 1 2"
    fi
fi

run_backbone() {
    local BACKBONE="$1"
    local GPU_ID="${2:-}"
    local marker artifact_count count dir
    local LATENT_MODULE
    local -a LATENT_ARGS
    local -a TRAIN_CMD
    local rendered_cmd

    [ -n "$BACKBONE" ] || return 0

    echo "Smoke test started for backbone=$BACKBONE at $(timestamp)"
    if [ -n "$GPU_ID" ]; then
        echo "Assigning backbone=$BACKBONE to CUDA_VISIBLE_DEVICES=$GPU_ID"
    fi
    marker="$OUT_ROOT/start-${BACKBONE}-$(date +%s)"
    touch "$marker"

    LATENT_ARGS=()
    if LATENT_MODULE="$(latent_module_for_backbone "$BACKBONE")"; then
        echo "Using latent_module=$LATENT_MODULE for backbone=$BACKBONE"
        LATENT_ARGS=(
            --override constraint.mode=latent_head
            --override "constraint.latent_module=$LATENT_MODULE"
        )
    fi

    TRAIN_CMD=(
        "$PYTHON" scripts/train.py
        --benchmark "$BENCHMARK"
        --backbone "$BACKBONE"
        --constraint "$CONSTRAINT"
        --budget "$BUDGET"
        --device "$DEVICE"
    )
    if [ "${#LATENT_ARGS[@]}" -gt 0 ]; then
        TRAIN_CMD+=("${LATENT_ARGS[@]}")
    fi
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
        TRAIN_CMD+=("${EXTRA_ARGS[@]}")
    fi

    if [ "$DRY_RUN" = "1" ]; then
        printf -v rendered_cmd '%q ' "${TRAIN_CMD[@]}"
        if [ -n "$GPU_ID" ]; then
            printf '[%s] DRY RUN: CUDA_VISIBLE_DEVICES=%q %s\n' "$BACKBONE" "$GPU_ID" "$rendered_cmd"
        else
            printf '[%s] DRY RUN: %s\n' "$BACKBONE" "$rendered_cmd"
        fi
    elif [ -n "$GPU_ID" ]; then
        CUDA_VISIBLE_DEVICES="$GPU_ID" "${TRAIN_CMD[@]}" \
            2>&1 | sed "s/^/[$BACKBONE] /" | tee "$OUT_ROOT/train-${BACKBONE}.log"
    else
        "${TRAIN_CMD[@]}" 2>&1 | tee "$OUT_ROOT/train-${BACKBONE}.log"
    fi

    artifact_count=0
    for dir in $RESULT_DIRS; do
        if [ -d "$dir" ]; then
            count="$(find "$dir" -type f -newer "$marker" | wc -l | tr -d ' ')"
            artifact_count=$((artifact_count + count))
        fi
    done

    if [ "$DRY_RUN" != "1" ] && [ "$REQUIRE_RESULT_ARTIFACT" = "1" ] && [ "$artifact_count" -eq 0 ]; then
        echo "ERROR: training finished but no new result artifacts were found." >&2
        echo "Checked RESULT_DIRS='$RESULT_DIRS' relative to $PROJECT_DIR." >&2
        echo "Set RESULT_DIRS to the directory your repo writes to, or set REQUIRE_RESULT_ARTIFACT=0." >&2
        exit 3
    fi

    echo "Smoke test passed for backbone=$BACKBONE; new artifacts found: $artifact_count"
}

if [ "$PARALLEL_BACKBONES" = "1" ]; then
    IFS=' ' read -r -a BACKBONE_LIST <<< "$SMOKE_BACKBONES"
    IFS=' ' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
    if [ "${#GPU_ID_LIST[@]}" -lt "${#BACKBONE_LIST[@]}" ]; then
        echo "ERROR: PARALLEL_BACKBONES=1 requires at least one GPU id per backbone." >&2
        echo "       SMOKE_BACKBONES='$SMOKE_BACKBONES'" >&2
        echo "       GPU_IDS='$GPU_IDS'" >&2
        exit 2
    fi

    pids=()
    for index in "${!BACKBONE_LIST[@]}"; do
        BACKBONE="${BACKBONE_LIST[$index]}"
        GPU_ID="${GPU_ID_LIST[$index]}"
        (
            run_backbone "$BACKBONE" "$GPU_ID"
        ) &
        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "ERROR: at least one parallel backbone run failed." >&2
        exit 3
    fi
else
    for BACKBONE in $SMOKE_BACKBONES; do
        run_backbone "$BACKBONE"
    done
fi

touch "$OUT_ROOT/SMOKE_TEST_PASSED"
echo "Job finished: $(timestamp)"
echo "Smoke-test logs and sentinel: $OUT_ROOT"
