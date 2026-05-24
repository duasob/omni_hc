#!/usr/bin/env bash
set -euo pipefail

# Sequential HPC runner.
#
# Define one training run per entry in RUNS below. Each entry is passed directly
# to `scripts/train.py`.
#
# Example entry:
#   "--benchmark plasticity --backbone FNO --constraint plasticity_mesh_consistency_constraint --budget smoke --override wandb_logging.image_log_every=10"
#
# Optional environment overrides:
#   PROJECT_DIR=/path/to/repo
#   VENV_PATH=$HOME/omni-hc-env
#   DEVICE=auto
#   DATA_PATH=$HOME/omni_hc/data
#   OUT_ROOT=$HOME/omni-hc-results/$PBS_JOBID
#   RUNS_FILE=/path/to/runs.txt
#   DRY_RUN=1

if type module >/dev/null 2>&1; then
    module load Python/3.10.8-GCCcore-12.2.0 || true
fi

PROJECT_DIR="${PROJECT_DIR:-${PBS_O_WORKDIR:-$PWD}}"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$HOME/omni-hc-env}"
PYTHON="${PYTHON:-$VENV_PATH/bin/python}"
DEVICE="${DEVICE:-auto}"
DATA_PATH="${DATA_PATH:-$HOME/omni_hc/data}"
OUT_ROOT="${OUT_ROOT:-$HOME/omni-hc-results/${PBS_JOBID:-manual}}"
RESULT_DIRS="${RESULT_DIRS:-outputs results runs checkpoints wandb logs}"
REQUIRE_RESULT_ARTIFACT="${REQUIRE_RESULT_ARTIFACT:-0}"
DRY_RUN="${DRY_RUN:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

RUNS=(
    "--benchmark darcy --backbone Transolver --constraint none --budget final"
)

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

slugify() {
    printf '%s' "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9._-]+/_/g; s/^_+//; s/_+$//' \
        | cut -c 1-120
}

print_header() {
    echo "Job started: $(timestamp)"
    echo "Host: $(hostname)"
    echo "Job ID: ${PBS_JOBID:-manual}"
    echo "Project dir: $PROJECT_DIR"
    echo "Output dir: $OUT_ROOT"
    echo "Python: $PYTHON"
    "$PYTHON" --version
}

check_environment() {
    if [ ! -x "$PYTHON" ]; then
        echo "ERROR: Python not found or not executable: $PYTHON" >&2
        exit 2
    fi

    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
        hash -r
    fi

    if [ ! -f scripts/train.py ] || [ ! -f scripts/test.py ]; then
        echo "ERROR: scripts/train.py or scripts/test.py was not found in $PROJECT_DIR" >&2
        exit 2
    fi

    if [ ! -e "$DATA_PATH" ]; then
        echo "ERROR: DATA_PATH does not exist: $DATA_PATH" >&2
        echo "Set DATA_PATH to the symlink or data directory expected by the repo." >&2
        exit 2
    fi

    mkdir -p "$OUT_ROOT"

    if [ -L "$DATA_PATH" ]; then
        echo "Data symlink: $DATA_PATH -> $(readlink "$DATA_PATH")"
    else
        echo "Data path: $DATA_PATH"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi || true
    fi
}

load_run_lines() {
    if [ -n "${RUNS_FILE:-}" ]; then
        if [ ! -f "$RUNS_FILE" ]; then
            echo "ERROR: RUNS_FILE does not exist: $RUNS_FILE" >&2
            exit 2
        fi
        mapfile -t RUN_LINES < "$RUNS_FILE"
    else
        RUN_LINES=("${RUNS[@]}")
    fi
}

run_one() {
    local index="$1"
    local run_args="$2"
    local marker log_name artifact_count count dir rendered_cmd script
    local -a args run_cmd

    if [ -z "${run_args// }" ] || [[ "$run_args" =~ ^[[:space:]]*# ]]; then
        return 0
    fi

    # Parse optional command prefix: test: or train: (default: train)
    if [[ "$run_args" =~ ^[[:space:]]*test:[[:space:]]*(.*) ]]; then
        script="scripts/test.py"
        run_args="${BASH_REMATCH[1]}"
    elif [[ "$run_args" =~ ^[[:space:]]*train:[[:space:]]*(.*) ]]; then
        script="scripts/train.py"
        run_args="${BASH_REMATCH[1]}"
    else
        script="scripts/train.py"
    fi

    if [ -z "${run_args// }" ]; then
        return 0
    fi

    read -r -a args <<< "$run_args"
    marker="$OUT_ROOT/start-run-${index}-$(date +%s)"
    touch "$marker"

    log_name="$(slugify "run_${index}_${run_args}")"
    if [ -z "$log_name" ]; then
        log_name="run_${index}"
    fi

    run_cmd=(
        "$PYTHON" "$script"
        --device "$DEVICE"
        "${args[@]}"
    )

    echo
    echo "Run $index started: $(timestamp)"
    printf 'Script: %s\n' "$script"
    printf 'Args: %s\n' "$run_args"

    if [ "$DRY_RUN" = "1" ]; then
        printf -v rendered_cmd '%q ' "${run_cmd[@]}"
        printf 'DRY RUN: %s\n' "$rendered_cmd"
    else
        "${run_cmd[@]}" 2>&1 | tee "$OUT_ROOT/${log_name}.log"
    fi

    artifact_count=0
    for dir in $RESULT_DIRS; do
        if [ -d "$dir" ]; then
            count="$(find "$dir" -type f -newer "$marker" | wc -l | tr -d ' ')"
            artifact_count=$((artifact_count + count))
        fi
    done

    if [ "$DRY_RUN" != "1" ] && [ "$REQUIRE_RESULT_ARTIFACT" = "1" ] && [ "$artifact_count" -eq 0 ]; then
        echo "ERROR: run $index finished but no new result artifacts were found." >&2
        echo "Checked RESULT_DIRS='$RESULT_DIRS' relative to $PROJECT_DIR." >&2
        exit 3
    fi

    echo "Run $index finished: $(timestamp); new artifacts: $artifact_count"
}

check_environment
print_header
load_run_lines

run_count=0
for run_args in "${RUN_LINES[@]}"; do
    run_count=$((run_count + 1))
    run_one "$run_count" "$run_args"
done

touch "$OUT_ROOT/RUNNER_FINISHED"
echo
echo "All runs finished: $(timestamp)"
echo "Logs and sentinel: $OUT_ROOT"
