#!/usr/bin/env bash
set -euo pipefail

# Sequential local batch runner.
#
# Reads the same line format as scripts/hpc/runner.sh:
#   test: --config outputs/.../resolved_config.yaml --checkpoint outputs/.../best.pt
#   train: --benchmark darcy --backbone Transolver --constraint none --budget final
#
# Optional environment overrides:
#   RUNS_FILE=/path/to/runs.txt
#   CONDA_ENV=omni-hc
#   PYTHON=/path/to/python
#   DEVICE=auto
#   OUT_ROOT=artifacts/batch_runs/manual
#   DRY_RUN=1

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

RUNS_FILE="${RUNS_FILE:-${1:-}}"
CONDA_ENV="${CONDA_ENV:-omni-hc}"
DEVICE="${DEVICE:-auto}"
OUT_ROOT="${OUT_ROOT:-artifacts/batch_runs/manual}"
DRY_RUN="${DRY_RUN:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

slugify() {
    printf '%s' "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9._-]+/_/g; s/^_+//; s/_+$//' \
        | cut -c 1-120
}

check_environment() {
    if [ -z "$RUNS_FILE" ]; then
        echo "ERROR: provide RUNS_FILE=/path/to/runs.txt or pass the file as arg 1." >&2
        exit 2
    fi
    if [ ! -f "$RUNS_FILE" ]; then
        echo "ERROR: RUNS_FILE does not exist: $RUNS_FILE" >&2
        exit 2
    fi
    if [ ! -f scripts/train.py ] || [ ! -f scripts/test.py ]; then
        echo "ERROR: scripts/train.py or scripts/test.py was not found in $PROJECT_DIR" >&2
        exit 2
    fi
    if [ -n "${PYTHON:-}" ] && [ ! -x "$PYTHON" ]; then
        echo "ERROR: PYTHON is not executable: $PYTHON" >&2
        exit 2
    fi
    mkdir -p "$OUT_ROOT"
}

python_cmd() {
    if [ -n "${PYTHON:-}" ]; then
        printf '%s\n' "$PYTHON"
    else
        printf '%s\n' "conda run -n $CONDA_ENV python"
    fi
}

run_one() {
    local index="$1"
    local run_args="$2"
    local script log_name rendered_cmd
    local -a args run_cmd py_parts

    if [ -z "${run_args// }" ] || [[ "$run_args" =~ ^[[:space:]]*# ]]; then
        return 0
    fi

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
    read -r -a py_parts <<< "$(python_cmd)"
    run_cmd=("${py_parts[@]}" "$script" --device "$DEVICE" "${args[@]}")

    log_name="$(slugify "run_${index}_${run_args}")"
    if [ -z "$log_name" ]; then
        log_name="run_${index}"
    fi

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

    echo "Run $index finished: $(timestamp)"
}

check_environment

echo "Batch started: $(timestamp)"
echo "Project dir: $PROJECT_DIR"
echo "Runs file: $RUNS_FILE"
echo "Output dir: $OUT_ROOT"
echo "Python command: $(python_cmd)"

run_count=0
while IFS= read -r run_args || [ -n "$run_args" ]; do
    if [ -z "${run_args// }" ] || [[ "$run_args" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    run_count=$((run_count + 1))
    run_one "$run_count" "$run_args"
done < "$RUNS_FILE"

touch "$OUT_ROOT/RUNNER_FINISHED"
echo
echo "All runs finished: $(timestamp)"
