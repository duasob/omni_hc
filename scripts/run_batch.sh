#!/usr/bin/env bash
set -euo pipefail

# Sequential local batch runner.
#
# Reads the same line format as scripts/hpc/runner.sh:
#   test: --config outputs/.../resolved_config.yaml --checkpoint outputs/.../best.pt
#   train: --benchmark darcy --backbone Transolver --constraint none --budget final
#   diagnose: --config outputs/.../resolved_config.yaml --checkpoint outputs/.../best.pt --write-yaml
#
# Optional environment overrides:
#   RUNS_FILE=/path/to/runs.txt
#   CONDA_ENV=omni-hc
#   PYTHON=/path/to/python  # or "python", "python3", etc.
#   DEVICE=auto
#   OUT_ROOT=artifacts/batch_runs/manual
#   DRY_RUN=1
#   CHECK_REPORTING_FINGERPRINT=0

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

RUNS_FILE="${RUNS_FILE:-${1:-}}"
CONDA_ENV="${CONDA_ENV:-omni-hc}"
DEVICE="${DEVICE:-auto}"
OUT_ROOT="${OUT_ROOT:-artifacts/batch_runs/manual}"
DRY_RUN="${DRY_RUN:-0}"
CHECK_REPORTING_FINGERPRINT="${CHECK_REPORTING_FINGERPRINT:-1}"

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
    if [ ! -f scripts/train.py ] || [ ! -f scripts/test.py ] || [ ! -f scripts/diagnose.py ]; then
        echo "ERROR: scripts/train.py, scripts/test.py, or scripts/diagnose.py was not found in $PROJECT_DIR" >&2
        exit 2
    fi
    if [ -n "${PYTHON:-}" ]; then
        if [[ "$PYTHON" == */* ]] && [ ! -x "$PYTHON" ]; then
            echo "ERROR: PYTHON is not executable: $PYTHON" >&2
            exit 2
        fi
        if [[ "$PYTHON" != */* ]] && ! command -v "$PYTHON" >/dev/null 2>&1; then
            echo "ERROR: PYTHON command was not found: $PYTHON" >&2
            exit 2
        fi
    fi
    mkdir -p "$OUT_ROOT"
}

python_cmd() {
    if [ -n "${PYTHON:-}" ]; then
        printf '%s\n' "$PYTHON"
    elif command -v conda >/dev/null 2>&1; then
        printf '%s\n' "conda run -n $CONDA_ENV python"
    elif command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
    elif command -v python >/dev/null 2>&1; then
        printf '%s\n' "python"
    else
        echo "ERROR: no Python command found. Set PYTHON=/path/to/python." >&2
        exit 2
    fi
}

runfile_meta() {
    local key="$1"
    sed -n "s/^# ${key}: //p" "$RUNS_FILE" | head -n 1
}

check_reporting_fingerprint() {
    local expected chapter name actual
    local -a py_parts fingerprint_cmd

    if [ "$CHECK_REPORTING_FINGERPRINT" != "1" ]; then
        return 0
    fi

    expected="$(runfile_meta reporting_fingerprint)"
    if [ -z "$expected" ]; then
        return 0
    fi
    if [ ! -f scripts/reporting/build.py ]; then
        return 0
    fi

    chapter="$(runfile_meta reporting_chapter)"
    name="$(runfile_meta reporting_name)"
    read -r -a py_parts <<< "$(python_cmd)"
    fingerprint_cmd=("${py_parts[@]}" -m scripts.reporting.build --print-fingerprint)
    if [ -n "$chapter" ] && [ "$chapter" != "all" ]; then
        fingerprint_cmd+=(--chapter "$chapter")
    fi
    if [ -n "$name" ] && [ "$name" != "all" ]; then
        fingerprint_cmd+=(--name "$name")
    fi
    actual="$("${fingerprint_cmd[@]}")"
    if [ "$actual" != "$expected" ]; then
        echo "ERROR: RUNS_FILE was generated from an older reporting registry." >&2
        echo "Expected fingerprint: $expected" >&2
        echo "Current fingerprint:  $actual" >&2
        echo "Regenerate it with: python -m scripts.reporting.build --write-missing-runs missing_runs.txt" >&2
        echo "Set CHECK_REPORTING_FINGERPRINT=0 to bypass this check." >&2
        exit 2
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
    elif [[ "$run_args" =~ ^[[:space:]]*diagnose:[[:space:]]*(.*) ]]; then
        script="scripts/diagnose.py"
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
check_reporting_fingerprint

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
