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
#   RUNS_FILES="a.txt b.txt c.txt"   # space-separated; runs are concatenated
#   bash run_batch.sh a.txt b.txt    # positional args also accepted
#   CONDA_ENV=omni-hc
#   PYTHON=/path/to/python  # or "python", "python3", etc.
#   DEVICE=auto
#   OUT_ROOT=artifacts/batch_runs/manual
#   DRY_RUN=1
#   CHECK_REPORTING_FINGERPRINT=0

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

RUNS_FILES_LIST=()
if [ -n "${RUNS_FILE:-}" ]; then
    RUNS_FILES_LIST+=("$RUNS_FILE")
fi
if [ -n "${RUNS_FILES:-}" ]; then
    # shellcheck disable=SC2206
    _runs_files_arr=( ${RUNS_FILES} )
    RUNS_FILES_LIST+=("${_runs_files_arr[@]}")
fi
if [ "$#" -gt 0 ]; then
    RUNS_FILES_LIST+=("$@")
fi

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
    local f
    if [ "${#RUNS_FILES_LIST[@]}" -eq 0 ]; then
        echo "ERROR: provide RUNS_FILE=/path/to/runs.txt, RUNS_FILES=\"a.txt b.txt\", or pass files as positional args." >&2
        exit 2
    fi
    for f in "${RUNS_FILES_LIST[@]}"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: runs file does not exist: $f" >&2
            exit 2
        fi
    done
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
    local runs_file="$2"
    sed -n "s/^# ${key}: //p" "$runs_file" | head -n 1
}

check_reporting_fingerprint() {
    local runs_file="$1"
    local expected chapter name actual
    local -a py_parts fingerprint_cmd

    if [ "$CHECK_REPORTING_FINGERPRINT" != "1" ]; then
        return 0
    fi

    expected="$(runfile_meta reporting_fingerprint "$runs_file")"
    if [ -z "$expected" ]; then
        return 0
    fi
    if [ ! -f scripts/reporting/build.py ]; then
        return 0
    fi

    chapter="$(runfile_meta reporting_chapter "$runs_file")"
    name="$(runfile_meta reporting_name "$runs_file")"
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
        echo "ERROR: $runs_file was generated from an older reporting registry." >&2
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

    local exit_code=0
    if [ "$DRY_RUN" = "1" ]; then
        printf -v rendered_cmd '%q ' "${run_cmd[@]}"
        printf 'DRY RUN: %s\n' "$rendered_cmd"
    else
        set +e
        "${run_cmd[@]}" 2>&1 | tee "$OUT_ROOT/${log_name}.log"
        exit_code=${PIPESTATUS[0]}
        set -e
    fi

    if [ "$exit_code" -ne 0 ]; then
        echo "Run $index FAILED (exit $exit_code): $(timestamp)" >&2
        FAILED_RUNS+=("$index: $run_args (exit $exit_code)")
    else
        echo "Run $index finished: $(timestamp)"
    fi
}

check_environment
for runs_file in "${RUNS_FILES_LIST[@]}"; do
    check_reporting_fingerprint "$runs_file"
done

echo "Batch started: $(timestamp)"
echo "Project dir: $PROJECT_DIR"
echo "Runs files: ${RUNS_FILES_LIST[*]}"
echo "Output dir: $OUT_ROOT"
echo "Python command: $(python_cmd)"

run_count=0
FAILED_RUNS=()
for runs_file in "${RUNS_FILES_LIST[@]}"; do
    echo
    echo "--- Loading runs from: $runs_file ---"
    while IFS= read -r run_args || [ -n "$run_args" ]; do
        if [ -z "${run_args// }" ] || [[ "$run_args" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        run_count=$((run_count + 1))
        run_one "$run_count" "$run_args"
    done < "$runs_file"
done

touch "$OUT_ROOT/RUNNER_FINISHED"
echo
echo "All runs finished: $(timestamp)"
echo "Total runs: $run_count, failed: ${#FAILED_RUNS[@]}"
if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
    echo "Failures:"
    for entry in "${FAILED_RUNS[@]}"; do
        echo "  - $entry"
    done
fi
