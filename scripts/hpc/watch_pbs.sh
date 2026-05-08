#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/hpc/watch_pbs_job.sh JOB_ID [INTERVAL_SECONDS]

Examples:
  scripts/hpc/watch_pbs_job.sh 2710135.pbs-7
  scripts/hpc/watch_pbs_job.sh 2710135.pbs-7 30

Optional environment:
  PBS_WATCH_NOTIFY_CMD
      Command run when the job finishes. The final status line is passed as
      one argument. Example:
        PBS_WATCH_NOTIFY_CMD='mail -s "PBS job finished" you@example.com'
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

JOB_ID="${1:-}"
INTERVAL_SECONDS="${2:-60}"

if [ -z "$JOB_ID" ]; then
    usage >&2
    exit 2
fi

if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || [ "$INTERVAL_SECONDS" -lt 1 ]; then
    echo "ERROR: interval must be a positive integer, got: $INTERVAL_SECONDS" >&2
    exit 2
fi

if ! command -v qstat >/dev/null 2>&1; then
    echo "ERROR: qstat is not available on PATH." >&2
    exit 2
fi

state_name() {
    case "$1" in
        B) printf "array-running" ;;
        E) printf "exiting-after-run" ;;
        F) printf "finished" ;;
        H) printf "held" ;;
        Q) printf "queued" ;;
        R) printf "running" ;;
        X) printf "subjob-completed-or-deleted" ;;
        *) printf "unknown" ;;
    esac
}

is_final_state() {
    case "$1" in
        F|X) return 0 ;;
        *) return 1 ;;
    esac
}

notify_finished() {
    local message="$1"
    printf '\a'
    if [ -n "${PBS_WATCH_NOTIFY_CMD:-}" ]; then
        # shellcheck disable=SC2086
        $PBS_WATCH_NOTIFY_CMD "$message" || true
    fi
}

last_state=""
last_line=""

echo "Watching PBS job $JOB_ID every ${INTERVAL_SECONDS}s. Press Ctrl-C to stop."

while true; do
    timestamp="$(date -Is)"
    qstat_output="$(qstat -x "$JOB_ID" 2>&1 || true)"

    if printf '%s\n' "$qstat_output" | grep -qiE 'Unknown Job Id|Unknown Job|not found|does not exist'; then
        message="$timestamp job=$JOB_ID state=unknown result='qstat no longer knows this job'"
        echo "$message"
        notify_finished "$message"
        exit 1
    fi

    job_line="$(printf '%s\n' "$qstat_output" | awk 'NR > 2 && NF {print; exit}')"
    if [ -z "$job_line" ]; then
        message="$timestamp job=$JOB_ID state=unknown result='no qstat row returned'"
        echo "$message"
        notify_finished "$message"
        exit 1
    fi

    state="$(printf '%s\n' "$job_line" | awk '{print $(NF-1)}')"
    queue="$(printf '%s\n' "$job_line" | awk '{print $NF}')"
    rendered_state="$(state_name "$state")"
    line="$timestamp job=$JOB_ID state=$state ($rendered_state) queue=$queue"

    if [ "$state" != "$last_state" ] || [ "$job_line" != "$last_line" ]; then
        echo "$line"
        last_state="$state"
        last_line="$job_line"
    fi

    if is_final_state "$state"; then
        notify_finished "$line"
        exit 0
    fi

    sleep "$INTERVAL_SECONDS"
done
