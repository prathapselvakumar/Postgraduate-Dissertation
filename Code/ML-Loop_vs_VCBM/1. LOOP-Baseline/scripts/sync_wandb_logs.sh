#!/bin/bash
# Push offline W&B run logs (written during CSF3 training, WANDB_MODE=offline)
# to the cloud. Run this on the CSF3 LOGIN node (has internet), not on a
# compute node — compute nodes are firewalled off from wandb.ai.
#
# Usage (from the project root, e.g. "1. LOOP-Baseline/"):
#   bash scripts/sync_wandb_logs.sh
#
# Optionally restrict to one experiment's runs:
#   bash scripts/sync_wandb_logs.sh csf3_7b_2gpu_90iter

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WANDB_LOG_DIR="$PROJECT_ROOT/_wandb_logs/wandb"

if [ -f "$PROJECT_ROOT/ml-loop.env" ]; then
    source "$PROJECT_ROOT/ml-loop.env"
elif [ -f "$HOME/.secrets/ml-loop.env" ]; then
    source "$HOME/.secrets/ml-loop.env"
fi

if [ -z "${WANDB_API_KEY:-}" ] || [ "${WANDB_API_KEY:-}" = "dummy_key_for_offline_use" ]; then
    echo "WANDB_API_KEY is not set to a real key (ml-loop.env / ~/.secrets/ml-loop.env)." >&2
    echo "Set it there first, then re-run this script." >&2
    exit 1
fi

unset WANDB_MODE

FILTER="${1:-}"

if [ ! -d "$WANDB_LOG_DIR" ]; then
    echo "No wandb log dir found at $WANDB_LOG_DIR" >&2
    exit 1
fi

shopt -s nullglob
RUN_DIRS=("$WANDB_LOG_DIR"/offline-run-*"$FILTER"*)
shopt -u nullglob

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "No offline run directories matched under $WANDB_LOG_DIR (filter: '${FILTER:-<none>}')" >&2
    exit 1
fi

echo "Syncing ${#RUN_DIRS[@]} offline run(s) to wandb.ai:"
printf '  %s\n' "${RUN_DIRS[@]}"
echo

poetry run wandb sync "${RUN_DIRS[@]}"

echo
echo "Done."
