#!/bin/bash
# Run script for RunPod.
# This script should be executed from the project root inside the RunPod container.
# Usage: bash scripts/run_loop_runpod.sh [allocation] [experiment_name]
#   [allocation] can be one of:
#     - two_learn_two_infer (default, uses 2 GPUs for training, 2 for inference, no restarts)
#     - four_learn_four_infer_shared (uses all 4 GPUs for training and inference, concurrency tuned for H200 141GB)
#     - two_learn_two_infer_shared (uses 2 GPUs for training and inference)
#     - four_learn_four_infer (uses 4 GPUs for training, 4 for inference, requires 8 GPUs total)
#   [experiment_name] defaults to "runpod_appworld_run"

# Exit on any error
set -euo pipefail

# Parse arguments
ALLOCATION=${1:-"two_learn_two_infer"}
EXPERIMENT_NAME=${2:-"runpod_appworld_run"}

EXTRA_ARGS=""
if [ "$ALLOCATION" = "two_learn_two_infer" ]; then
    NUM_GPUS=2
elif [ "$ALLOCATION" = "four_learn_four_infer_shared" ]; then
    NUM_GPUS=4
    # Override memory bounds so FSDP training and vLLM inference fit concurrently on H200 (141 GB VRAM)
    EXTRA_ARGS="rl.inference_requires_memory_gb=65 rl.learning_requires_memory_gb=65"
elif [ "$ALLOCATION" = "two_learn_two_infer_shared" ]; then
    NUM_GPUS=2
elif [ "$ALLOCATION" = "four_learn_four_infer" ]; then
    NUM_GPUS=4
else
    echo "ERROR: Invalid GPU allocation '$ALLOCATION'." >&2
    echo "Available options: two_learn_two_infer, four_learn_four_infer_shared, two_learn_two_infer_shared, four_learn_four_infer" >&2
    exit 1
fi

echo "========================================================"
echo "Starting RunPod Execution for ml-loop..."
echo "Selected Allocation: $ALLOCATION ($NUM_GPUS training processes)"
echo "Experiment Name    : $EXPERIMENT_NAME"
[ -n "$EXTRA_ARGS" ] && echo "Memory Tuning      : Concurrency enabled ($EXTRA_ARGS)"
echo "========================================================"

# Determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# ==============================================================================
# CREDENTIALS SETUP
# ==============================================================================
SECRETS_FILE="/workspace/ml-loop.env"
if [ -f "$SECRETS_FILE" ]; then
    echo "Loading credentials from $SECRETS_FILE..."
    source "$SECRETS_FILE"
elif [ -f "${PROJECT_ROOT}/ml-loop.env" ]; then
    echo "Loading credentials from ${PROJECT_ROOT}/ml-loop.env..."
    source "${PROJECT_ROOT}/ml-loop.env"
fi

if [ -z "${HF_TOKEN:-}" ] || [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: Environment variables HF_TOKEN and WANDB_API_KEY must be set." >&2
    echo "       Please export them or configure /workspace/ml-loop.env." >&2
    exit 1
fi

# ==============================================================================
# ENVIRONMENT VARIABLES & CACHING
# ==============================================================================
export WANDB_ENABLE=True
export WANDB_MODE=${WANDB_MODE:-"online"}
export WANDB_RESUME="allow"

# Keep all Hugging Face assets on the persistent volume
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
export HF_HOME="/workspace/.cache/huggingface"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
export HF_HUB_DISABLE_TELEMETRY=1

export RAY_USAGE_STATS_ENABLED=0
export VLLM_NO_USAGE_STATS=1

export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

# NCCL Environment Tuning for H200 Node
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_BLOCKING_WAIT=1

# Setup workspace directories
export APPWORLD_ROOT="/workspace/appworld_data"
mkdir -p "${HF_HOME}"
mkdir -p "/workspace/logs"
mkdir -p "/workspace/checkpoints"

# Paths configuration
export PATH="${PROJECT_ROOT}/appworld-env/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Activate the main poetry virtualenv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ==============================================================================
# HEALTHCHECK
# ==============================================================================
echo "Checking GPU / NCCL distributed communication health..."
if python scripts/torch_dist_healthcheck.py; then
    echo "NCCL healthcheck successful!"
else
    echo "ERROR: NCCL healthcheck failed. Terminating to prevent wasteful compute charges." >&2
    exit 1
fi

echo
echo "======================================================"
echo "HOSTNAME                : ${HOSTNAME:-}"
echo "CUDA_VISIBLE_DEVICES    : ${CUDA_VISIBLE_DEVICES:-all}"
echo "Python                  : $(which python) ($(python --version))"
echo "Poetry                  : $(poetry --version 2>/dev/null || echo 'not in PATH')"
echo "======================================================"
echo

# ==============================================================================
# BACKGROUND GPU PERFORMANCE MONITORING
# ==============================================================================
echo "Starting background GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu \
    --format=csv -l 30 > "/workspace/logs/${EXPERIMENT_NAME}_gpu_utilization.csv" &
MONITOR_PID=$!

# Ensure the background monitoring process is killed when this script exits
cleanup() {
    echo "Cleaning up background monitoring process (PID: $MONITOR_PID)..."
    kill $MONITOR_PID || true
}
trap cleanup EXIT

# ==============================================================================
# LAUNCH TRAINING LOOP (WITH AUTO-RESUME AND LOCAL PERSISTENT LOGGING)
# ==============================================================================
echo "Launching training... Output is being piped to /workspace/logs/${EXPERIMENT_NAME}.log"
# Inject $EXTRA_ARGS if present for memory tuning
poetry run accelerate launch \
    --config_file ./phi_agents/rl/conf/accelerate_config.yaml \
    --num_processes=${NUM_GPUS} \
    ./phi_agents/rl/train.py \
    +global@_global_=appworld \
    rl/gpu_allocation=${ALLOCATION} \
    llm=qwen_2_5_32b_train \
    experiment_name=${EXPERIMENT_NAME} \
    rl.cloud_path=/workspace/checkpoints/${EXPERIMENT_NAME} \
    wandb.run=${EXPERIMENT_NAME} \
    ${EXTRA_ARGS} \
    wandb.enable=${WANDB_ENABLE} \
    wandb.group="32B_runpod" \
    wandb.project="ml-loop" \
    rl.params.total_iterations=90 \
    rl.max_ckpts=5 \
    rl.scenario_runner.appworld_config.env.max_interactions=40 \
    rl.eval.overrides.rl.scenario_runner.appworld_config.env.max_interactions=40 \
    rl.num_scenario_runners=32 \
    rl.params.scenarios_per_iteration=40 \
    rl.params.minibatch_size=32 \
    rl.params.rollouts_per_scenario=6 \
    rl.scenario_sampler.dataset_name=train_difficulty_1_and_train_difficulty_2 \
    rl.learning_max_seq_len=32000 \
    rl.rollouts_fraction=0.9 \
    rl.rollouts_per_scenario_fraction=0.75 2>&1 | tee "/workspace/logs/${EXPERIMENT_NAME}.log"
