#!/bin/bash

###############################################################################
# RunPod Training Script
###############################################################################

set -euo pipefail

###############################################################################
# Arguments
###############################################################################
GPU_ALLOCATION=${1:-"two_learn_two_infer"}
EXPERIMENT_NAME=${2:-"runpod_32b"}

case "${GPU_ALLOCATION}" in
    two_learn_two_infer)
        NUM_GPUS=2
        ;;
    two_learn_two_infer_shared)
        NUM_GPUS=2
        ;;
    four_learn_four_infer_shared)
        NUM_GPUS=4
        ;;
    four_learn_four_infer)
        NUM_GPUS=4
        ;;
    *)
        echo "Unknown GPU allocation: ${GPU_ALLOCATION}"
        exit 1
        ;;
esac

MODEL_CONFIG="qwen_2_5_32b_train"

###############################################################################
# Load credentials
###############################################################################
if [ -f "/workspace/ml-loop.env" ]; then
    source /workspace/ml-loop.env
elif [ -f "./ml-loop.env" ]; then
    source ./ml-loop.env
else
    echo "ERROR: ml-loop.env not found."
    exit 1
fi

###############################################################################
# Paths
###############################################################################
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

cd "${PROJECT_ROOT}"

export APPWORLD_ROOT="/workspace/appworld_data"

export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

mkdir -p "${HF_HOME}"
mkdir -p "/workspace/logs"
mkdir -p "/workspace/checkpoints"

###############################################################################
# Environment
###############################################################################
export WANDB_ENABLE=True
export WANDB_MODE=${WANDB_MODE:-online}

export HF_HUB_DISABLE_TELEMETRY=1

export RAY_USAGE_STATS_ENABLED=0
export VLLM_NO_USAGE_STATS=1

export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

export PATH="${HOME}/.local/bin:${PATH}"
export PATH="${PROJECT_ROOT}/appworld-env/bin:${PATH}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

###############################################################################
# Activate Poetry
###############################################################################
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

###############################################################################
# Diagnostics
###############################################################################
echo "======================================================"
echo "HOSTNAME                : ${HOSTNAME}"
echo "PROJECT                 : ${PROJECT_ROOT}"
echo "Experiment              : ${EXPERIMENT_NAME}"
echo "GPU Allocation          : ${GPU_ALLOCATION}"
echo "Processes               : ${NUM_GPUS}"

echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

echo
echo "Python"
which python
python --version

echo
echo "Poetry"
poetry --version

echo
echo "Accelerate"
poetry run accelerate env

echo
echo "GPU"
nvidia-smi

echo
echo "Torch Diagnostics"

poetry run python <<EOF
import os
import torch

print("Torch Version :", torch.__version__)
print("CUDA Version  :", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

print("Device Count :", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
EOF

echo "======================================================"

###############################################################################
# NCCL Health Check
###############################################################################
echo "Running NCCL health check..."

poetry run python scripts/torch_dist_healthcheck.py

###############################################################################
# GPU Utilisation Logging
###############################################################################
mkdir -p /workspace/logs

nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw \
    --format=csv \
    -l 30 \
    > /workspace/logs/${EXPERIMENT_NAME}_gpu.csv &

MONITOR_PID=$!

cleanup() {
    kill ${MONITOR_PID} 2>/dev/null || true
}

trap cleanup EXIT

###############################################################################
# Launch Training
###############################################################################
echo
echo "======================================================"
echo "Launching Training..."
echo "======================================================"

poetry run accelerate launch \
    --config_file ./phi_agents/rl/conf/accelerate_config.yaml \
    --num_processes=${NUM_GPUS} \
    ./phi_agents/rl/train.py \
    +global@_global_=appworld \
    rl/gpu_allocation=${GPU_ALLOCATION} \
    llm=${MODEL_CONFIG} \
    experiment_name=${EXPERIMENT_NAME} \
    rl.cloud_path=/workspace/checkpoints/${EXPERIMENT_NAME} \
    wandb.run=${EXPERIMENT_NAME} \
    wandb.enable=${WANDB_ENABLE} \
    wandb.group="RunPod-32B" \
    wandb.project="ml-loop" \
    rl.params.total_iterations=90 \
    rl.max_ckpts=90 \
    rl.scenario_runner.appworld_config.env.max_interactions=40 \
    rl.eval.overrides.rl.scenario_runner.appworld_config.env.max_interactions=40 \
    rl.num_scenario_runners=32 \
    rl.params.scenarios_per_iteration=40 \
    rl.params.minibatch_size=32 \
    rl.params.rollouts_per_scenario=6 \
    rl.scenario_sampler.dataset_name=train_difficulty_1_and_train_difficulty_2 \
    rl.learning_max_seq_len=32000 \
    rl.rollouts_fraction=0.9 \
    rl.rollouts_per_scenario_fraction=0.75 \
    2>&1 | tee /workspace/logs/${EXPERIMENT_NAME}.log