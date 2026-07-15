#!/bin/bash --login
# ==============================================================================
# Slurm Job Submission Script for ml-loop on University of Manchester CSF3.
# Submit this script using: sbatch scripts/submit_loop_csf3.sbatch [allocation] [exp_name]
# ==============================================================================

# SBATCH Directives
#SBATCH --job-name=ml_loop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2               # Request 2 GPUs (adjust based on allocation)
#SBATCH --nodes=1                  # Single node execution
#SBATCH --ntasks-per-node=1        # Run one primary launch task
#SBATCH --cpus-per-task=8          # Adjust CPU allocation (typically 4-8 per GPU)
#SBATCH --time=24:00:00            # Walltime limit (HH:MM:SS)
#SBATCH --output=logs/ml_loop_%j.out
#SBATCH --error=logs/ml_loop_%j.err

set -euo pipefail

# ==============================================================================
# Arguments & Configurations
# ==============================================================================
GPU_ALLOCATION=${1:-"two_learn_two_infer"}
EXPERIMENT_NAME=${2:-"csf3_32b"}
MODEL_CONFIG="qwen_2_5_32b_train"

case "${GPU_ALLOCATION}" in
    two_learn_two_infer | two_learn_two_infer_shared)
        NUM_GPUS=2
        ;;
    four_learn_four_infer_shared | four_learn_four_infer)
        NUM_GPUS=4
        ;;
    *)
        # Try to auto-detect from Slurm environment or CUDA_VISIBLE_DEVICES
        if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
            NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr -cd ',' | wc -c)
            NUM_GPUS=$((NUM_GPUS + 1))
        elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
            NUM_GPUS="${SLURM_GPUS_ON_NODE}"
        else
            echo "Unknown GPU allocation ${GPU_ALLOCATION}. Defaulting to 2 GPUs."
            NUM_GPUS=2
        fi
        ;;
esac

# ==============================================================================
# Paths & Quota Settings (scratch space)
# ==============================================================================
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${PROJECT_ROOT}"

SCRATCH_DIR="${HOME}/scratch"
export APPWORLD_ROOT="${SCRATCH_DIR}/appworld_data"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

mkdir -p "${SCRATCH_DIR}/logs"
mkdir -p "${SCRATCH_DIR}/checkpoints"

# ==============================================================================
# Load Credentials
# ==============================================================================
if [ -f "${PROJECT_ROOT}/ml-loop.env" ]; then
    source "${PROJECT_ROOT}/ml-loop.env"
else
    echo "ERROR: ml-loop.env not found in project root."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ] || [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: HF_TOKEN and WANDB_API_KEY must be set in ml-loop.env."
    exit 1
fi

# ==============================================================================
# Web Proxy Setup for Outbound Internet (WandB, HuggingFace Hub)
# Checks if a proxy is already defined (e.g. via ml-loop.env or shell environment)
# for SSH reverse tunneling SOCKS proxies. Otherwise, checks for direct internet
# or falls back to the university web proxy.
# ==============================================================================
if [ -n "${http_proxy:-}" ]; then
    echo "Using pre-configured proxy: ${http_proxy}"
    export https_proxy="${https_proxy:-$http_proxy}"
elif curl -s --connect-timeout 3 https://huggingface.co > /dev/null; then
    echo "Direct internet connection detected. Bypassing proxy."
else
    echo "Direct connection to Hugging Face failed. Testing university web proxy..."
    if curl -s --connect-timeout 3 -x http://webproxy.its.manchester.ac.uk:3128 https://huggingface.co > /dev/null; then
        echo "University web proxy is reachable. Configuring proxy settings..."
        export http_proxy=http://webproxy.its.manchester.ac.uk:3128
        export https_proxy=http://webproxy.its.manchester.ac.uk:3128
    else
        echo "WARNING: Neither direct internet nor university proxy was reachable. Connection issues may occur."
    fi
fi

# Exclude local addresses to prevent Ray/vLLM communication from going through proxy
export no_proxy="localhost,127.0.0.1,$(hostname -i 2>/dev/null || echo ''),$(hostname)"
export NO_PROXY="${no_proxy}"

# ==============================================================================
# Environment Configuration
# ==============================================================================
export WANDB_ENABLE=True
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HUB_DISABLE_TELEMETRY=1
export RAY_USAGE_STATS_ENABLED=0
export VLLM_NO_USAGE_STATS=1

export PATH="${HOME}/.local/bin:${PATH}"
export PATH="${PROJECT_ROOT}/appworld-env/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ==============================================================================
# Load Software Modules
# ==============================================================================
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1 || declare -f module >/dev/null; then
    echo "Loading modules..."
    module load anaconda3/2023.09 || module load anaconda3/2022.10 || echo "WARNING: Anaconda module load failed."
    module load cuda/12.1 || module load cuda/12.4 || echo "WARNING: CUDA module load failed."
fi

# ==============================================================================
# Activate virtual environment
# ==============================================================================
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ==============================================================================
# Diagnostics
# ==============================================================================
echo "======================================================"
echo "HOSTNAME                : ${HOSTNAME}"
echo "PROJECT                 : ${PROJECT_ROOT}"
echo "Experiment              : ${EXPERIMENT_NAME}"
echo "GPU Allocation          : ${GPU_ALLOCATION}"
echo "Processes               : ${NUM_GPUS}"
echo "Job ID                  : ${SLURM_JOB_ID:-N/A}"
echo "CUDA_VISIBLE_DEVICES    : ${CUDA_VISIBLE_DEVICES:-all}"

echo
echo "Python:"
which python
python --version

echo
echo "Poetry:"
poetry --version

echo
echo "Accelerate:"
poetry run accelerate env || true

echo
echo "GPU:"
nvidia-smi || true

echo
echo "Torch Diagnostics:"
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

# ==============================================================================
# NCCL Health Check
# ==============================================================================
echo "Running NCCL health check..."
poetry run python scripts/torch_dist_healthcheck.py

# ==============================================================================
# GPU Utilisation Logging
# ==============================================================================
nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw \
    --format=csv \
    -l 30 \
    > "${SCRATCH_DIR}/logs/${EXPERIMENT_NAME}_gpu.csv" &

MONITOR_PID=$!

cleanup() {
    kill ${MONITOR_PID} 2>/dev/null || true
}

trap cleanup EXIT

# ==============================================================================
# Launch Training
# ==============================================================================
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
    rl.cloud_path=${SCRATCH_DIR}/checkpoints/${EXPERIMENT_NAME} \
    wandb.run=${EXPERIMENT_NAME} \
    wandb.enable=${WANDB_ENABLE} \
    wandb.group="CSF3-32B" \
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
    2>&1 | tee "${SCRATCH_DIR}/logs/${EXPERIMENT_NAME}.log"
