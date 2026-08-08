#!/bin/bash --login
#
#SBATCH --job-name=ml_loop
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/ml_loop_%j.out
#SBATCH --error=logs/ml_loop_%j.err

set -euo pipefail

echo "=================================================="
echo "Starting ml-loop on CSF3"
echo "=================================================="

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo

########################################################################
# Project directory
########################################################################

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

cd "$PROJECT_ROOT"

echo "Project Root : $PROJECT_ROOT"
pwd
echo

########################################################################
# Check project
########################################################################

if [ -f "ml-loop.env" ]; then
    echo "Loading ml-loop.env from project root"
    source ml-loop.env
elif [ -f "$HOME/.secrets/ml-loop.env" ]; then
    echo "Loading ml-loop.env from ~/.secrets"
    source "$HOME/.secrets/ml-loop.env"
else
    echo "ml-loop.env not found. Using offline defaults."
fi

# Ensure offline environment configuration
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -z "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="dummy_token_for_offline_use"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY="dummy_key_for_offline_use"
fi

# Prevent any outbound proxy requests
unset http_proxy
unset https_proxy

########################################################################
# Modules
########################################################################

module purge
module load cuda/12.6.2

echo "========== Environment =========="
module list
echo "PATH=$PATH"

echo
echo "CUDA:"
which nvcc
nvcc --version

echo
echo "GPU:"
nvidia-smi
echo "================================="

########################################################################
# Conda
########################################################################

source ~/scratch/miniconda3/etc/profile.d/conda.sh

conda activate ml-loop

# Ensure Poetry is on PATH (may not be set on compute nodes)
export PATH="$HOME/.local/bin:$PATH"

# NCCL settings to prevent silent hangs on multi-GPU jobs
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN

# Disable torchdynamo/torch.compile entirely on CSF3.
# vLLM has per-function @torch.compile decorators (e.g. get_masked_input_and_mask)
# that invoke Inductor, which tries to call nvcc. nvcc is not in PATH on CSF3
# compute nodes (it's a module). TORCHDYNAMO_DISABLE=1 prevents all compilation.
export TORCHDYNAMO_DISABLE=1

echo
echo "Python:"
which python
python --version

echo
echo "Poetry:"
which poetry
poetry --version


export SCRATCH_DIR="$HOME/scratch"

export APPWORLD_ROOT="$SCRATCH_DIR/appworld_data_loop"

export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"


mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

echo "APPWORLD_ROOT=$APPWORLD_ROOT"
echo "HF_HOME=$HF_HOME"
########################################################################
# GPU
########################################################################

echo
echo "GPU Information"
nvidia-smi

########################################################################
# Torch
########################################################################

echo
echo "Torch Test"

poetry run python - <<EOF
import torch

print("Torch:", torch.__version__)
print("CUDA :", torch.version.cuda)
print("Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
EOF

########################################################################
# NCCL Test
########################################################################

echo
echo "Running NCCL health check..."

poetry run python scripts/torch_dist_healthcheck.py

########################################################################
# Training
########################################################################
echo "Checking AppWorld..."
ls -lah "$APPWORLD_ROOT"
echo
echo "Launching Training..."
echo

poetry run accelerate launch \
    --config_file phi_agents/rl/conf/accelerate_config.yaml \
    --num_processes=2 \
    phi_agents/rl/train.py \
    +global@_global_=appworld \
    rl/gpu_allocation=two_shared \
    rl.async_rollouts=False \
    llm=qwen_2_5_7b_train \
    llm.vllm_server.eager_mode=true \
    llm.max_gpu_mem_utilization=0.4 \
    llm.vllm_server.max_model_len=16384 \
    rl.inference_requires_memory_gb=30 \
    rl.learning_requires_memory_gb=45 \
    experiment_name=csf3_7b_2gpu_90iter \
    rl.cloud_path=$HOME/scratch/checkpoints/csf3_7b_2gpu_90iter \
    wandb.project=ml-loop \
    wandb.group=CSF3 \
    wandb.run=csf3_7b_2gpu_90iter \
    rl.params.total_iterations=90 \
    rl.max_ckpts=90 \
    rl.params.scenarios_per_iteration=12 \
    rl.params.rollouts_per_scenario=4 \
    rl.params.minibatch_size=32 \
    rl.eval.eval_every_n_iterations=5 \
    rl.eval.overrides.rl.num_scenario_runners=12 \
    rl.eval.overrides.rl.params.scenarios_per_iteration=10

