#!/bin/bash --login

###############################################################################
# SLURM
###############################################################################
#SBATCH --job-name=ml-loop-train
#SBATCH --partition=gpuA
#SBATCH --account=gpu-fse-ugpgt01

# Resources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00

#SBATCH --output=logs/ml_loop_%j.out
#SBATCH --error=logs/ml_loop_%j.err

set -euo pipefail

###############################################################################
# Load secrets
###############################################################################
if [ -f "${HOME}/.secrets/ml-loop.env" ]; then
    source "${HOME}/.secrets/ml-loop.env"
else
    echo "ERROR: ${HOME}/.secrets/ml-loop.env not found."
    exit 1
fi

###############################################################################
# Environment
###############################################################################
export WANDB_ENABLE=True
export WANDB_MODE=offline

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

export RAY_USAGE_STATS_ENABLED=0
export VLLM_NO_USAGE_STATS=1

export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

###############################################################################
# Directories
###############################################################################
PROJECT_ROOT="/net/scratch/t83821ps/ml-loop"

export APPWORLD_ROOT="/net/scratch/t83821ps/appworld_data"

export HF_HOME="/net/scratch/t83821ps/.cache/huggingface"

mkdir -p "${HF_HOME}"
mkdir -p "${PROJECT_ROOT}/logs"

###############################################################################
# Modules
###############################################################################
module purge

module load python/3.13.1
module load libs/cuda

###############################################################################
# PATHS
###############################################################################
export PATH="${HOME}/.local/bin:${PATH}"
export PATH="${PROJECT_ROOT}/appworld-env/bin:${PATH}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${PROJECT_ROOT}"

###############################################################################
# GPU Configuration
###############################################################################
NUM_GPUS=2

GPU_ALLOCATION="two_learn_two_infer_shared"

MODEL_CONFIG="qwen_2_5_32b_train"

###############################################################################
# Diagnostics
###############################################################################
echo "======================================================"
echo "HOSTNAME                : ${HOSTNAME:-}"
echo "SLURM_JOB_ID            : ${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST          : ${SLURM_NODELIST:-}"

echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"

echo
echo "Loaded Modules"
module list

echo
echo "Python"
which python
python --version

echo
echo "Poetry"
which poetry
poetry --version

echo
echo "Virtualenv Python"
poetry run which python

echo
echo "nvidia-smi"
nvidia-smi

echo
echo "Torch Diagnostics"

poetry run python - <<EOF
import os
import torch

print("Torch version :", torch.__version__)
print("Torch CUDA    :", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

try:
    print("Device Count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as e:
    print(e)
EOF

echo "======================================================"

###############################################################################
# Launch
###############################################################################
poetry run accelerate launch \
    --config_file ./phi_agents/rl/conf/accelerate_config.yaml \
    --num_processes=${NUM_GPUS} \
    ./phi_agents/rl/train.py \
    +global@_global_=appworld \
    rl/gpu_allocation=${GPU_ALLOCATION} \
    llm=${MODEL_CONFIG} \
    experiment_name=csf3_appworld_run \
    wandb.enable=${WANDB_ENABLE} \
    wandb.group="32B" \
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
    rl.rollouts_per_scenario_fraction=0.75