#!/bin/bash --login
#SBATCH -p gpuA                # Request the gpuA (A100) partition (use gpuL for L40s)
#SBATCH --account=gpu-free     # Account required to run GPU jobs on CSF3
#SBATCH -G 2                   # Number of GPUs (e.g. 1, 2, 4, 8)
#SBATCH -n 4                   # Number of CPU cores
#SBATCH -t 08:00:00            # Time limit (HH:MM:SS)
#SBATCH -J ml-loop-train       # Job name
#SBATCH -o logs/ml_loop_%j.out # Standard output file (create logs dir first)
#SBATCH -e logs/ml_loop_%j.err # Standard error file

# Exit on any error
set -e

# ==============================================================================
# USER CONFIGURATION (Fill these in before running sbatch)
# ==============================================================================
export HF_TOKEN="KGAT_811566e15507f7b62dcf15eccc2e219c"
export WANDB_API_KEY="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"
export WANDB_ENABLE="True" # Set to False to disable wandb tracking
export WANDB_MODE="offline" # WandB must run in offline mode on air-gapped compute nodes

# Enable Hugging Face offline mode to use local pre-downloaded cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# ==============================================================================

# Ensure logs directory exists
mkdir -p logs

# Load CSF3 Modules
echo "Loading environment modules..."
module load python/3.13.1
module load libs/cuda # Use default libs/cuda module version available on CSF3

# Add local poetry bin to PATH
export PATH="${HOME}/.local/bin:${PATH}"

# Set up project variables
PROJECT_ROOT="${HOME}/scratch/ml-loop"
export APPWORLD_ROOT="${HOME}/scratch/appworld_data"
export HF_HOME="${HOME}/scratch/.cache/huggingface"

# Add appworld virtualenv to PATH (critical for AppWorld API sandboxes)
export PATH="${PROJECT_ROOT}/appworld-env/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# PyTorch CUDA configuration
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Move to the project root
cd "${PROJECT_ROOT}"

# Run GPU-specific configurations.
# Below are examples for Single-GPU and Multi-GPU setups.
# Change the parameters based on the number of GPUs requested in '#SBATCH -G' above.

# Example 1: SINGLE GPU SETUP (Requests 1 GPU)
# Make sure you requested '#SBATCH -G 1' above.
# NUM_GPUS=1
# GPU_ALLOCATION="single_gpu"
# MODEL_CONFIG="qwen_2_5_7b_train"

# Example 2: TWO GPU SETUP (1 for training, 1 for inference)
# Uncomment the lines below if you requested '#SBATCH -G 2' above:
NUM_GPUS=2
GPU_ALLOCATION="one_learn_one_infer"
MODEL_CONFIG="qwen_2_5_32b_train"

# Example 3: EIGHT GPU SETUP (4 for training, 4 for inference)
# Uncomment the lines below if you requested '#SBATCH -G 8' above:
# NUM_GPUS=8
# GPU_ALLOCATION="four_learn_four_infer"
# MODEL_CONFIG="qwen_2_5_32b_train"

echo "========================================================================="
echo "Starting LOOP training on CSF3..."
echo "GPUs requested:  $NUM_GPUS"
echo "GPU Allocation:  $GPU_ALLOCATION"
echo "Model config:    $MODEL_CONFIG"
echo "========================================================================="

# Run the training script using accelerate
poetry run accelerate launch \
  --config_file ./phi_agents/rl/conf/accelerate_config.yaml \
  --num_processes=${NUM_GPUS} \
  ./phi_agents/rl/train.py \
  +global@_global_=appworld \
  rl/gpu_allocation=${GPU_ALLOCATION} \
  llm=${MODEL_CONFIG} \
  experiment_name=csf3_appworld_run \
  wandb.enable=${WANDB_ENABLE} \
  wandb.group="32 billion parameter" \
  wandb.project="ml-loop" \
  rl.params.total_iterations=200 \
  rl.max_ckpts=200 \
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
