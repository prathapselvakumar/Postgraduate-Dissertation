#!/bin/bash --login
#
#SBATCH --job-name=vcbm7b
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/vcbm7b_%j.out
#SBATCH --error=logs/vcbm7b_%j.err
#
# Qwen2.5-7B-Instruct, LoRA, FSDP2 + vLLM sharing 2x A100 80GB (the published 7B settings).
# Submit FROM THE 2. VCC-Experiment FOLDER for both arms, so they share identical code and
# the same patched AppWorld install:
#
#   sbatch scripts/submit_7b_2gpu.sbatch vcbm 1     # VCBM-blended LOOP, seed 1
#   sbatch scripts/submit_7b_2gpu.sbatch loop 1     # plain LOOP (beta = 0), seed 1
#   ITERS=2 sbatch scripts/submit_7b_2gpu.sbatch vcbm 1   # smoke test first
#
# Runs ITERS iterations (default 400), re-queueing itself across the 4-day limit and
# resuming from the last checkpoint until done. CONVERGE=true also stops early once the
# training return plateaus (phi_agents/rl/convergence.py);
# AUTO_RESUBMIT=0 disables re-queueing. Stop a chain with: scancel -u $USER

set -euo pipefail

ARM="${1:-vcbm}"
SEED="${2:-1}"
ITERS="${ITERS:-400}"          # total iterations (a cap when CONVERGE=true)
CONVERGE="${CONVERGE:-false}"  # CONVERGE=true also stops early once the return plateaus
case "$ARM" in
    vcbm) BETA=1.0 ;;
    loop) BETA=0.0 ;;
    *) echo "First argument must be 'vcbm' or 'loop', got '$ARM'"; exit 1 ;;
esac
EXPERIMENT="csf3_7b_2gpu_${ARM}_s${SEED}"
echo "Arm         : $ARM (vcc_blend_alpha=$BETA)"
echo "Seed        : $SEED"
echo "Iterations  : $ITERS (early stop on convergence: $CONVERGE)"
echo "Experiment  : $EXPERIMENT"

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

# 2. VCC-Experiment and 3. ML-LOOP-Upstream both install a package named phi-agents into the
# same conda env, and whichever was installed last wins. Put this folder first on the import
# path so the job always runs its own code (checked after the environment is set up below).
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

########################################################################
# Run until convergence
########################################################################
# Training stops itself when the return plateaus (rl.convergence, see
# phi_agents/rl/convergence.py) and writes $CKPT_DIR/CONVERGED. Each job queues its own
# continuation, which only runs if this job is cut off by the time limit; a converged,
# finished, failed or cancelled job ends the chain.

CKPT_DIR="$HOME/scratch/checkpoints/$EXPERIMENT"
CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
MAX_CHAIN="${MAX_CHAIN:-10}"

if [ -f "$CKPT_DIR/CONVERGED" ]; then
    echo "Already converged, nothing to do: $(cat "$CKPT_DIR/CONVERGED")"
    exit 0
fi

if [ -n "${PREV_JOB:-}" ]; then
    PREV_STATE="$(sacct -j "$PREV_JOB" -X -n -o State | head -1 | awk '{print $1}')"
    echo "Previous segment $PREV_JOB ended with state: $PREV_STATE"
    case "$PREV_STATE" in
        TIMEOUT|PREEMPTED|NODE_FAIL) ;;
        *)
            echo "Previous segment was not cut off by the cluster; not continuing."
            exit 0
            ;;
    esac
fi

NEXT_JOB=""
if [ "${AUTO_RESUBMIT:-1}" = "1" ] && [ "$CHAIN_DEPTH" -lt "$MAX_CHAIN" ]; then
    NEXT_JOB="$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,PREV_JOB="$SLURM_JOB_ID",CHAIN_DEPTH="$((CHAIN_DEPTH + 1))" \
        "$PROJECT_ROOT/scripts/submit_7b_2gpu.sbatch" "$@")"
    echo "Queued continuation $NEXT_JOB (segment $((CHAIN_DEPTH + 1)) of at most $MAX_CHAIN);"
    echo "it only runs if this job hits the time limit."
fi

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

export APPWORLD_ROOT="$SCRATCH_DIR/appworld_data_vcc"

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

echo
PHI_AGENTS_DIR="$(poetry run python -c 'import phi_agents, os; print(os.path.dirname(os.path.abspath(phi_agents.__file__)))')"
echo "phi_agents imported from: $PHI_AGENTS_DIR"
if [ "$PHI_AGENTS_DIR" != "$(pwd -P)/phi_agents" ] && [ "$PHI_AGENTS_DIR" != "$PROJECT_ROOT/phi_agents" ]; then
    echo "phi_agents resolves outside $PROJECT_ROOT; refusing to train with another folder's code."
    exit 1
fi

########################################################################
# Training
########################################################################
echo "Checking AppWorld..."
ls -lah "$APPWORLD_ROOT"

echo
echo "Checking the 7B model is in the offline HF cache..."
if ! ls -d "$HF_HOME"/hub/models--Qwen--Qwen2.5-7B-Instruct >/dev/null 2>&1; then
    echo "Qwen/Qwen2.5-7B-Instruct not found under $HF_HOME/hub (jobs run offline)."
    echo "Download it on a login node first: huggingface-cli download Qwen/Qwen2.5-7B-Instruct"
    exit 1
fi

echo
echo "Patching and verifying the AppWorld bookmark endpoint..."
# Both arms need a working endpoint: VCBM to get credit, LOOP so its vcbm/* bookmark
# statistics are comparable. A 404 here would silently turn VCBM into plain LOOP.
poetry run python scripts/patch_appworld_server.py
poetry run python scripts/preflight_bookmarks.py
echo
echo "Launching Training..."
echo

# Two 2-GPU jobs can share one 4-GPU node, so accelerate's default rendezvous port
# (29500) may already be taken by the other job (EADDRINUSE). Use a free port instead.
MAIN_PORT="$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
echo "Distributed rendezvous port: $MAIN_PORT"

# Ray joins the newest cluster under its temp root when ray.init() has no address
# (phi_agents/rl/utils/ray_utils.py). On a shared node that could be the other job's
# cluster, so give this job its own temp root and never inherit a RAY_ADDRESS.
unset RAY_ADDRESS
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"
echo "Ray temp dir: $RAY_TMPDIR"

poetry run accelerate launch \
    --config_file phi_agents/rl/conf/accelerate_config.yaml \
    --main_process_port "$MAIN_PORT" \
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
    rl.seed=$SEED \
    experiment_name=$EXPERIMENT \
    rl.cloud_path=$HOME/scratch/checkpoints/$EXPERIMENT \
    wandb.project=ml-loop \
    wandb.group=CSF3-7B \
    wandb.run=$EXPERIMENT \
    rl.params.total_iterations=$ITERS \
    rl.convergence.enable=$CONVERGE \
    rl.max_ckpts=20 \
    rl.params.scenarios_per_iteration=12 \
    rl.params.rollouts_per_scenario=4 \
    rl.params.minibatch_size=32 \
    rl.eval.eval_every_n_iterations=5 \
    rl.eval.overrides.rl.num_scenario_runners=12 \
    rl.eval.overrides.rl.params.scenarios_per_iteration=10 \
    rl.vcc_weighting_mode=unified \
    rl.vcc_temperature=0.5 \
    rl.vcc_top_k=6 \
    rl.vcc_alpha=0.05 \
    rl.vcc_blend_alpha=$BETA

# Training exited cleanly (converged or reached the cap): the queued continuation is not needed.
if [ -n "$NEXT_JOB" ]; then
    scancel "$NEXT_JOB" && echo "Cancelled continuation $NEXT_JOB."
fi
if [ -f "$CKPT_DIR/CONVERGED" ]; then
    echo "Converged: $(cat "$CKPT_DIR/CONVERGED")"
fi
