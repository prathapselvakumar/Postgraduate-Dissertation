#!/bin/bash
# One-time environment setup script for RunPod.
# This script should be executed from the project root inside the RunPod container.

# Exit on any error
set -e

echo "========================================================"
echo "Starting RunPod setup for ml-loop..."
echo "========================================================"

# Determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# ==============================================================================
# GPU & CUDA PRE-VERIFICATION
# ==============================================================================
echo "Checking CUDA device counts..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed or not in PATH." >&2
    exit 1
fi

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "Detected system GPUs: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 4 ]; then
    echo "WARNING: Expected 4 H200 GPUs for optimal configuration, but found $GPU_COUNT."
fi

# ==============================================================================
# CREDENTIALS SETUP
# ==============================================================================
# First check if they are already in the environment
if [ -z "${HF_TOKEN:-}" ] || [ -z "${WANDB_API_KEY:-}" ]; then
    # Otherwise check if a secrets file exists
    SECRETS_FILE="/workspace/ml-loop.env"
    if [ -f "$SECRETS_FILE" ]; then
        echo "Loading credentials from $SECRETS_FILE..."
        source "$SECRETS_FILE"
    elif [ -f "${PROJECT_ROOT}/ml-loop.env" ]; then
        echo "Loading credentials from ${PROJECT_ROOT}/ml-loop.env..."
        source "${PROJECT_ROOT}/ml-loop.env"
    fi
fi

# Assert credentials are set
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Please export HF_TOKEN or define it in /workspace/ml-loop.env" >&2
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is not set. Please export WANDB_API_KEY or define it in /workspace/ml-loop.env" >&2
    exit 1
fi

# Ensure secrets are saved to a file so the run script can also load them
SECRETS_FILE="/workspace/ml-loop.env"
if [ ! -f "$SECRETS_FILE" ]; then
    echo "Saving credentials to $SECRETS_FILE for runner script access..."
    mkdir -p "$(dirname "$SECRETS_FILE")"
    cat <<EOF > "$SECRETS_FILE"
export HF_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="$WANDB_API_KEY"
EOF
    chmod 600 "$SECRETS_FILE"
fi

# ==============================================================================
# SYSTEM DEPENDENCIES
# ==============================================================================
echo "1. Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    # Check if we run as root or need sudo
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update && apt-get install -y python3-venv python3-pip git tmux
    else
        sudo apt-get update && sudo apt-get install -y python3-venv python3-pip git tmux
    fi
fi

# Ensure local user path is set
export PATH="${HOME}/.local/bin:${PATH}"

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    python3 -m pip install --user poetry --break-system-packages || python3 -m pip install --user poetry
fi

# ==============================================================================
# PYTHON ENVIRONMENT SETUP
# ==============================================================================
echo "2. Setting up Poetry environment inheriting system packages..."
# Create virtualenv that inherits system PyTorch 2.8.0 and CUDA libraries
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# Configure poetry to use the active virtualenv
poetry config virtualenvs.in-project true
poetry install --no-root

# Install compilation dependencies (ninja & flash-attn)
echo "Installing ninja..."
pip install ninja

echo "Installing flash-attn..."
# Build without isolation so it can leverage the pre-installed PyTorch build
pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn compilation failed. Continuing without it..."

# ==============================================================================
# APPWORLD SETUP
# ==============================================================================
echo "3. Creating separate AppWorld virtualenv..."
python3 -m venv appworld-env
appworld-env/bin/pip install click==8.2.1 appworld

echo "4. Installing AppWorld environments..."
appworld-env/bin/appworld install

# Set up and download AppWorld data to persistent volume
echo "5. Downloading AppWorld datasets..."
export APPWORLD_ROOT="/workspace/appworld_data"
mkdir -p "${APPWORLD_ROOT}"
appworld-env/bin/appworld download data --root "${APPWORLD_ROOT}"

# ==============================================================================
# MODEL DOWNLOADS (CACHED CHECK)
# ==============================================================================
# Pre-download HF models to persistent volume
echo "6. Checking/Pre-downloading HuggingFace models..."
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
export HF_HOME="/workspace/.cache/huggingface"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
mkdir -p "${HF_HOME}"

# Function to check if model has been already cached
check_model_cached() {
    local model_name=$1
    # Replacing / with -- to check the folder format in huggingface cache hub
    local folder_name="models--$(echo "$model_name" | tr '/' '--')"
    if [ -d "${HF_HOME}/hub/${folder_name}" ]; then
        return 0
    else
        return 1
    fi
}

# Download Qwen2.5-32B-Instruct if not cached
if check_model_cached "Qwen/Qwen2.5-32B-Instruct"; then
    echo "Model Qwen/Qwen2.5-32B-Instruct is already cached. Skipping download."
else
    echo "Downloading Qwen/Qwen2.5-32B-Instruct..."
    poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-32B-Instruct', token='$HF_TOKEN')"
fi

# Download Qwen2.5-7B-Instruct if not cached
if check_model_cached "Qwen/Qwen2.5-7B-Instruct"; then
    echo "Model Qwen/Qwen2.5-7B-Instruct is already cached. Skipping download."
else
    echo "Downloading Qwen/Qwen2.5-7B-Instruct..."
    poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', token='$HF_TOKEN')"
fi

echo "========================================================"
echo "Setup successfully completed!"
echo "To run the training job, execute:"
echo "  bash scripts/run_loop_runpod.sh"
echo "========================================================"
