#!/bin/bash
# ==============================================================================
# One-time environment setup script for RunPod.
# Run this once after cloning the repository.
# ==============================================================================

set -euo pipefail

echo "========================================================"
echo "Starting RunPod setup for ml-loop..."
echo "========================================================"

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${PROJECT_ROOT}"

# ==============================================================================
# LOAD CREDENTIALS
# ==============================================================================
if [ -f "/workspace/ml-loop.env" ]; then
    echo "Loading credentials from /workspace/ml-loop.env..."
    source /workspace/ml-loop.env
elif [ -f "${PROJECT_ROOT}/ml-loop.env" ]; then
    echo "Loading credentials from ${PROJECT_ROOT}/ml-loop.env..."
    source "${PROJECT_ROOT}/ml-loop.env"
else
    echo "ERROR: ml-loop.env not found."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ] || [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: HF_TOKEN and WANDB_API_KEY must be set."
    exit 1
fi

# ==============================================================================
# INSTALL POETRY
# ==============================================================================
echo
echo "1. Installing Poetry..."

if ! command -v poetry >/dev/null 2>&1; then
    curl -sSL https://install.python-poetry.org | python3 -
fi

export PATH="$HOME/.local/bin:$PATH"

echo "Poetry:"
poetry --version

# ==============================================================================
# CREATE PYTHON ENVIRONMENT
# ==============================================================================
echo
echo "2. Installing Python dependencies..."

poetry config virtualenvs.in-project true

poetry install

# ==============================================================================
# CREATE APPWORLD ENVIRONMENT
# ==============================================================================
echo
echo "3. Creating AppWorld virtual environment..."

rm -rf appworld-env

python3 -m venv appworld-env

appworld-env/bin/pip install --upgrade pip

appworld-env/bin/pip install \
    click==8.2.1 \
    appworld

# ==============================================================================
# INSTALL APPWORLD
# ==============================================================================
echo
echo "4. Installing AppWorld..."

appworld-env/bin/appworld install

# ==============================================================================
# DOWNLOAD DATASET
# ==============================================================================
echo
echo "5. Downloading AppWorld dataset..."

export APPWORLD_ROOT="/workspace/appworld_data"

mkdir -p "${APPWORLD_ROOT}"

appworld-env/bin/appworld download data \
    --root "${APPWORLD_ROOT}"

# ==============================================================================
# HUGGINGFACE CACHE
# ==============================================================================
echo
echo "6. Downloading HuggingFace models..."

export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"

mkdir -p "${HF_HOME}"

poetry run python - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-32B-Instruct",
    token="${HF_TOKEN}"
)

snapshot_download(
    "Qwen/Qwen2.5-7B-Instruct",
    token="${HF_TOKEN}"
)
EOF

# ==============================================================================
# VERIFY INSTALLATION
# ==============================================================================
echo
echo "========================================================"
echo "Verifying installation..."
echo "========================================================"

python3 --version

echo
poetry --version

echo
poetry run python --version

echo
poetry run accelerate env

echo
nvidia-smi

echo
echo "AppWorld version:"
appworld-env/bin/appworld --version || true

echo
echo "========================================================"
echo "RunPod setup completed successfully!"
echo
echo "Start training with:"
echo
echo "bash scripts/run_loop_runpod.sh two_learn_two_infer runpod_32b"
echo
echo "Or for 4-GPU shared mode:"
echo
echo "bash scripts/run_loop_runpod.sh four_learn_four_infer_shared runpod_32b"
echo
echo "Monitor training:"
echo
echo "tail -f /workspace/logs/runpod_32b.log"
echo "========================================================"