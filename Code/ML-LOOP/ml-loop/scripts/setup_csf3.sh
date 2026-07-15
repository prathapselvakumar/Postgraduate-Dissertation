#!/bin/bash
# One-time environment setup script on CSF3.
# This script should be executed on the CSF3 login node.

# Exit on any error
set -e

# Route network traffic through University of Manchester web proxy on the login node
module load tools/env/proxy2

echo "========================================================"
echo "Starting CSF3 setup for ml-loop..."
echo "========================================================"

# ==============================================================================
# CREDENTIALS
# Tokens are kept out of this file. Copy ml-loop.env.example to
# ~/.secrets/ml-loop.env, fill in real values, then: chmod 600 ~/.secrets/ml-loop.env
# ==============================================================================
if [ -f "${HOME}/.secrets/ml-loop.env" ]; then
    source "${HOME}/.secrets/ml-loop.env"
else
    echo "ERROR: ${HOME}/.secrets/ml-loop.env not found. See ml-loop.env.example." >&2
    exit 1
fi

# Load Python
echo "1. Loading Python module..."
module load python/3.13.1

# Fix for missing shared libraries of python/3.13.1 on CSF3
export LD_LIBRARY_PATH="/opt/apps/pkg/interpreters/python/3.13.1/gcc-14.2.0/lib:${LD_LIBRARY_PATH}"

# Ensure local user path is set
export PATH="${HOME}/.local/bin:${PATH}"

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    python3 -m pip install --user poetry
fi

# Initialize and install dependencies with Poetry
echo "2. Setting up Poetry environment..."
poetry env use python3.13
poetry update

# Setup AppWorld separate environment
echo "3. Creating separate AppWorld virtualenv..."
python3 -m venv appworld-env
appworld-env/bin/pip install click==8.2.1 appworld

echo "4. Installing AppWorld environments..."
appworld-env/bin/appworld install

# Set up and download AppWorld data
echo "5. Downloading AppWorld datasets..."
export APPWORLD_ROOT="${HOME}/scratch/appworld_data"
mkdir -p "${APPWORLD_ROOT}"
appworld-env/bin/appworld download data --root "${APPWORLD_ROOT}"

# Pre-download HF models since compute nodes have no internet access
echo "6. Pre-downloading HuggingFace models (Qwen-2.5-32B and 7B-Instruct)..."
export HF_HOME="${HOME}/scratch/.cache/huggingface"
poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-32B-Instruct', token='$HF_TOKEN')"
poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', token='$HF_TOKEN')"

echo "========================================================"
echo "Setup successfully completed!"
echo "To run the training job, execute:"
echo "  sbatch scripts/submit_loop_csf3.sbatch"
echo "========================================================"