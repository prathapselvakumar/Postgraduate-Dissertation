#!/bin/bash
# One-time environment setup script on CSF3.
# This script should be executed on the CSF3 login node.

# Exit on any error
set -e

# Route network traffic through University of Manchester web proxy on the login node
export http_proxy="http://webproxy.its.manchester.ac.uk:3128"
export https_proxy="http://webproxy.its.manchester.ac.uk:3128"

echo "========================================================"
echo "Starting CSF3 setup for ml-loop..."
echo "========================================================"

# Load Python
echo "1. Loading Python module..."
module load python/3.13.1

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
export HF_TOKEN="KGAT_811566e15507f7b62dcf15eccc2e219c"
export HF_HOME="${HOME}/scratch/.cache/huggingface"
poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-32B-Instruct', token='$HF_TOKEN')"
poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', token='$HF_TOKEN')"

echo "========================================================"
echo "Setup successfully completed!"
echo "To run the training job, execute:"
echo "  sbatch scripts/submit_loop_csf3.sbatch"
echo "========================================================"
