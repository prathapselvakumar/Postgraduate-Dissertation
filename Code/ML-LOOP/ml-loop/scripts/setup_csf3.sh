#!/bin/bash
# ==============================================================================
# One-time environment setup script for University of Manchester CSF3.
# Run this once after cloning/syncing the repository to CSF3.
# ==============================================================================

set -euo pipefail

echo "========================================================"
echo "Starting CSF3 setup for ml-loop..."
echo "========================================================"

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${PROJECT_ROOT}"

# CSF3 Scratch space setup to bypass home directory quota limits
SCRATCH_DIR="${HOME}/scratch"
mkdir -p "${SCRATCH_DIR}"

# ==============================================================================
# LOAD CREDENTIALS
# ==============================================================================
if [ -f "${PROJECT_ROOT}/ml-loop.env" ]; then
    echo "Loading credentials from ${PROJECT_ROOT}/ml-loop.env..."
    source "${PROJECT_ROOT}/ml-loop.env"
else
    echo "ERROR: ml-loop.env not found in the project root."
    echo "Please create an ml-loop.env file with your HF_TOKEN and WANDB_API_KEY."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ] || [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: HF_TOKEN and WANDB_API_KEY must be set in ml-loop.env."
    exit 1
fi

# ==============================================================================
# PROXY CONFIGURATION FOR INTERNET ACCESS ON CSF3
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

export no_proxy="localhost,127.0.0.1,$(hostname -i 2>/dev/null || echo ''),$(hostname)"
export NO_PROXY="${no_proxy}"

# ==============================================================================
# LOAD MODULES
# ==============================================================================
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1 || declare -f module >/dev/null; then
    echo "Loading python module..."
    module load python/3.13.1 || echo "WARNING: Could not load python module. Proceeding with default environment."
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
# DOWNLOAD DATASET (TO SCRATCH)
# ==============================================================================
echo
echo "5. Downloading AppWorld dataset to scratch..."

export APPWORLD_ROOT="${SCRATCH_DIR}/appworld_data"

mkdir -p "${APPWORLD_ROOT}"

appworld-env/bin/appworld download data \
    --root "${APPWORLD_ROOT}"

# ==============================================================================
# HUGGINGFACE CACHE (TO SCRATCH)
# ==============================================================================
echo
echo "6. Downloading HuggingFace models to scratch..."

export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"

mkdir -p "${HF_HOME}"

poetry run python - <<EOF
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-32B-Instruct",
    token=os.environ.get("HF_TOKEN")
)

snapshot_download(
    "Qwen/Qwen2.5-7B-Instruct",
    token=os.environ.get("HF_TOKEN")
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
poetry run accelerate env || echo "Warning: Accelerate config not found or initialized."

echo
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not available (non-GPU node)"
fi

echo
echo "AppWorld version:"
appworld-env/bin/appworld --version || true

echo
echo "========================================================"
echo "CSF3 setup completed successfully!"
echo "Datasets and models stored in ${SCRATCH_DIR}."
echo "You can now submit training jobs using Slurm."
echo "========================================================"
