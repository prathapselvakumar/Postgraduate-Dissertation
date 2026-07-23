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

# Unload conflicting cluster Python modules
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1 || declare -f module >/dev/null; then
    echo "Unloading conflicting cluster python modules..."
    module unload python || true
    module unload anaconda3 || true
fi

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
fiif [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN not set. Defaulting to dummy token for offline use."
    export HF_TOKEN="dummy_token_for_offline_use"
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY not set. Defaulting to dummy key for offline use."
    export WANDB_API_KEY="dummy_key_for_offline_use"
fi

# ==============================================================================
# PROXY CONFIGURATION FOR INTERNET ACCESS ON CSF3
# Checks if a proxy is already defined (e.g. via ml-loop.env or shell environment)
# for SSH reverse tunneling SOCKS proxies. Otherwise, checks for direct internet
# or falls back to the university web proxy.
# Bypassed if HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1 is set.
# ==============================================================================
if [ "${HF_HUB_OFFLINE:-0}" = "1" ] || [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ]; then
    echo "Offline mode enabled via environment variables. Bypassing proxy setup."
    unset http_proxy
    unset https_proxy
else
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
fi

export no_proxy="localhost,127.0.0.1,$(hostname -i 2>/dev/null || echo ''),$(hostname)"
export NO_PROXY="${no_proxy}"

# ==============================================================================
# INSTALL & SETUP LOCAL MINICONDA (FOR PYTHON 3.12 SUPPORT)
# Bypasses cluster-wide python/anaconda version constraints and module load errors.
# ==============================================================================
echo
echo "1. Checking and preparing local Conda environment..."
CONDA_DIR="${SCRATCH_DIR}/miniconda3"
if [ ! -d "${CONDA_DIR}" ]; then
    if [ "${HF_HUB_OFFLINE:-0}" = "1" ] || [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ]; then
        echo "ERROR: Miniconda is not installed at ${CONDA_DIR} and cannot download it in offline mode."
        exit 1
    fi
    echo "Installing Miniconda to scratch space: ${CONDA_DIR}"
    # Download installer
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
    rm -f /tmp/miniconda.sh
fi

# Activate Conda
echo "Activating Miniconda..."
source "${CONDA_DIR}/bin/activate"
eval "$(conda shell.bash hook)"

# Accept Terms of Service if required (Anaconda licensing updates)
echo "Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Create python 3.12 environment if not present
if [ ! -d "${CONDA_DIR}/envs/ml-loop" ]; then
    if [ "${HF_HUB_OFFLINE:-0}" = "1" ] || [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ]; then
        echo "ERROR: Conda environment 'ml-loop' is not created and cannot download packages in offline mode."
        exit 1
    fi
    echo "Creating Python 3.12 conda environment named 'ml-loop'..."
    conda create -y -n ml-loop python=3.12
fi
conda activate ml-loop

# ==============================================================================
# INSTALL POETRY
# ==============================================================================
echo
echo "2. Checking Poetry installation..."

if ! command -v poetry >/dev/null 2>&1; then
    if [ "${HF_HUB_OFFLINE:-0}" = "1" ] || [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ]; then
        echo "WARNING: Poetry not found and offline mode is active. Skipping installation."
    else
        echo "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python -
    fi
fi

export PATH="$HOME/.local/bin:$PATH"

if command -v poetry >/dev/null 2>&1; then
    echo "Poetry Version:"
    poetry --version
fi

# ==============================================================================
# CREATE PYTHON ENVIRONMENT
# ==============================================================================
echo
echo "3. Installing Python dependencies..."

if [ "${HF_HUB_OFFLINE:-0}" = "1" ] || [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ]; then
    echo "Offline mode: skipping poetry install (dependency downloads)."
else
    poetry config virtualenvs.in-project true
    poetry install
fi

# ==============================================================================
# CREATE APPWORLD ENVIRONMENT
# ==============================================================================
echo
echo "4. Creating AppWorld virtual environment..."

if [ -d "appworld-env" ]; then
    echo "AppWorld virtual environment already exists. Skipping creation."
else
    if [ "${HF_HUB_OFFLINE:-0}" = "1" ] || [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ]; then
        echo "Offline mode: appworld-env does not exist and cannot be installed offline. Skipping."
    else
        echo "Creating new appworld-env..."
        python -m venv appworld-env
        appworld-env/bin/pip install --upgrade pip
        appworld-env/bin/pip install \
            click==8.2.1 \
            appworld
    fi
fi

# ==============================================================================
# INSTALL APPWORLD
# ==============================================================================
echo
echo "5. Installing AppWorld..."

if [ -d "appworld-env" ]; then
    if [ "${HF_HUB_OFFLINE:-0}" = "1" ] || [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ]; then
        echo "Offline mode: skipping appworld install step."
    else
        appworld-env/bin/appworld install
    fi
else
    echo "WARNING: appworld-env not available. Skipping AppWorld install."
fi

# ==============================================================================
# DOWNLOAD DATASET (TO SCRATCH)
# ==============================================================================
echo
echo "6. Downloading AppWorld dataset to scratch..."

export APPWORLD_ROOT="${SCRATCH_DIR}/appworld_data"

if [ -d "${APPWORLD_ROOT}" ] && [ "$(ls -A "${APPWORLD_ROOT}" 2>/dev/null)" ]; then
    echo "AppWorld dataset already exists in ${APPWORLD_ROOT}. Skipping download."
else
    if [ "${HF_HUB_OFFLINE:-0}" = "1" ] || [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ]; then
        echo "Offline mode: dataset not found, but cannot download offline. Skipping."
    else
        mkdir -p "${APPWORLD_ROOT}"
        appworld-env/bin/appworld download data \
            --root "${APPWORLD_ROOT}"
    fi
fi

# ==============================================================================
# HUGGINGFACE CACHE (TO SCRATCH)
# ==============================================================================
echo
echo "7. Downloading HuggingFace models to scratch..."

export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"

mkdir -p "${HF_HOME}"

if [ "${HF_HUB_OFFLINE:-0}" = "1" ] || [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ]; then
    echo "Offline mode: skipping HuggingFace model downloads."
else
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
fi

# ==============================================================================
# VERIFY INSTALLATION
# ==============================================================================
echo
echo "========================================================"
echo "Verifying installation..."
echo "========================================================"

python --version

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
echo "Datasets, models, and environments stored in ${SCRATCH_DIR}."
echo "You can now submit training jobs using Slurm."
echo "========================================================"
