#!/bin/bash
# Sync local codebase to RunPod via rsync.
# Run this script from the project root directory.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# RunPod connection variables (based on your exposed TCP SSH info)
POD_IP="213.181.104.34"
POD_PORT="10878"
POD_USER="root"
REMOTE_DIR="/workspace/Postgraduate-Dissertation"

echo "========================================================================="
echo "Syncing local codebase to RunPod..."
echo "Local source:  $PROJECT_ROOT"
echo "Remote target: $POD_USER@$POD_IP:$POD_PORT:$REMOTE_DIR"
echo "========================================================================="

# Ensure the destination folder exists on RunPod
ssh -p "$POD_PORT" "$POD_USER@$POD_IP" "mkdir -p $REMOTE_DIR"

# Run rsync to sync the folder, ignoring git metadata, environments and large caches
rsync -avz -e "ssh -p $POD_PORT" \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='appworld-env/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='logs/' \
    --exclude='wandb/' \
    --exclude='outputs/' \
    --exclude='checkpoints/' \
    "$PROJECT_ROOT/" "$POD_USER@$POD_IP:$REMOTE_DIR/"

echo "========================================================================="
echo "Sync completed successfully!"
echo "========================================================================="
