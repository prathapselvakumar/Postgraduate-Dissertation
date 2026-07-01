#!/bin/bash
# Sync local ml-loop codebase to CSF3 cluster.
# This script should be run from your local machine.

# Determine the project root directory dynamically
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

REMOTE_USER="t83821ps"
REMOTE_HOST="csf3.itservices.manchester.ac.uk"
REMOTE_DIR="~/scratch/ml-loop"

echo "========================================================================="
echo "Syncing ml-loop codebase to CSF3..."
echo "Local source:  $PROJECT_ROOT"
echo "Remote dest:    $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
echo "========================================================================="
echo "Note: You will be prompted for your CSF3 password and Duo authentication."
echo "========================================================================="

rsync -avz --progress \
  --exclude='.git/' \
  --exclude='.github/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.venv/' \
  --exclude='appworld-env/' \
  --exclude='outputs/' \
  --exclude='experiments/' \
  --exclude='.env' \
  "$PROJECT_ROOT/" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

if [ $? -eq 0 ]; then
  echo "========================================================================="
  echo "Sync complete!"
  echo "You can now log into CSF3 and run the training scripts."
  echo "========================================================================="
else
  echo "========================================================================="
  echo "Sync failed! Please check your credentials and network connection."
  echo "========================================================================="
  exit 1
fi
