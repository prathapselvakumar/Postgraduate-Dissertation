#!/bin/bash
# Sync local ml-loop codebase to CSF3 cluster via Git.
# This script should be run from your local machine.

# Determine the project root directory dynamically
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

REMOTE_USER="t83821ps"
REMOTE_HOST="csf3.itservices.manchester.ac.uk"
REMOTE_DIR="~/scratch/ml-loop"

echo "========================================================================="
echo "Syncing ml-loop codebase via Git..."
echo "Local source:  $PROJECT_ROOT"
echo "Flow:          Local -> Git -> CSF3"
echo "========================================================================="

# Change to the project root directory to ensure git commands run in the repo context
cd "$PROJECT_ROOT"

# Get current branch name
CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
    echo "ERROR: Could not detect current Git branch." >&2
    exit 1
fi
echo "Current branch: $CURRENT_BRANCH"

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git status --porcelain)" ]; then
    echo "Staging and committing local changes..."
    git add -A
    git commit -m "Auto-sync local changes: $(date)"
else
    echo "No local changes to commit."
fi

# Push to remote Git repository
echo "Pushing changes to Git remote..."
git push origin "$CURRENT_BRANCH"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to push local changes to Git." >&2
    exit 1
fi

# Pull on CSF3
echo "========================================================================="
echo "Connecting to CSF3 to pull changes..."
echo "You will be prompted for your CSF3 password and Duo authentication."
echo "========================================================================="

ssh -t "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && git checkout $CURRENT_BRANCH && git pull origin $CURRENT_BRANCH"

if [ $? -eq 0 ]; then
  echo "========================================================================="
  echo "Sync complete!"
  echo "You can now log into CSF3 and run the training scripts."
  echo "========================================================================="
else
  echo "========================================================================="
  echo "Sync failed on CSF3 pull! Please check your credentials and CSF3 repository state."
  echo "========================================================================="
  exit 1
fi