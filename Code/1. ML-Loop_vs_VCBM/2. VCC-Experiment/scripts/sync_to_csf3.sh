#!/bin/bash
# Sync local codebase to University of Manchester CSF3 via rsync.
# Run this script from the project root directory.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Delegate to Python script to support Windows/HPC environments without local rsync
python "${SCRIPT_DIR}/sync_to_csf3.py" "$@"
