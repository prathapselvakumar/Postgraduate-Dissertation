#!/bin/bash
# Sync local codebase to University of Manchester CSF3 via rsync.
# Run this script from the project root directory (Code/Ml-Loop/ml-loop).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Delegate to Python script to support Windows/HPC environments without local rsync
python "${SCRIPT_DIR}/sync_to_csf3.py" "$@"
