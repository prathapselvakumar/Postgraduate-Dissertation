#!/usr/bin/env python3
# ==============================================================================
# Python-based Sync Script for University of Manchester CSF3.
# This script archives the codebase, transfers it via SCP, and extracts it via SSH.
# It does not require 'rsync' on the local Windows machine.
# ==============================================================================

import os
import sys
import subprocess
import tarfile
import tempfile
import re

# Directory and file names to exclude from sync
EXCLUDES = {
    '.git', '.venv', 'appworld-env', '__pycache__', 'logs', 'wandb', 'outputs', 'checkpoints', 'ml-loop.env'
}

def load_env(env_path):
    env = {}
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Match export VAR="VAL" or VAR="VAL"
                match = re.match(r'^\s*(?:export\s+)?(\w+)\s*=\s*["\']?(.*?)["\']?\s*$', line)
                if match:
                    env[match.group(1)] = match.group(2)
    return env

def should_exclude(path, base_path):
    rel_path = os.path.relpath(path, base_path)
    parts = rel_path.split(os.sep)
    for part in parts:
        if part in EXCLUDES or part.endswith('.pyc') or part == '.DS_Store':
            return True
    return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Local ml-loop root directory (where scripts/ folder lives)
    ml_loop_dir = os.path.abspath(os.path.join(script_dir, '..'))
    env_path = os.path.join(ml_loop_dir, 'ml-loop.env')
    
    # Load username from ml-loop.env
    env = load_env(env_path)
    csf3_user = env.get('CSF3_USER') or os.environ.get('CSF3_USER')
    
    if not csf3_user:
        try:
            csf3_user = input("Enter your CSF3 username: ").strip()
        except KeyboardInterrupt:
            print("\nSync cancelled.")
            sys.exit(1)
            
    if not csf3_user:
        print("ERROR: CSF3_USER must be provided.")
        sys.exit(1)
        
    csf3_host = "csf3.itservices.manchester.ac.uk"
    # Target directory on CSF3 parent to ml-loop
    remote_dir = "~/scratch/Postgraduate-Dissertation/Code/ML-LOOP"
    
    print("=========================================================================")
    print("Preparing local archive...")
    print(f"Source: {ml_loop_dir}")
    print("=========================================================================")
    
    temp_dir = tempfile.gettempdir()
    tar_filename = 'ml_loop_sync.tar.gz'
    tar_path = os.path.join(temp_dir, tar_filename)
    
    # Create the tar.gz archive
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for root, dirs, files in os.walk(ml_loop_dir):
                # Prune excluded directories in-place to speed up walk
                dirs[:] = [d for d in dirs if d not in EXCLUDES]
                for file in files:
                    full_path = os.path.join(root, file)
                    if not should_exclude(full_path, ml_loop_dir):
                        # Force standard relative path with ml-loop/ prefix
                        rel_path = os.path.relpath(full_path, ml_loop_dir)
                        arcname = os.path.join('ml-loop', rel_path).replace(os.sep, '/')
                        tar.add(full_path, arcname=arcname)
    except Exception as e:
        print(f"ERROR: Failed to create local archive: {e}")
        sys.exit(1)
                    
    print(f"Archive created at {tar_path} ({os.path.getsize(tar_path) / (1024*1024):.2f} MB)")
    print("=========================================================================")
    print(f"Uploading archive to {csf3_user}@{csf3_host}...")
    print("Please enter your password and Duo passcode when prompted.")
    print("=========================================================================")
    
    remote_tar_path = f"~/scratch/{tar_filename}"
    
    # Upload via scp
    scp_cmd = ["scp", tar_path, f"{csf3_user}@{csf3_host}:{remote_tar_path}"]
    try:
        subprocess.run(scp_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: scp failed with exit code {e.returncode}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        sys.exit(1)
        
    print("\n=========================================================================")
    print("Extracting archive on CSF3...")
    print("Please enter your password and Duo passcode again to finalize extraction.")
    print("=========================================================================")
    
    # SSH commands to:
    # 1. Ensure target parent directory exists.
    # 2. Clean remote ml-loop files except for environments, logs, checkpoints, and credentials.
    # 3. Extract tarball.
    # 4. Remove remote archive.
    ssh_commands = [
        f"mkdir -p {remote_dir}",
        # Clean remote ml-loop folder of stale files, protecting python virtualenvs, logs, checkpoints, and credentials env file
        f"if [ -d {remote_dir}/ml-loop ]; then find {remote_dir}/ml-loop -mindepth 1 -maxdepth 1 ! -name '.venv' ! -name 'appworld-env' ! -name 'logs' ! -name 'checkpoints' ! -name 'ml-loop.env' -exec rm -rf {{}} +; fi",
        f"tar -xzf {remote_tar_path} -C {remote_dir}",
        f"rm -f {remote_tar_path}"
    ]
    
    ssh_cmd = ["ssh", f"{csf3_user}@{csf3_host}", "; ".join(ssh_commands)]
    try:
        subprocess.run(ssh_cmd, check=True)
        print("\n=========================================================================")
        print("Sync completed successfully!")
        print("=========================================================================")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ssh commands failed with exit code {e.returncode}")
        sys.exit(1)
    finally:
        if os.path.exists(tar_path):
            os.remove(tar_path)

if __name__ == '__main__':
    main()
