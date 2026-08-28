# University of Manchester CSF3 Setup & Execution Guide

This document describes how to sync, setup, and run the `ml-loop` reinforcement learning pipeline on the **Computational Shared Facility 3 (CSF3)** at the University of Manchester.

---

## 1. Prerequisites & Preparation

### VPN Access
If connecting from off-campus, you **must** connect to the University of Manchester **GlobalProtect VPN** before executing any SSH or sync commands.

### Credentials
Create a file named `ml-loop.env` in your local project root (`ml-loop/ml-loop.env`) with your tokens. This file is excluded from Git to protect your secrets:
```bash
export HF_TOKEN="your_huggingface_write_token"
export WANDB_API_KEY="your_wandb_api_key"
export CSF3_USER="t83821ps"  # (Optional: to prefill username for the sync script)
```

### Option: Route Internet through Local Wi-Fi (SSH SOCKS Tunnel)
If you want the cluster to route all internet traffic (HuggingFace, WandB) through your local machine's Wi-Fi connection instead of the University proxy:
1. **Connect with Tunneling**: When you SSH into CSF3, specify a remote dynamic port forward (`-R`):
   ```bash
   ssh -R 1080 t83821ps@csf3.itservices.manchester.ac.uk
   ```
   *(If you use MobaXterm, set up an SSH tunnel with "Remote port forwarding" mapping remote port `1080` dynamically).*
2. **Add proxy to env**: In your `ml-loop.env`, add the SOCKS5 proxy variables:
   ```bash
   export http_proxy="socks5h://127.0.0.1:1080"
   export https_proxy="socks5h://127.0.0.1:1080"
   ```
The scripts will automatically detect this and route traffic securely through your local Wi-Fi.

---

## 2. Step 1: Sync local code to CSF3

You can run the synchronization script directly from your local PowerShell terminal (from the project root):
```powershell
python scripts/sync_to_csf3.py
```
*(Alternatively, you can run `bash scripts/sync_to_csf3.sh`).*

This Python script does not require `rsync` on Windows. Instead, it:
1. Archives your local codebase into a temporary `.tar.gz` file (excluding large dependencies like `.venv`, `appworld-env`, `logs`, checkpoints, and `.git` metadata).
2. Uploads the archive to CSF3 via standard `scp`.
3. SSHs into CSF3 to extract it in `~/scratch/Postgraduate-Dissertation/Code/ML-LOOP/ml-loop` and performs a clean mirroring by deleting extraneous remote files (while carefully preserving cluster-local directories like `appworld-env`, `.venv`, `checkpoints`, and `logs`).

---

## 3. Step 2: Running the One-Time Setup on CSF3

Log into CSF3:
```bash
ssh t83821ps@csf3.itservices.manchester.ac.uk
```

Navigate to the synced workspace and run the setup script:
```bash
cd ~/scratch/Postgraduate-Dissertation/Code/ML-LOOP/ml-loop
bash scripts/setup_csf3.sh
```

### What `setup_csf3.sh` does:
1. **Local Miniconda Environment**: Downloads and installs Miniconda to `~/scratch/miniconda3` and creates a Python 3.12 conda environment named `ml-loop`. This bypasses module-loading issues and version conflicts with vLLM on Python 3.13.
2. **Quota Protection**: Automatically sets all large caching and download directories (`HF_HOME`, `APPWORLD_ROOT`) under `~/scratch` (e.g. `~/scratch/.cache`, `~/scratch/appworld_data`).
3. **Proxy Configuration**: Automatically configures proxy settings or detects direct connections to ensure internet connectivity on the login node.
4. **Poetry**: Installs Poetry (if not installed) and installs project dependencies into `.venv`.
5. **AppWorld Env**: Creates a separate `appworld-env` virtualenv and downloads the AppWorld dataset.
6. **Model Caching**: Downloads Qwen 2.5 32B & 7B Instruct models directly to scratch space.

---

## 4. Step 3: Submitting Training Jobs

CSF3 uses the Slurm workload manager. Submit a batch training job using:

```bash
sbatch scripts/submit_loop_csf3.sbatch [allocation] [experiment_name]
```

### Examples:
- To run with 2-GPU allocation (default):
  ```bash
  sbatch scripts/submit_loop_csf3.sbatch two_learn_two_infer csf3_32b_run1
  ```
- To run with 4-GPU allocation:
  ```bash
  sbatch scripts/submit_loop_csf3.sbatch four_learn_four_infer csf3_32b_run2
  ```

*Note: Make sure to modify the `#SBATCH --gres=gpu:2` or `#SBATCH --gres=gpu:4` directive in `scripts/submit_loop_csf3.sbatch` to match the number of GPUs requested.*

---

## 5. Monitoring & Maintenance

### Check Job Status
```bash
squeue -u $USER
```

### Cancel a Job
```bash
scancel <job_id>
```

### Reading Logs & Checkpoints
- Run logs are output to: `~/scratch/logs/<experiment_name>.log`
- Live GPU metrics are output to: `~/scratch/logs/<experiment_name>_gpu.csv`
- Model checkpoints are saved to: `~/scratch/checkpoints/<experiment_name>`
- Slurm console stdout/stderr are written to: `logs/ml_loop_<job_id>.out` and `logs/ml_loop_<job_id>.err`

### Important: CSF3 Scratch Cleanup Policy
- **Warning**: Files in `~/scratch` that have not been accessed or updated for **3 months** are subject to automated system deletion.
- **Action**: Once training is complete, download or copy important checkpoints and results to your permanent backed-up space or local machine.
