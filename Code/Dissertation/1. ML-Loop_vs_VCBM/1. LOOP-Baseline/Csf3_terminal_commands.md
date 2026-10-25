# CSF3 Terminal Commands — LOOP Baseline

**New baseline runs come from `../3. ML-LOOP-Upstream`, not this folder.** That folder is Apple's
upstream ml-loop plus CSF3 infrastructure fixes only, with no VCBM code (see its `README_CSF3.md`).
This folder (`1. LOOP-Baseline`) is the fork used for the published 7B baseline run; it contains the
full VCBM code path, and that run was plain LOOP only because its AppWorld server returned 404 on
every bookmark request. Its commands are kept at the end for reading that run's logs.

The paired VCBM runs are in `../2. VCC-Experiment` (commands in its `Csf3_terminal_commands.md`).

## Connect

Off campus, connect to the university VPN first (`vpnconnect.manchester.ac.uk`).

```bash
ssh t83821ps@csf3.itservices.manchester.ac.uk
```

> Use your CSF3 login password; keep credentials out of tracked files (SSH keys are better).

Go to the ML-LOOP baseline folder and load the environment:

```bash
cd ~/Postgraduate-Dissertation/"Code/Dissertation/1. ML-Loop_vs_VCBM/3. ML-LOOP-Upstream"
source ~/scratch/miniconda3/etc/profile.d/conda.sh && conda activate ml-loop
export PATH="$HOME/.local/bin:$PATH"
```

Get the latest code (after pushing from the laptop):

```bash
cd ~/Postgraduate-Dissertation && git pull origin main && cd - >/dev/null
```

## One-time setup for the ML-LOOP folder

```bash
mkdir -p logs
[ -e appworld-env ] || ln -s ~/Postgraduate-Dissertation/"Code/ML-Loop_vs_VCBM/2. VCC-Experiment/appworld-env" appworld-env
ls -d ~/scratch/appworld_data_loop      # the baseline's AppWorld data folder; must exist
```

Do **not** run `poetry install` here: the job puts this folder first on `PYTHONPATH` and refuses to
train if it would import another folder's `phi_agents`.

## Submit

| Script (in `3. ML-LOOP-Upstream`) | Model / GPUs | Log prefix | Experiment name |
| --- | --- | --- | --- |
| `scripts/submit_mlloop_7b_2gpu.sbatch` | Qwen2.5-7B, 2× A100 | `mlloop7b_` | `csf3_7b_2gpu_mlloop_s<seed>` |
| `scripts/submit_mlloop_32b_4gpu.sbatch` | Qwen2.5-32B, 4× A100 | `mlloop32b_` | `csf3_32b_4gpu_mlloop_s<seed>` |

Smoke test first (2 iterations, no continuation queued):

```bash
ITERS=2 AUTO_RESUBMIT=0 sbatch scripts/submit_mlloop_7b_2gpu.sbatch 1
```

Full run, after a trial passes (clear the trial's checkpoints first or it resumes from iteration 2):

```bash
rm -rf ~/scratch/checkpoints/csf3_7b_2gpu_mlloop_s1
sbatch scripts/submit_mlloop_7b_2gpu.sbatch 1
```

The full run trains **until convergence** with the same rule as the VCBM arm (stop when the
20-iteration moving average of return has not improved by more than 0.01 for 40 iterations, after
at least 100) and re-queues itself across the 4-day limit until then. Options (set before `sbatch`):

- `ITERS=600` — hard cap on total iterations (default 600)
- `CONVERGE=false` — run exactly `ITERS` iterations, no early stop
- `AUTO_RESUBMIT=0` — do not queue a continuation

Use the same seed number as the VCBM run it is paired with (`... 1` with `vcbm 1`, and so on).

## Manage jobs

```bash
squeue -u t83821ps                         # R = running, PD = queued; a (Dependency) job is the queued continuation
sacct -u t83821ps -S today --format=JobID,JobName,State,ExitCode,Elapsed
scancel <jobid>                            # one job
scancel -u t83821ps                        # everything, including queued continuations
```

## Logs

Latest ML-LOOP 7B job ID, so the commands below need no editing:

```bash
JOB=$(ls -t logs/mlloop7b_*.out | head -1 | grep -oE '[0-9]+'); echo "$JOB"
```

```bash
tail -f logs/mlloop7b_$JOB.out            # Ctrl+C stops watching, not the job
tail -f logs/mlloop7b_$JOB.err
```

Health checks for a run:

```bash
grep "phi_agents imported from" logs/mlloop7b_$JOB.out      # must end in 3. ML-LOOP-Upstream/phi_agents
grep -c "Could not connect" logs/mlloop7b_$JOB.err          # want 0 (AppWorld servers started)
grep -E "Queued continuation|Previous segment|Cancelled continuation|Converged" logs/mlloop7b_$JOB.out
```

The baseline never calls the bookmark endpoint, so there are no `execute_with_bookmark` lines to check.

Why a job failed (the first Python traceback is the real error; the launcher summary at the end is not):

```bash
sacct -j $JOB --format=JobID,State,ExitCode,Elapsed
grep -m1 -A40 "Traceback (most recent call last)" logs/mlloop7b_$JOB.err
grep -iE "out of memory|CUDA error|Could not connect|refusing" logs/mlloop7b_$JOB.out logs/mlloop7b_$JOB.err | head
```

## Convergence and checkpoints

```bash
EXP=csf3_7b_2gpu_mlloop_s1
cat ~/scratch/checkpoints/$EXP/CONVERGED 2>/dev/null || echo "not converged yet"
ls -d ~/scratch/checkpoints/$EXP/checkpoint-*/ 2>/dev/null | sort -V | tail -3     # latest checkpoints
```

Convergence state (moving average, best point, iterations since best):

```bash
EXP=csf3_7b_2gpu_mlloop_s1 python3 - <<'EOF'
import json, os
d = os.path.expanduser(f"~/scratch/checkpoints/{os.environ['EXP']}")
s = json.load(open(f"{d}/convergence.json"))
r, w = s["returns"], 20
ma = sum(r[-w:]) / w if len(r) >= w else float("nan")
print(f"iterations {len(r)}  latest return {r[-1]:.4f}  moving avg {ma:.4f}")
print(f"best moving avg {s['best_moving_average']:.4f} at iteration {s['best_iteration']}  "
      f"({len(r) - s['best_iteration']} iterations since; stops at 40)")
EOF
```

Compare both arms side by side (run from anywhere):

```bash
python3 - <<'EOF'
import json, os
for exp in ["csf3_7b_2gpu_mlloop_s1", "csf3_7b_2gpu_vcbm_s1"]:
    d = os.path.expanduser(f"~/scratch/checkpoints/{exp}")
    try:
        s = json.load(open(f"{d}/convergence.json"))
    except FileNotFoundError:
        print(f"{exp:28} no convergence history yet"); continue
    done = "CONVERGED" if os.path.exists(f"{d}/CONVERGED") else "running"
    print(f"{exp:28} {len(s['returns']):>4} iters  best MA {s['best_moving_average']:.4f} "
          f"at {s['best_iteration']:>4}  {done}")
EOF
```

Per-iteration wall-clock time, from checkpoint timestamps:

```bash
EXP=csf3_7b_2gpu_mlloop_s1 python3 - <<'EOF'
import datetime, glob, os
d = os.path.expanduser(f"~/scratch/checkpoints/{os.environ['EXP']}")
dirs = sorted(glob.glob(f"{d}/checkpoint-*/"), key=lambda p: int(p.rstrip("/").split("-")[-1]))
prev = None
for p in dirs:
    t = datetime.datetime.fromtimestamp(os.path.getmtime(p))
    print(f"{p.rstrip('/').split('-')[-1]:>5}  {t:%Y-%m-%d %H:%M:%S}  {'' if prev is None else t - prev}")
    prev = t
EOF
```

## Training curve from the offline W&B logs

`avg_return` and `adv_filtered_fraction` per iteration, read from the trainer's offline W&B runs
(no upload needed). ML-LOOP runs write to `3. ML-LOOP-Upstream/_wandb_logs/wandb`:

```bash
EXP=csf3_7b_2gpu_mlloop_s1 BASE="$PWD/_wandb_logs/wandb" python3 - <<'EOF'
import glob, json, os
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

rows = []
for run_dir in sorted(glob.glob(f"{os.environ['BASE']}/offline-run-*-{os.environ['EXP']}")):
    for wandb_file in glob.glob(f"{run_dir}/run-*.wandb")[:1]:
        ds = DataStore(); ds.open_for_scan(wandb_file)
        while (data := ds.scan_data()) is not None:
            rec = pb.Record(); rec.ParseFromString(data)
            if rec.WhichOneof("record_type") != "history":
                continue
            row = {}
            for item in rec.history.item:
                key = ".".join(item.nested_key) if item.nested_key else item.key
                try:
                    row[key] = json.loads(item.value_json)
                except Exception:
                    row[key] = item.value_json
            if "avg_return" in row:
                rows.append(row)

print(f"{'iter':>5} {'avg_return':>11} {'adv_filt':>9} {'out_tokens':>10}")
for r in rows:
    print(f"{r.get('iterations_completed', '?'):>5} {r.get('avg_return', float('nan')):>11.4f} "
          f"{r.get('adv_filtered_fraction', float('nan')):>9.3f} {r.get('avg_output_tokens', float('nan')):>10.1f}")
EOF
```

## Push W&B logs to the cloud

Compute nodes run with `WANDB_MODE=offline` (no internet), so runs are written under
`_wandb_logs/wandb/offline-run-*` and never uploaded automatically. Sync from the **login node**,
inside the folder whose runs you want to upload.

One-time: put a real key in `~/.secrets/ml-loop.env`:

```bash
echo 'WANDB_API_KEY=your_real_key_here' >> ~/.secrets/ml-loop.env
```

```bash
bash scripts/sync_wandb_logs.sh                            # everything in this folder
bash scripts/sync_wandb_logs.sh csf3_7b_2gpu_mlloop_s1      # one experiment (substring match)
```

Safe to re-run at any time, including while a job is still running.

## Published 7B baseline run (read-only reference)

The published run, `csf3_7b_2gpu_90iter` (200 iterations of 12 scenarios × 4 rollouts; the `90iter`
in the name is a leftover), was job `18631089` from this fork. Its files are in the old pre-reset
folder, which git no longer tracks:

```bash
OLD=~/Postgraduate-Dissertation/"Code/ML-Loop_vs_VCBM/1. LOOP-Baseline"
ls "$OLD/logs/" | tail
ls -d --time-style=full-iso -l ~/scratch/checkpoints/csf3_7b_2gpu_90iter/checkpoint-*/ 2>/dev/null | sort -V -k9 | tail -3
```

Confirm it never used bookmarks (every request should be a 404):

```bash
grep -h "execute_with_bookmark" "$OLD"/logs/ml_loop_18631089.out | grep -oE '" [0-9]{3}' | sort | uniq -c
```

Its training curve: run the W&B reader above with
`EXP=csf3_7b_2gpu_90iter BASE="$OLD/_wandb_logs/wandb"`. Training runs log `llm.temperature = 1.0`;
the evaluator's separate runs log `0.0`.

Its logged configuration (shows `rl.vcc_blend_alpha = 1.0` and server patch status):

```bash
python ~/Postgraduate-Dissertation/"Code/Dissertation/1. ML-Loop_vs_VCBM/1. LOOP-Baseline/scripts/verify_published_runs.py" ~/Postgraduate-Dissertation/Code/ML-Loop_vs_VCBM
```

## Interactive sessions

```bash
srun --partition=gpuA --gres=gpu:a100_80g:1 --cpus-per-task=12 --time=1-00:00:00 --pty bash   # 1 GPU, 1 day
srun --partition=gpuA --gres=gpu:a100_80g:2 --cpus-per-task=24 --time=1-00:00:00 --pty bash   # 2 GPU, 1 day
srun --partition=gpuA --gres=gpu:a100_80g:4 --cpus-per-task=48 --time=4-00:00:00 --pty bash   # 4 GPU, 4 days
nvidia-smi
```
