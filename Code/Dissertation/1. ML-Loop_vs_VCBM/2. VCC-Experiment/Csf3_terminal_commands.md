# CSF3 Terminal Commands — VCC-Experiment (VCBM)

The paired baseline runs from `../3. ML-LOOP-Upstream` (upstream Apple ml-loop, no VCBM code);
its commands are in `../1. LOOP-Baseline/Csf3_terminal_commands.md`.

## Connect

Off campus, connect to the university VPN first (`vpnconnect.manchester.ac.uk`).

```bash
ssh t83821ps@csf3.itservices.manchester.ac.uk
```

Go to this folder and load the environment:

```bash
cd ~/Postgraduate-Dissertation/"Code/Dissertation/1. ML-Loop_vs_VCBM/2. VCC-Experiment"
source ~/scratch/miniconda3/etc/profile.d/conda.sh && conda activate ml-loop
export PATH="$HOME/.local/bin:$PATH"
```

Get the latest code (after pushing from the laptop):

```bash
cd ~/Postgraduate-Dissertation && git pull origin main && cd - >/dev/null
```

## One-time setup for this folder

```bash
mkdir -p logs
[ -e appworld-env ] || ln -s ~/Postgraduate-Dissertation/"Code/ML-Loop_vs_VCBM/2. VCC-Experiment/appworld-env" appworld-env
poetry run python scripts/patch_appworld_server.py     # must print "Patched:" or "Already patched:"
```

## Submit

| Script | Model / GPUs | Log prefix | Experiment name |
| --- | --- | --- | --- |
| `scripts/submit_7b_2gpu.sbatch` | Qwen2.5-7B, 2× A100 | `vcbm7b_` | `csf3_7b_2gpu_vcbm_s<seed>` |
| `scripts/submit_32b_4gpu.sbatch` | Qwen2.5-32B, 4× A100 | `ml32b_` | `csf3_32b_4gpu_vcbm_s<seed>` |
| `scripts/submit_vcc_csf3.sbatch` | old published 7B run (do not reuse) | `vcbm_` | `csf3_7b_2gpu_90iter_vcbm_r2` |

Smoke test first (2 iterations, no continuation queued):

```bash
ITERS=2 AUTO_RESUBMIT=0 sbatch scripts/submit_7b_2gpu.sbatch vcbm 1
```

Full run, after a trial passes (clear the trial's checkpoints first or it resumes from iteration 2):

```bash
rm -rf ~/scratch/checkpoints/csf3_7b_2gpu_vcbm_s1
sbatch scripts/submit_7b_2gpu.sbatch vcbm 1
```

The full run trains **until convergence**: it stops itself when the 20-iteration moving average
of return has not improved by more than 0.01 for 40 iterations (after at least 100), and it
re-queues itself across the 4-day limit until then. Options (set before `sbatch`):

- `ITERS=600` — hard cap on total iterations (default 600)
- `CONVERGE=false` — run exactly `ITERS` iterations, no early stop
- `AUTO_RESUBMIT=0` — do not queue a continuation

Extra seed (worth more than anything else for the comparison): `sbatch scripts/submit_7b_2gpu.sbatch vcbm 2`,
paired with the ML-LOOP baseline's seed 2.

## Manage jobs

```bash
squeue -u t83821ps                         # R = running, PD = queued; a (Dependency) job is the queued continuation
sacct -u t83821ps -S today --format=JobID,JobName,State,ExitCode,Elapsed
scancel <jobid>                            # one job
scancel -u t83821ps                        # everything, including queued continuations
```

## Logs

Latest VCBM 7B job ID, so the commands below need no editing:

```bash
JOB=$(ls -t logs/vcbm7b_*.out | head -1 | grep -oE '[0-9]+'); echo "$JOB"
```

```bash
tail -f logs/vcbm7b_$JOB.out              # Ctrl+C stops watching, not the job
tail -f logs/vcbm7b_$JOB.err
```

Health checks for a run:

```bash
grep -E "PREFLIGHT (OK|FAILED)" logs/vcbm7b_$JOB.out                                        # want OK
grep "phi_agents imported from" logs/vcbm7b_$JOB.out                                        # must end in 2. VCC-Experiment/phi_agents
grep "execute_with_bookmark" logs/vcbm7b_$JOB.out | grep -oE '" [0-9]{3}' | sort | uniq -c  # want only 200
grep -E "Queued continuation|Previous segment|Cancelled continuation|Converged" logs/vcbm7b_$JOB.out
```

Why a job failed (the first Python traceback is the real error; the launcher summary at the end is not):

```bash
sacct -j $JOB --format=JobID,State,ExitCode,Elapsed
grep -m1 -A40 "Traceback (most recent call last)" logs/vcbm7b_$JOB.err
grep -iE "out of memory|CUDA error|Could not connect|refusing" logs/vcbm7b_$JOB.out logs/vcbm7b_$JOB.err | head
```

## Convergence and checkpoints

```bash
EXP=csf3_7b_2gpu_vcbm_s1
cat ~/scratch/checkpoints/$EXP/CONVERGED 2>/dev/null || echo "not converged yet"
ls -d ~/scratch/checkpoints/$EXP/checkpoint-*/ 2>/dev/null | sort -V | tail -3     # latest checkpoints
```

Convergence state (moving average, best point, iterations since best):

```bash
EXP=csf3_7b_2gpu_vcbm_s1 python3 - <<'EOF'
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

Per-iteration wall-clock time, from checkpoint timestamps:

```bash
EXP=csf3_7b_2gpu_vcbm_s1 python3 - <<'EOF'
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

`avg_return`, `adv_filtered_fraction` and the VCBM bookmark statistics per iteration, read from
the trainer's offline W&B runs (no upload needed). Set `EXP` and `BASE` for the run you want:

```bash
EXP=csf3_7b_2gpu_vcbm_s1 BASE="$PWD/_wandb_logs/wandb" python3 - <<'EOF'
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

print(f"{'iter':>5} {'avg_return':>11} {'adv_filt':>9} {'bm_frac':>8} {'bm/rollout':>10} {'retrieval':>9}")
for r in rows:
    print(f"{r.get('iterations_completed', '?'):>5} {r.get('avg_return', float('nan')):>11.4f} "
          f"{r.get('adv_filtered_fraction', float('nan')):>9.3f} "
          f"{r.get('vcbm/frac_rollouts_with_bookmarks', float('nan')):>8.3f} "
          f"{r.get('vcbm/mean_bookmarks_per_rollout', float('nan')):>10.2f} "
          f"{r.get('vcbm/frac_rollouts_retrieval_path', float('nan')):>9.3f}")
EOF
```

`bm_frac` must stay above 0: a run where it is 0 trained as plain LOOP (that is what happened to the
published runs, whose bookmark requests all returned 404).

For the published 7B run, use `EXP=csf3_7b_2gpu_90iter_vcbm_r2` and
`BASE=~/Postgraduate-Dissertation/"Code/ML-Loop_vs_VCBM/2. VCC-Experiment/_wandb_logs/wandb"`.

## Push W&B logs to the cloud

Compute nodes run with `WANDB_MODE=offline` (no internet), so runs are written under
`_wandb_logs/wandb/offline-run-*` and never uploaded automatically. Sync from the **login node**.

One-time: put a real key in `~/.secrets/ml-loop.env`:

```bash
echo 'WANDB_API_KEY=your_real_key_here' >> ~/.secrets/ml-loop.env
```

```bash
bash scripts/sync_wandb_logs.sh                          # everything in this folder
bash scripts/sync_wandb_logs.sh csf3_7b_2gpu_vcbm_s1      # one experiment (substring match)
```

Safe to re-run at any time, including while a job is still running.

## Interactive sessions

```bash
srun --partition=gpuA --gres=gpu:a100_80g:1 --cpus-per-task=12 --time=1-00:00:00 --pty bash   # 1 GPU, 1 day
srun --partition=gpuA --gres=gpu:a100_80g:2 --cpus-per-task=24 --time=1-00:00:00 --pty bash   # 2 GPU, 1 day
srun --partition=gpuA --gres=gpu:a100_80g:4 --cpus-per-task=48 --time=4-00:00:00 --pty bash   # 4 GPU, 4 days
nvidia-smi
```
