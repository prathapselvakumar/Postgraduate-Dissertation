# CSF3 Terminal Commands — LOOP-Baseline

Placeholders used below:

- `csf3_7b_2gpu_90iter` — checkpoint/wandb run identifier, e.g. `csf3_7b_2gpu` (25-iter) or `csf3_7b_2gpu_90iter` (90-iter)
- `18423252` — SLURM job ID for the current submission

## Connect

```bash
ssh t83821ps@csf3.itservices.manchester.ac.uk
```

> Password: **Prathapsk!080902** — keep credentials out of tracked files.

```bash
cd ~/Postgraduate-Dissertation/Code/ML-Loop_vs_VCBM/1.\ LOOP-Baseline && source ~/scratch/miniconda3/etc/profile.d/conda.sh && conda activate ml-loop
```

## Submit / manage the job

Edit the sbatch file:

```bash
nano scripts/submit_loop_csf3.sbatch
```

Submit a new job:

```bash
sbatch scripts/submit_loop_csf3.sbatch
```

Check job status:

```bash
squeue -j 18423252
```

Cancel job:

```bash
scancel 18423252
```

## Logs

stdout:

```bash
tail -f logs/ml_loop_18423252.out
```

stderr:

```bash
tail -f logs/ml_loop_18423252.err
```

Exit code / elapsed time:

```bash
sacct -j 18423252 --format=JobID,State,ExitCode,Elapsed
```

Root cause of a failure:

```bash
grep -n "Root Cause" logs/ml_loop_18423252.err
```

## Checkpoints

Finished iterations with timestamps:

```bash
ls -d --time-style=full-iso -l ~/scratch/checkpoints/csf3_7b_2gpu_90iter/checkpoint-*/ 2>/dev/null | sort -V -k9
```

Trainer state across all checkpoints:

```bash
for d in ~/scratch/checkpoints/csf3_7b_2gpu_90iter/checkpoint-*/; do echo "=== $d ==="; python -c "
import torch
d = torch.load('$d/trainer_state.pt', map_location='cpu', weights_only=False)
print(d)
"; done
```

Per-iteration wall-clock runtime, from checkpoint mtimes:

```bash
python3 -c "
import glob, os, datetime
dirs = sorted(glob.glob(os.path.expanduser('~/scratch/checkpoints/csf3_7b_2gpu_90iter/checkpoint-*/')), key=lambda p: int(p.rstrip('/').split('-')[-1]))
times = [(int(p.rstrip('/').split('-')[-1]), datetime.datetime.fromtimestamp(os.path.getmtime(p))) for p in dirs]
print(f\"{'iter':>5} {'timestamp':>20} {'duration':>12}\")
prev = None
for i, t in times:
    dur = '' if prev is None else str(t - prev)
    print(f'{i:>5} {t.strftime(\"%Y-%m-%d %H:%M:%S\"):>20} {dur:>12}')
    prev = t
"
```

## Convergence / training curve

Check `wandb-summary.json` for loss/KL/return (if present):

```bash
find ~/scratch -iname "wandb-summary.json" -path "*csf3_7b_2gpu_90iter*" 2>/dev/null -exec sh -c 'echo "=== {} ==="; cat {}; echo' \;
```

Grep raw metrics from the offline `.out` log:

```bash
grep -oE "avg_loss[^,}]*|kl_estimate[^,}]*|grad_norm[^,}]*|mean_return[^,}]*|episode_return[^,}]*" logs/ml_loop_18423252.out | tail -200
```

Sync an offline wandb run to the cloud:

```bash
find ~/scratch -maxdepth 6 -type d -iname "offline-run-*" 2>/dev/null
# then: wandb sync <path-to-run-dir-from-above>
```

`avg_return` / `adv_filtered_fraction` trend across all trainer wandb runs:

```bash
python3 -c "
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
import glob, json

base = '/mnt/iusers01/fse-ugpgt01/mace01/t83821ps/Postgraduate-Dissertation/Code/ML-Loop_vs_VCBM/1. LOOP-Baseline/_wandb_logs/wandb'
run_dirs = sorted(glob.glob(f'{base}/offline-run-*-csf3_7b_2gpu_90iter'))

def is_trainer(run_dir):
    dbg = f'{run_dir}/logs/debug.log'
    try:
        with open(dbg) as f:
            content = f.read()
        return 'MainThread' in content and 'run started' in content
    except FileNotFoundError:
        return False

trainer_dirs = [d for d in run_dirs if is_trainer(d)]
print('trainer run dirs (chronological):')
for d in trainer_dirs:
    print(' ', d)
print()

all_iter_rows = []
for run_dir in trainer_dirs:
    wandb_files = glob.glob(f'{run_dir}/run-*.wandb')
    if not wandb_files:
        continue
    ds = DataStore()
    ds.open_for_scan(wandb_files[0])
    while True:
        data = ds.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        if record.WhichOneof('record_type') == 'history':
            row = {}
            for item in record.history.item:
                key = '.'.join(item.nested_key) if item.nested_key else item.key
                try:
                    row[key] = json.loads(item.value_json)
                except Exception:
                    row[key] = item.value_json
            if 'avg_return' in row:
                row['_run_dir'] = run_dir.split('/')[-1]
                all_iter_rows.append(row)

print(f'{len(all_iter_rows)} iteration-summary rows with avg_return found')
print()
print(f\"{'run':45} {'iter':>5} {'avg_return':>11} {'adv_filt%':>10}\")
for r in all_iter_rows:
    print(f\"{r.get('_run_dir',''):45} {r.get('iterations_completed','?'):>5} {r.get('avg_return',float('nan')):>11.4f} {r.get('adv_filtered_fraction',float('nan')):>10.3f}\")
"
```

## Active runs

| Run | `csf3_7b_2gpu_90iter` | `18423252` | Notes |
| --- | --- | --- | --- |
| LOOP baseline, 25-iter | `csf3_7b_2gpu` | `18423252` | original run |
| VCBM | `csf3_7b_2gpu_vcbm` | — | |
| LOOP baseline, 90-iter / half-rollout | `csf3_7b_2gpu_90iter` | — | `scenarios_per_iteration` halved 24→12 (48 rollouts/iteration instead of 96), `total_iterations=90`, `max_ckpts=90` |

## Interactive sessions

1 GPU, 1 day:

```bash
srun --partition=gpuA --gres=gpu:a100_80g:1 --cpus-per-task=12 --time=1-00:00:00 --pty bash
```

2 GPU, 1 day:

```bash
srun --partition=gpuA --gres=gpu:a100_80g:2 --cpus-per-task=24 --time=1-00:00:00 --pty bash
```

4 GPU, 4 days:

```bash
srun --partition=gpuA --gres=gpu:a100_80g:4 --cpus-per-task=48 --time=4-00:00:00 --pty bash
```

Check GPU status:

```bash
nvidia-smi
```
