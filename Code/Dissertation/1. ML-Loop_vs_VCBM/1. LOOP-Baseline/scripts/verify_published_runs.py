"""
verify_published_runs.py -- check the two published AppWorld runs from their saved W&B files.

Run on CSF3 (no GPU, no network needed):

    python scripts/verify_published_runs.py ~/scratch

It answers the open questions in the paper's AppWorld section:

1. Did the LOOP baseline actually run with VCBM disabled? Prints every rl.vcc_* value in each
   run's saved config. If the baseline shows vcc_blend_alpha = 1.0, it trained with bookmark
   credit whenever its AppWorld server was patched (checked in step 3).
2. Which temperature produced the logged "temperature=0"? Prints every config key containing
   "temperature" with its full path.
3. Was each experiment's AppWorld server patched with /execute_with_bookmark? Without the
   patch, no turn is ever bookmarked and VCBM is plain LOOP regardless of vcc_blend_alpha.
4. Held-out results. Training ran an evaluation every 5 iterations; prints the final values
   of every summary metric whose name mentions eval or test, so the two runs can be compared
   on held-out scenarios rather than on one training batch.
"""

import glob
import json
import os
import sys

RUNS = ["csf3_7b_2gpu_90iter", "csf3_7b_2gpu_90iter_vcbm_r2"]


def flatten(d, prefix=""):
    """Flatten a W&B config.yaml, unwrapping its {desc, value} wrappers at any depth."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and "value" in v and set(v) <= {"desc", "value"}:
            v = v["value"]
        if isinstance(v, dict):
            yield from flatten(v, key)
        else:
            yield key, v


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        sys.exit("pip install pyyaml (or run inside the ml-loop conda env)")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run_dirs(root, run):
    pattern = os.path.join(root, "**", f"*run-*-{run}")
    exact = [d for d in glob.glob(pattern, recursive=True) if d.rstrip("/").endswith(f"-{run}")]
    return sorted(exact)


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/scratch")
    for run in RUNS:
        print("=" * 78)
        print(run)
        dirs = run_dirs(root, run)
        if not dirs:
            print(f"  no W&B run directory named *-{run} under {root}")
            continue
        for d in dirs:
            print(f"  run dir: {d}")
            cfg_path = os.path.join(d, "files", "config.yaml")
            if os.path.exists(cfg_path):
                flat = dict(flatten(load_yaml(cfg_path)))
                for key in sorted(flat):
                    if "vcc_" in key or "temperature" in key or key.endswith("total_iterations"):
                        print(f"    config {key} = {flat[key]}")
            else:
                print("    (no files/config.yaml)")
            summary_path = os.path.join(d, "files", "wandb-summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
                for key in sorted(summary):
                    if any(s in key.lower() for s in ("eval", "test", "vcbm/")):
                        print(f"    summary {key} = {summary[key]}")
            else:
                print("    (no files/wandb-summary.json; run `wandb sync` first or check the .wandb file)")

    print("=" * 78)
    print("AppWorld server patch status (needs to be True for bookmarks to exist):")
    for env_file in glob.glob(os.path.join(root, "**", "appworld", "serve", "environment.py"),
                              recursive=True):
        with open(env_file) as f:
            patched = "execute_with_bookmark" in f.read()
        print(f"  {patched!s:5}  {env_file}")


if __name__ == "__main__":
    main()
