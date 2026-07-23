# Code

Two research tracks, each comparing a temporal/leave-one-out credit-assignment baseline against
the **Visual Causal Chains (VCC)** variant proposed in this dissertation.

## [Rudder/](Rudder/)

Small-scale proof of concept on a toy delayed-reward task (repairing and selling pocket watches,
from the [RUDDER](https://github.com/ml-jku/rudder) demonstration). Used to validate the causal
credit-assignment mechanism before scaling up to `ML-Loop/`.

| Folder | Description |
|---|---|
| `1. RUDDER-Baseline/` | Reference RUDDER return-decomposition implementation |
| `2. VCC-Experiment/` | VCC variant applied to the same environment |
| `3. Outputs/` | Training logs and result plots for both runs |

Each subfolder is a standalone Python script (`watch_repair.py`), not a package — run directly,
see the local `README.md` in `1. RUDDER-Baseline/` for usage.

## [ML-Loop/](ML-Loop/)

Main-line experiments on the [AppWorld](https://appworld.dev/) benchmark, forked from
[apple/ml-loop](https://github.com/apple/ml-loop) (Chen et al., 2025, *Reinforcement Learning for
Long-Horizon Interactive LLM Agents*). Each subfolder is a **separate git submodule checkout** of
the same fork ([prathapselvakumar/VCBM](https://github.com/prathapselvakumar/VCBM)), kept as
independent working trees so baseline and VCC runs (config, checkpoints, CSF3 job state) don't
collide.

| Folder | Description |
|---|---|
| `1. LOOP-Baseline/`¹ | Unmodified LOOP training/eval pipeline (leave-one-out PPO) |
| `2. VCC-Experiment/` | LOOP pipeline patched with the VCC event detector and memory bank |
| `3. Outputs/` | Weights & Biases run exports (PDF) |

¹ *Pending rename: still on disk as `1. ML-Loop/` — an open editor held a lock on it during the
last reorg pass. Rename manually once no editor/terminal has it open, then update this table.*

See each submodule's own `README.md` for installation, training, and CSF3 cluster usage
(`scripts/README_CSF3.md`).

### Working with the submodules

```bash
git submodule update --init --recursive
```

`1. LOOP-Baseline/` and `2. VCC-Experiment/` are independent checkouts — commits made in one are
not automatically reflected in the other. Sync manually (`git pull`/`git push` within each) when
changes should propagate between them.
