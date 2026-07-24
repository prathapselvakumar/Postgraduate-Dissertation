# Watch-Repair: VCBM

VCBM (Visual Causal-chain BookMarking) on a **two-decision** extension of
the classic Watch-Repair credit-assignment task. Unlike a single-decision
environment, where a method hardcoded to credit t=0 can look like it
"solves" credit assignment, this environment has **two** meaningful
decision points and VCBM is not told where they are.

This folder is VCBM-only and fully self-contained: the environment
(`TwoDecisionWatchRepair`) is defined directly inside `watch_repair.py`,
and there is intentionally no RUDDER comparison path or `rudder.py` here.
For a RUDDER comparison on the same task, see `1. RUDDER-Baseline` at the
repo root.

## Why two decisions instead of one

The original Watch-Repair task has exactly one decision (repair/pass at
t=0). This variant adds a second, later decision (express/standard shipping
at t=25, only relevant for repaired watches), so VCBM must **discover**
both decision points via bookmark detection (see `vcbm.py`) before it can
estimate their values -- it is not told in advance that t=0 and t=25
matter.

#### Requirements:
* python3 >= 3.6
* numpy
* argparse
* tqdm
* if plots are desired: matplotlib >= 3.1.0
* if logging to Weights & Biases: wandb

#### Running the demonstration:
```
python3 watch_repair.py --target 0.99 --n_seeds 10
```

Useful arguments:
* `--target` (default 0.99): fraction of good decisions (over the
  `avg_window`-sized moving window) required before training stops.
* `--max_episodes` (default 20000): hard cap per seed (0 for unlimited).
* `--n_seeds` (default 1): number of independent training seeds to run.
* `--use_wandb` / `--no_wandb`: enable/disable Weights & Biases logging (off by default).
* `--wandb_project`, `--wandb_group`, `--wandb_name`: wandb run naming (each seed logs as its own run).
* `--vcbm2_warmup_transitions`, `--vcbm2_classifier_z`, `--vcbm2_forced_min_samples`, `--vcbm2_conf_z`, `--vcbm2_min_samples`, `--vcbm2_eps_start`, `--vcbm2_eps_decay`, `--vcbm2_eps_min`: VCBM's bookmark-classifier and exploration hyperparameters.

## Problem Description

### Environment: two-decision Watch-Repair

You have to repair pocket watches and then sell them, same as the original
task, but now with a **second** decision partway through the episode.

**t=0 -- REPAIR decision.** Action 0 = repair (charged immediately),
action 1 = pass. Optimal (same as the original single-decision task): pass
brands 0 and 3; repair brands 1 and 2.
* repair cost by brand (mean, var): brand0 (1, 2.0), brand1 (4, 2.0), brand2 (5, 2.0), brand3 (24.5, 1.0).
* sales price by brand: brand0 18, brand1 28, brand2 31, brand3 59.

**t=25 -- SHIPPING decision** (only meaningful if the watch was repaired).
Action 0 = express shipping (cost charged immediately), action 1 =
standard. Express reduces the brand-related transport cost paid at
delivery, but only pays off if the reduction exceeds its cost.
* express cost: 2.0.
* express reduction in transport cost, by brand: brand0 0.0, brand1 1.0, brand2 4.0, brand3 0.0.
* so express is a **poor decision** for brand1 (reduction 1.0 < cost 2.0 -> standard is optimal) but a **good decision** for brand2 (reduction 4.0 > cost 2.0 -> express is optimal). Brands 0 and 3 are never repaired, so the shipping decision never arises for them.

All other timesteps: actions have no effect.

Delivery costs (brand-related + general) are unknown and paid only at the
end of the episode, exactly as in the original single-decision task:
* brand-related transport cost (mean, var): brand0 (0.5, 1.5), brand1 (2.5, 1.5), brand2 (4.5, 1.5), brand3 (18, 1.5).
* general delivery events, same for every brand: traffic jams (cost 0.1, probability 0.1 per timestep), flat tires (cost 7.0, probability 0.05 per timestep).

Every episode is 50 time steps long.

### Task

You have to discover, purely from the delayed reward, that there are TWO
decision points (not just t=0), and estimate the value of each action at
each of them:
* the repair decision follows the same brand-dependent logic as the
  original single-decision task;
* the shipping decision is only relevant for repaired watches, and its
  optimal choice depends on whether the express reduction for that brand
  exceeds the express cost.

For example, repairing brand2 pays off, and additionally paying for express
shipping on brand2 pays off further (reduction 4.0 > cost 2.0). But
repairing brand1 pays off while express shipping on brand1 does *not*
(reduction 1.0 < cost 2.0) -- a method that treats "repair implies express
is always worth it" will make the wrong call on brand1.

The state is coded using 6 features:
* Status of watch (repaired or not)
* Status of shipping (express or not)
* Number of traffic jams which increase delivery costs
* Number of flat tires which increase delivery costs
* Brand of watch
* Time
