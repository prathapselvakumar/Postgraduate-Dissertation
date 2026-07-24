# Mars Rover Experiment (VCBM)

VCBM (Visual Causal-chain BookMarking) on the same tough, long-horizon
credit-assignment benchmark used in `1. RUDDER-Baseline/2. Mars-Rover-Experiment`:
a rover mission with **three** sequential decisions spread across a long
episode, with a **fully sparse, terminal-only** reward. Where the classic
Watch-Repair task has exactly one meaningful decision (repair/pass at t=0),
this mission has three, and VCBM is not told where any of them are.

This folder is VCBM-focused and intentionally has **no `rudder.py`** --
`mars_rover.py` still accepts `--policy_learning RUDDER` as a CLI choice,
but that path imports `rudder.py` lazily and will fail here since the file
is not bundled in this folder. For a working RUDDER comparison on this
environment, see `1. RUDDER-Baseline/2. Mars-Rover-Experiment`.

## Why this is harder than Watch-Repair

* **Three decision points instead of one**, spread over a 150-sol mission
  (vs. 50 sols for Watch-Repair), so the credit-assignment gap between a
  decision and the terminal reward is much larger.
* **The reward is 100% terminal** -- no partial costs are paid at the
  decision points themselves (unlike Watch-Repair, where the repair cost is
  charged immediately at t=0). Every consequence is deferred to mission end.
* **The decisions are conditionally entangled three ways**: the value of the
  route choice (t=40) depends on the power mode chosen at t=0, and the value
  of the drill choice (t=90) depends on BOTH earlier decisions (via remaining
  battery and whether the rover was damaged). VCBM must discover all three
  decision points *and* their interaction, not just treat them as
  independent single-step credit-assignment problems.

#### Requirements:
* python3 >= 3.6
* numpy
* argparse
* tqdm
* if plots are desired: matplotlib >= 3.1.0
* if logging to Weights & Biases: wandb

#### Running the demonstration:
```
python3 mars_rover.py --policy_learning VCBM --target 0.9907 --n_seeds 10
```

`--policy_learning` accepts `VCBM` or `RUDDER`, but only `VCBM` will
actually run in this folder (see note above).

Other useful arguments:
* `--target` (default 0.9907): fraction of good decisions (over the
  `avg_window`-sized moving window) required before training stops.
* `--max_episodes` (default 40000): hard cap per seed (0 for unlimited).
* `--n_seeds` (default 10): number of independent training seeds to run.
* `--use_wandb` / `--no_wandb`: enable/disable Weights & Biases logging (on by default).
* `--wandb_project`, `--wandb_group`, `--wandb_name`: override the default
  `Dissertation` / `VCBM: Rudder vs Vcbm` / `VCBM- ME` wandb targets.
* `--vcbm2_warmup_transitions`, `--vcbm2_classifier_z`, `--vcbm2_forced_min_samples`, `--vcbm2_conf_z`, `--vcbm2_min_samples`, `--vcbm2_eps_start`, `--vcbm2_eps_decay`, `--vcbm2_eps_min`: VCBM's bookmark-classifier and exploration hyperparameters.

All seeds in a single invocation are logged to **one** wandb run, with each
seed's curves under its own key (`moving_avg_optimal/seed_<n>`,
`poor_decisions/seed_<n>`), so all seeds appear as separate lines on the
same charts. After training, graphs are also saved locally:
* `Code/Rudder_vs_VCBM/3. Outputs/Mars-Rover-Experiment/VCBM/Graphs/Local-Graph` -- all seeds overlaid, plotted directly from the run.
* `Code/Rudder_vs_VCBM/3. Outputs/Mars-Rover-Experiment/VCBM/Graphs/Wandb` -- the same metrics, re-plotted from the logged wandb history via the wandb API.

## Problem Description

### Environment: a rover mission

A rover lands on one of 4 terrain types (a static, decision-relevant context
revealed at reset, analogous to "brand" in Watch-Repair) and must survive a
150-sol mission while making three consequential decisions:

**t=0 -- POWER decision.** Action 0 = high-power mode (drains the battery
faster, but is required to afford a deep drill later); action 1 =
low-power/conservative mode (slow drain, safer, but leaves insufficient
reserves for deep drilling regardless of what is chosen at t=90).
* battery drain per sol: high-power 0.55, low-power 0.30 (battery capacity 100).

**t=40 -- ROUTE decision** (only meaningful if the rover is still active).
Action 0 = risky shortcut across the crater rim (cheap, but risks dust-storm
damage); action 1 = safe detour (costs more battery, never causes damage).
* extra battery cost: risky 2.0, safe 9.0.
* dust-storm damage probability: 0.35 if high-power mode was chosen (degraded
  shielding), 0.12 if low-power.
* damage is permanent for the rest of the mission and rules out deep
  drilling entirely.

**t=90 -- DRILL decision** (only meaningful if the rover has battery left and
reached this point undamaged). Action 0 = deep drill (large battery cost,
high science value, but only pays off if undamaged and battery allows);
action 1 = shallow sample (small battery cost, modest guaranteed value).
* battery cost: deep drill 28.0, shallow sample 6.0.
* deep drill science value by terrain (mean, var): terrain0 (40, 4), terrain1
  (65, 5), terrain2 (90, 6), terrain3 (55, 5).
* shallow sample science value by terrain (mean, var): terrain0 (12, 1.5),
  terrain1 (14, 1.5), terrain2 (13, 1.5), terrain3 (15, 1.5).

All other timesteps: actions have no effect (pure filler, exactly like the
"all other timesteps" in Watch-Repair's two-decision variant).

### Reward: entirely terminal, paid once at t=150

```
final_reward = science_value - damage_penalty - battery_penalty
```
* `science_value`: from the drill decision (see above), zero if the mission
  never reached t=90.
* `damage_penalty`: 30.0 if the risky route was taken and a dust-storm hit.
* `battery_penalty`: 50.0 if the battery ever went negative (mode + route +
  drill choices were too aggressive for the available power budget).

### Task

VCBM must DISCOVER, purely from the terminal reward, all three decision
points, when each occurs, and how they interact -- via bookmark detection
(see `vcbm.py`), not by being told in advance where t=0, t=40, and t=90 are:
* the power mode determines whether a deep drill is even affordable later;
* the route choice trades battery for damage risk, and that risk depends on
  the power mode already chosen;
* the drill choice only pays off if the rover banked enough battery and
  avoided damage -- both consequences of the two earlier decisions.

For example, on terrain2 (highest deep-drill value, 90): choosing high-power
at t=0, the safe route at t=40 (to protect the battery for drilling), and
the deep drill at t=90 is optimal. On terrain0 (lowest deep-drill value, 40),
the shallow-sample plan never justifies banking battery, so low-power and
the risky shortcut are optimal since there is no downstream battery to
protect.

The state is coded using 5 features:
* Power mode (chosen at t=0)
* Damaged (whether a dust-storm hit on the risky route)
* Battery-low flag (coarse discretization of remaining battery, so the state
  stays small enough for VCBM's tabular value estimates)
* Terrain
* Time
