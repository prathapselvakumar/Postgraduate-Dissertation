# Watch-Repair Experiment (VCBM)

VCBM (Visual Causal-chain BookMarking) on the same pocket-watch-repair task used
in `1. RUDDER-Baseline/1. Watch-Repair`: a single meaningful decision (repair or
not, at t=0) followed by 50 sols of pure filler before a brand-dependent,
partially-terminal reward. VCBM is not told in advance that t=0 is the only
decision that matters -- it must discover this via bookmark detection (see
`vcbm.py`).

This folder is VCBM-focused and intentionally has **no `rudder.py`** (and no
`nn.py`, which `rudder.py` depends on) -- for the RUDDER/TD/MC comparison on
this same environment, see `1. RUDDER-Baseline/1. Watch-Repair`.

#### Location:
```
D:\Presentations\Masters Data\Postgraduate-Dissertation\Code\Rudder_vs_VCBM\2. VCBM-Experiment\1. Watch-Repair
```

#### Requirements:
* python3 >= 3.6
* numpy
* argparse
* tqdm
* if plots are desired: matplotlib >= 3.1.0
* if logging to Weights & Biases: wandb

#### Running the demonstration:
```
cd "D:\Presentations\Masters Data\Postgraduate-Dissertation\Code\Rudder_vs_VCBM\2. VCBM-Experiment\1. Watch-Repair"
python3 watch_repair.py --policy_learning VCBM --n_seeds 20
```

Other useful arguments:
* `--target` (default 0.95): fraction of good decisions (over the
  `avg_window`-sized moving window) required before training stops.
* `--max_episodes` (default 20000): hard cap per seed (0 for unlimited).
* `--n_seeds` (default 1): number of independent training seeds to run.
* `--use_wandb` / `--no_wandb`: enable/disable Weights & Biases logging (on by default).
* `--wandb_project`, `--wandb_group`, `--wandb_name`: override the default
  `Dissertation` / `Rudder vs Vcbm` / `VCBM - WR` wandb targets.
* `--vcbm2_warmup_transitions`, `--vcbm2_classifier_z`, `--vcbm2_forced_min_samples`, `--vcbm2_conf_z`, `--vcbm2_min_samples`, `--vcbm2_eps_start`, `--vcbm2_eps_decay`, `--vcbm2_eps_min`: VCBM's bookmark-classifier and exploration hyperparameters.

All seeds in a single invocation are logged to **one** wandb run, with each
seed's curves under its own key (`moving_avg_optimal/seed_<n>`,
`poor_decisions/seed_<n>`), so all seeds appear as separate lines on the
same charts. After training, graphs are also saved locally:
* `Code/Rudder_vs_VCBM/3. Outputs/Watch-Repair/VCBM/Graphs/Local-Graph` -- all seeds overlaid, plotted directly from the run.
* `Code/Rudder_vs_VCBM/3. Outputs/Watch-Repair/VCBM/Graphs/Wandb` -- the same metrics, re-plotted from the logged wandb history via the wandb API.

## How VCBM works here

VCBM does not hardcode the assumption that credit belongs to t=0. Instead
(see `vcbm.py`):
1. A `TransitionClassifier` watches every `(state, action, next_state)`
   transition and classifies each of the 5 state components -- `repaired`,
   the 2 transport-condition counters, `brand`, `time` -- as `clock`
   (`time`, changes every step regardless of action), `static` (never
   changes within an episode), `controlled` (change probability depends
   significantly on the action -- this is how `repaired` is identified),
   or `noise` (changes randomly, independent of action -- the transport
   counters).
2. Whenever a `controlled` component changes, the context it changed
   *from* is stored as a bookmark **opportunity**. This is how t=0 is
   discovered rather than assumed.
3. At each opportunity, VCBM keeps running mean/variance (Welford) of the
   episode return per `(context key, action)`, where the context key
   projects the state onto `static + controlled + clock` components --
   `noise` components (the transport counters) are excluded, which is the
   sample-sharing step a plain tabular Q-table cannot do.
4. Action selection at opportunities uses forced balanced sampling until
   both actions have enough observations, then a two-sample z-test gated
   greedy policy with a permanent exploration floor. At non-opportunity
   timesteps (all 50 filler steps), the action is chosen uniformly at
   random, since VCBM has determined the environment ignores them.

## Problem Description

### Environment

In this task you have to repair pocket watches and then sell them. For a
particular brand of watch you have to decide whether repairing pays off.
There are 4 uniformly randomly chosen initial states which indicate the
brand of watch. The possible actions are repairing (a=0) or not repairing
(a=1). After this first decision (repair or not repair), the next
transitions only depend on the state-transition probabilities.

Repairing a particular brand pays off if the sales price minus the expected
immediate repair costs minus the expected future delivery costs is
positive.

Repairing the watch (a=0) has an immediate negative reward (repair costs).
The immediate repair costs are normal random variables:
* repair cost for brand0: (mean 1, var 2.0)
* repair cost for brand1: (mean 4, var 2.0)
* repair cost for brand2: (mean 5, var 2.0)
* repair cost for brand3: (mean 24.5, var 1.0)

The sales price is known (deterministic value):
* brand0: 18
* brand1: 28
* brand2: 31
* brand3: 59

Delivery costs are unknown.

### Task

VCBM must estimate the expected brand-related delivery costs, which are
e.g. packing costs. These brand-related costs are normal random variables:
* brand0: (mean 0.5, var 1.5)
* brand1: (mean 2.5, var 1.5)
* brand2: (mean 4.5, var 1.5)
* brand3: (mean 18, var 1.5)

These brand-related costs are superimposed by brand-independent general
delivery costs for shipment (e.g. time spent for delivery). General
delivery costs are indicated by patterns in the input, e.g. traffic jams or
flat tires which delay delivery. These events are the same for each brand.
Each event has a cost and a probability to happen at any time step:
* traffic jams: costs 0.1, with probability of 0.1
* flat tires: costs 7.0, with probability of 0.05

Every episode is 50 time steps long. The average general delivery cost is
18. Average general delivery costs: 18.0 = 50 * (0.1 * 0.1 + 7 * 0.05)

For example, repairing brand0 does not pay off, i.e. repairing it is a
**poor decision**: 18 - 1 - 0.5 - 18.0 = -1.5

But, repairing brand1 does pay off: 28 - 5 - 2.5 - 18.0 = 2.5

The state is coded using 5 features:
* Status of watch (repaired or not)
* Number of traffic jams which increase delivery costs
* Number of flat tires which increase delivery costs
* Brand of watch
* Time
