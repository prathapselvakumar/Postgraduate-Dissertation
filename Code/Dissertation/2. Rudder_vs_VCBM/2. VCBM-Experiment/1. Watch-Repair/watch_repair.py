"""
watch_repair.py -- VCBM (learned bookmark discovery) on the WatchRepairEnvironment.

This folder is VCBM-focused and intentionally has no rudder.py (see
`1. RUDDER-Baseline/1. Watch-Repair` for the RUDDER/TD/MC comparison on this
same environment). Only `--policy_learning VCBM` is supported here.

VCBM is not told that the only meaningful decision is the repair/pass choice
at t=0 -- it must discover that via bookmark detection (see vcbm.py) from the
terminal, brand-dependent reward alone.

Usage:
  python watch_repair.py --policy_learning VCBM --n_seeds 20
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from Environment import WatchRepairEnvironment
from vcbm import VCBM

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Code/Rudder_vs_VCBM/2. VCBM-Experiment/1. Watch-Repair/watch_repair.py
#   -> Code/Rudder_vs_VCBM/3. Outputs/Watch-Repair/VCBM/Graphs/...
_OUTPUTS_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "3. Outputs",
    "Watch-Repair", "VCBM", "Graphs"))
LOCAL_GRAPH_DIR = os.path.join(_OUTPUTS_ROOT, "Local-Graph")
WANDB_GRAPH_DIR = os.path.join(_OUTPUTS_ROOT, "Wandb")

parser = argparse.ArgumentParser(description="Watch-Repair -- VCBM")
parser.add_argument("--policy_learning", default="VCBM", type=str, dest="pl",
                    choices=["VCBM"])
parser.add_argument("--max_episodes", default=20000, type=int,
                    help="Hard cap per seed (0 for unlimited).")
parser.add_argument("--target", default=0.95, type=float)
parser.add_argument("--n_seeds", default=1, type=int)
parser.add_argument("--use_wandb", action="store_true", default=True)
parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
parser.add_argument("--wandb_project", default="Dissertation", type=str)
parser.add_argument("--wandb_group", default="Rudder vs Vcbm", type=str)
parser.add_argument("--wandb_entity", default=None, type=str)
parser.add_argument("--wandb_name", default="VCBM - WR", type=str)

# VCBM hyperparameters
parser.add_argument("--vcbm2_warmup_transitions", default=2000, type=int)
parser.add_argument("--vcbm2_classifier_z", default=4.0, type=float)
parser.add_argument("--vcbm2_forced_min_samples", default=60, type=int)
parser.add_argument("--vcbm2_conf_z", default=2.0, type=float)
parser.add_argument("--vcbm2_min_samples", default=8, type=int)
parser.add_argument("--vcbm2_eps_start", default=0.20, type=float)
parser.add_argument("--vcbm2_eps_decay", default=0.995, type=float)
parser.add_argument("--vcbm2_eps_min", default=0.005, type=float)
args = parser.parse_args()

update_rule = args.pl
max_episodes = args.max_episodes if args.max_episodes != 0 else None
target = args.target
n_seeds = max(1, args.n_seeds)

max_time = 50
avg_window = 750


def run_single_seed(rnd_seed, use_wandb, seed_idx=None, total_seeds=1):
    np.random.seed(rnd_seed)

    env = WatchRepairEnvironment(avg_window=avg_window, transport_time=max_time)
    ctrl = VCBM(n_components=env.get_state_shape()[-1],
                 warmup_transitions=args.vcbm2_warmup_transitions,
                 classifier_z=args.vcbm2_classifier_z,
                 forced_min_samples=args.vcbm2_forced_min_samples,
                 conf_z=args.vcbm2_conf_z,
                 min_samples=args.vcbm2_min_samples,
                 eps_start=args.vcbm2_eps_start,
                 eps_decay=args.vcbm2_eps_decay,
                 eps_min=args.vcbm2_eps_min)

    seed_tag = f"[seed {rnd_seed} ({seed_idx}/{total_seeds})] " if total_seeds > 1 else ""
    print(f"\n{seed_tag}Starting training: VCBM on WatchRepairEnvironment")
    print(f"Target: {target*100:.0f}% good decisions over {avg_window}-decision window")
    print("-" * 63)

    episode = 0
    start_time = time.time()
    pbar = tqdm.tqdm(ncols=0)
    poor_history = []
    opt_frac_history = []

    while ((max_episodes is None or episode < max_episodes)
           and (len(env.optimal_actions_list) < avg_window
                or np.mean(env.optimal_actions_list) < target)):
        episode += 1

        state = env.reset()
        done = False
        ep_return = 0.0
        opportunity_records = []
        while not done:
            a, was_opp = ctrl.select_action(state, episode)
            if was_opp:
                opportunity_records.append((state.copy(), a))
            next_state, reward, done = env.step(a)
            ctrl.observe_transition(state, a, next_state)
            ep_return += reward
            state = next_state
        ctrl.update_episode(opportunity_records, ep_return)

        opt_frac = (float(np.mean(env.optimal_actions_list))
                    if env.optimal_actions_list else 0.0)
        poor = episode - env.optimal_choices
        pbar.set_description(f"{seed_tag}{episode:7} | poor={poor:6} | {opt_frac:0.4f}")
        pbar.update(1)
        poor_history.append(poor)
        opt_frac_history.append(opt_frac)

        if use_wandb:
            wandb.log({f"poor_decisions/seed_{rnd_seed}": poor,
                       f"moving_avg_optimal/seed_{rnd_seed}": opt_frac,
                       f"episode/seed_{rnd_seed}": episode})

    pbar.close()
    elapsed = time.time() - start_time
    final_pct = np.mean(env.optimal_actions_list) * 100
    converged = (max_episodes is None) or (episode < max_episodes)

    print(f"\n{seed_tag}Done!  runtime={elapsed:.2f}s  episodes={episode}"
          f"  final_good_pct={final_pct:.2f}%  converged={converged}")
    if not converged:
        print(f"{seed_tag}WARNING: hit --max_episodes cap without reaching target.")

    discovery = ctrl.summary()
    print("\nVCBM discovered component labels:", discovery["component_labels"])
    print("VCBM per-decision-point value estimates:")
    for row in discovery["keys"]:
        print(f"  key={row['key']}: "
              f"a0 mean={row['mean_a0']:+.2f} (n={row['n_a0']})  "
              f"a1 mean={row['mean_a1']:+.2f} (n={row['n_a1']})  "
              f"greedy=a{row['greedy_action']}")

    return dict(seed=rnd_seed, episodes=episode, elapsed=elapsed,
                final_pct=final_pct, converged=converged, discovery=discovery,
                poor_history=poor_history, opt_frac_history=opt_frac_history)


if __name__ == "__main__":
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, group=args.wandb_group,
                   entity=args.wandb_entity, name=args.wandb_name,
                   config=dict(policy_learning=update_rule, target=target,
                               max_episodes=max_episodes, n_seeds=n_seeds,
                               env="WatchRepairEnvironment",
                               vcbm2_warmup_transitions=args.vcbm2_warmup_transitions,
                               vcbm2_classifier_z=args.vcbm2_classifier_z,
                               vcbm2_forced_min_samples=args.vcbm2_forced_min_samples,
                               vcbm2_conf_z=args.vcbm2_conf_z,
                               vcbm2_eps_min=args.vcbm2_eps_min))

    results = []
    for i, seed in enumerate(range(1, n_seeds + 1), start=1):
        results.append(run_single_seed(seed, use_wandb, seed_idx=i, total_seeds=n_seeds))

    if n_seeds > 1:
        n_converged = sum(1 for r in results if r["converged"])
        conv = [r for r in results if r["converged"]]
        episodes_arr = np.array([r["episodes"] for r in conv], dtype=float)
        elapsed_arr = np.array([r["elapsed"] for r in conv], dtype=float)
        final_arr = np.array([r["final_pct"] for r in results], dtype=float)

        print("\n" + "=" * 63)
        print(f"MULTI-SEED SUMMARY  (VCBM on watch-repair env, n_seeds={n_seeds})")
        print("=" * 63)
        print(f"Converged: {n_converged}/{n_seeds} seeds")
        if n_converged:
            print(f"Episodes to converge : {episodes_arr.mean():.1f} +/- {episodes_arr.std():.1f}"
                  f"   (min={episodes_arr.min():.0f}, max={episodes_arr.max():.0f})"
                  f"   [converged seeds only]")
            print(f"Wall-clock time (s)  : {elapsed_arr.mean():.2f} +/- {elapsed_arr.std():.2f}")
        print(f"Final good-decision %: {final_arr.mean():.2f} +/- {final_arr.std():.2f}  [all seeds]")

    run_path = None
    if use_wandb:
        run_path = wandb.run.path
        final_arr_all = np.array([r["final_pct"] for r in results], dtype=float)
        episodes_arr_all = np.array([r["episodes"] for r in results], dtype=float)
        elapsed_arr_all = np.array([r["elapsed"] for r in results], dtype=float)
        wandb.log({"summary/final_pct_mean": final_arr_all.mean(),
                   "summary/final_pct_std": final_arr_all.std(),
                   "summary/episodes_mean": episodes_arr_all.mean(),
                   "summary/episodes_std": episodes_arr_all.std(),
                   "summary/elapsed_mean": elapsed_arr_all.mean(),
                   "summary/elapsed_std": elapsed_arr_all.std()})
        wandb.finish()

    # Local graph: all seeds overlaid on one figure, saved into
    # Code/Rudder_vs_VCBM/3. Outputs/Watch-Repair/VCBM/Graphs/Local-Graph
    os.makedirs(LOCAL_GRAPH_DIR, exist_ok=True)
    fig, ax = plt.subplots()
    for r in results:
        ax.plot(np.arange(len(r["poor_history"])), r["poor_history"],
               label=f"seed {r['seed']}", alpha=0.8)
    ax.set_title(f"Total number of poor decisions (VCBM, {n_seeds} seeds)")
    ax.set_ylabel("#poor decisions")
    ax.set_xlabel("#episodes")
    ax.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    local_path = os.path.join(LOCAL_GRAPH_DIR, "watch_repair_vcbm_all_seeds.png")
    fig.savefig(local_path)
    plt.close(fig)
    print(f"Saved local graph to: {local_path}")

    # Wandb graph: all seeds were logged into the SAME wandb run (per-seed
    # keyed metrics), so pull that one run's history back via the wandb API
    # and re-plot/save into
    # Code/Rudder_vs_VCBM/3. Outputs/Watch-Repair/VCBM/Graphs/Wandb
    if use_wandb and run_path:
        os.makedirs(WANDB_GRAPH_DIR, exist_ok=True)
        api = wandb.Api()
        run = api.run(run_path)
        all_rows = list(run.scan_history())
        for metric in ["moving_avg_optimal", "poor_decisions"]:
            series_by_seed = {r["seed"]: [] for r in results}
            for row in all_rows:
                for seed in series_by_seed:
                    val = row.get(f"{metric}/seed_{seed}")
                    if val is not None:
                        series_by_seed[seed].append(val)
            fig, ax = plt.subplots()
            for seed, values in series_by_seed.items():
                if values:
                    ax.plot(np.arange(len(values)), values,
                           label=f"seed {seed}", alpha=0.8)
            ax.set_title(f"{metric} (VCBM, {n_seeds} seeds) -- from wandb")
            ax.set_ylabel(metric)
            ax.set_xlabel("logged step")
            ax.legend(fontsize="small", ncol=2)
            plt.tight_layout()
            wandb_path = os.path.join(WANDB_GRAPH_DIR, f"{metric}_vcbm.png")
            fig.savefig(wandb_path)
            plt.close(fig)
            print(f"Saved wandb graph to: {wandb_path}")
