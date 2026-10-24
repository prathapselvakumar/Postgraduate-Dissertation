"""
run_fair_comparison.py -- controlled re-run of the watch-repair comparison.

Addresses three confounds in the original RUDDER-vs-VCBM comparison:

1. Exploration ceiling. The original TabularActor explores at a constant 10%, so its
   expected correct-decision rate can never exceed 0.90 + 0.10 * 0.5 = 95%, which is
   exactly the convergence target. VCBM's exploration decays to 0.5%. `--schedule matched`
   gives the baselines VCBM's schedule (0.20 decaying by 0.995 per episode to 0.005).
2. Missing trajectory-level baseline. Monte Carlo (`MC`) and Q-learning (`TD`) already
   exist in TabularActor with a constant step size of 0.1; this runner reports them alongside
   RUDDER and VCBM, plus `MC_AVG`, Monte Carlo with a 1/n (sample-mean) step size. MC_AVG
   uses the same estimator as VCBM's Welford statistics without any causal discovery, so
   VCBM vs. MC_AVG isolates what the discovery step contributes.
3. Tuning on the evaluation seeds. VCBM's hyperparameters were chosen on seeds 1-20;
   `--seeds 21-40` evaluates on held-out seeds.

Writes one CSV row per (method, seed) so the paper's statistics can be recomputed.

Usage:
  python run_fair_comparison.py --methods VCBM RUDDER MC TD --schedule matched --seeds 21-40
  python run_fair_comparison.py --methods VCBM --forced_min_samples 10 30 60 --seeds 21-40
"""

import argparse
import csv
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUDDER_DIR = os.path.normpath(os.path.join(HERE, "..", "1. RUDDER-Baseline", "1. Watch-Repair"))
VCBM_DIR = os.path.normpath(os.path.join(HERE, "..", "2. VCBM-Experiment", "1. Watch-Repair"))

# rudder.py logs LSTM losses to W&B and nn.py imports matplotlib for plotting helpers; this
# runner uses neither (results go to CSV), so stand in no-op modules when they are missing.
try:
    import wandb  # noqa: F401
except ImportError:
    sys.modules["wandb"] = types.SimpleNamespace(log=lambda *a, **k: None)
try:
    import matplotlib  # noqa: F401
except ImportError:
    _mpl = types.ModuleType("matplotlib")
    _mpl.pyplot = types.ModuleType("matplotlib.pyplot")
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _mpl.pyplot

sys.path.insert(0, RUDDER_DIR)
from Environment import TabularActor, WatchRepairEnvironment  # noqa: E402
from rudder import LessonBuffer  # noqa: E402
from rudder import RRLSTM as LSTM  # noqa: E402

sys.path.insert(0, VCBM_DIR)
from vcbm import VCBM  # noqa: E402

AVG_WINDOW = 750
MAX_TIME = 50
SCHEDULES = {
    # (eps_start, eps_decay, eps_min)
    "original": (0.10, 1.0, 0.10),
    "matched": (0.20, 0.995, 0.005),
}


def not_converged(env, target):
    return len(env.optimal_actions_list) < AVG_WINDOW or np.mean(env.optimal_actions_list) < target


def run_tabular(method, seed, schedule, target, max_episodes):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = WatchRepairEnvironment(avg_window=AVG_WINDOW, transport_time=MAX_TIME)
    env.reset()
    eps_start, eps_decay, eps_min = SCHEDULES[schedule]
    agent = TabularActor(env, lr=0.1, eps_start=eps_start, eps_decay=eps_decay, eps_min=eps_min)
    visit_counts = {}

    def update_monte_carlo_average(actor, states, actions, rewards):
        # Monte Carlo control with a 1/n step size: each Q entry is the sample mean of the
        # returns observed from it, the same estimator VCBM's Welford statistics use, but with
        # no causal discovery and no bookmark selection.
        returns_to_go = rewards[::-1].cumsum()[::-1]
        for i in range(actions.shape[0]):
            idx = tuple([actions[i]] + states[i, :].tolist())
            visit_counts[idx] = visit_counts.get(idx, 0) + 1
            actor.q_table[idx] += (returns_to_go[i] - actor.q_table[idx]) / visit_counts[idx]

    improve = {
        "RUDDER": TabularActor.update_direct_q_estimation,
        "MC": TabularActor.update_monte_carlo,
        "MC_AVG": update_monte_carlo_average,
        "TD": TabularActor.update_q_learning,
    }[method]
    if method == "RUDDER":
        buffer = LessonBuffer(size=2048, max_time=MAX_TIME, n_features=env.get_state_shape()[-1])
        lstm = LSTM(state_input_size=8, n_actions=env.get_n_actions()[-1], buffer=buffer, n_units=16,
                    lstm_lr=1e-2, l2_regularization=1e-6, return_scaling=10,
                    lstm_batch_size=8, continuous_pred_factor=0.5)

    episode = 0
    while not_converged(env, target) and episode < max_episodes:
        episode += 1
        agent.episode = episode
        state = agent.reset()
        states, actions, rewards = [state], [], []
        done = False
        while not done:
            state, a, reward, done = agent.act()
            actions.append(a)
            states.append(state)
            rewards.append(reward)
        agent.reset()
        states, actions, rewards = np.stack(states), np.array(actions), np.array(rewards)
        if method == "RUDDER":
            buffer.add(states=states, actions=actions, rewards=rewards)
            if buffer.different_returns_encountered() and buffer.full_enough():
                if episode % 25 == 0:
                    lstm.train(episode=episode)
                rewards = lstm.redistribute_reward(states=np.expand_dims(states, 0),
                                                   actions=np.expand_dims(actions, 0))[0, :]
        improve(agent, states, actions, rewards)
    return episode, float(np.mean(env.optimal_actions_list)), ""


def run_vcbm(seed, forced_min_samples, target, max_episodes):
    np.random.seed(seed)
    env = WatchRepairEnvironment(avg_window=AVG_WINDOW, transport_time=MAX_TIME)
    ctrl = VCBM(n_components=env.get_state_shape()[-1], forced_min_samples=forced_min_samples)
    episode = 0
    while not_converged(env, target) and episode < max_episodes:
        episode += 1
        state = env.reset()
        done = False
        ep_return = 0.0
        records = []
        while not done:
            a, was_opp = ctrl.select_action(state, episode)
            if was_opp:
                records.append((state.copy(), a))
            next_state, reward, done = env.step(a)
            ctrl.observe_transition(state, a, next_state)
            ep_return += reward
            state = next_state
        ctrl.update_episode(records, ep_return)
    labels = ctrl.classifier.labels() or []
    return episode, float(np.mean(env.optimal_actions_list)), "|".join(labels)


def parse_seeds(spec):
    lo, hi = (int(x) for x in spec.split("-"))
    return list(range(lo, hi + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=["VCBM", "RUDDER", "MC", "MC_AVG", "TD"],
                   choices=["VCBM", "RUDDER", "MC", "MC_AVG", "TD"])
    p.add_argument("--schedule", default="matched", choices=sorted(SCHEDULES),
                   help="Exploration schedule for the TabularActor baselines (VCBM uses its own).")
    p.add_argument("--forced_min_samples", nargs="+", type=int, default=[60],
                   help="VCBM forced-sampling minimum(s) to sweep.")
    p.add_argument("--seeds", default="21-40", help="Inclusive seed range, e.g. 1-20 or 21-40.")
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--max_episodes", type=int, default=20000)
    p.add_argument("--out", default=os.path.join(HERE, "results", "fair_comparison.csv"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["method", "schedule", "forced_min_samples", "seed", "episodes",
                             "converged", "final_window_rate", "vcbm_labels", "runtime_s"])
        for method in args.methods:
            settings = args.forced_min_samples if method == "VCBM" else [""]
            for forced in settings:
                for seed in parse_seeds(args.seeds):
                    t0 = time.time()
                    if method == "VCBM":
                        eps, rate, labels = run_vcbm(seed, forced, args.target, args.max_episodes)
                        schedule = "vcbm"
                    else:
                        eps, rate, labels = run_tabular(method, seed, args.schedule, args.target,
                                                        args.max_episodes)
                        schedule = args.schedule
                    writer.writerow([method, schedule, forced, seed, eps, int(eps < args.max_episodes),
                                     f"{rate:.4f}", labels, f"{time.time() - t0:.1f}"])
                    f.flush()
                    print(f"{method:6} {schedule:8} forced={forced!s:3} seed={seed:3} episodes={eps}")


if __name__ == "__main__":
    main()
