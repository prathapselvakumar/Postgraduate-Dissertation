"""
watch_pair2.py -- VCBM2 (learned bookmark discovery) vs RUDDER on the
TwoDecisionWatchRepair environment.

Unlike watch_pair.py, this comparison is an honest test of long-horizon
credit assignment: there are TWO meaningful decision points (t=0
repair/pass, t=25 express/standard for repaired items), and neither
method is told where they are.

- VCBM2 must DISCOVER both decision points via bookmark detection
  (see vcbm2.py) before it can estimate their values.
- RUDDER must learn to redistribute the delayed reward onto both
  decisions via its LSTM (rudder.py, unmodified -- input preparation for
  the extra state component is handled by a subclass here).

Usage:
  python watch_pair2.py --policy_learning VCBM2  --target 0.99 --n_seeds 10
  python watch_pair2.py --policy_learning RUDDER --target 0.99 --n_seeds 10
"""

import argparse
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from two_decision_env import TwoDecisionWatchRepair
from vcbm2 import VCBM2

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

parser = argparse.ArgumentParser(description="Two-decision Watch Repair — RUDDER vs VCBM2")
parser.add_argument("--policy_learning", default="VCBM2", type=str, dest="pl",
                    required=True, choices=["RUDDER", "VCBM2"])
parser.add_argument("--max_episodes", default=20000, type=int,
                    help="Hard cap per seed (0 for unlimited).")
parser.add_argument("--target", default=0.99, type=float)
parser.add_argument("--n_seeds", default=1, type=int)
parser.add_argument("--use_wandb", action="store_true", default=False)
parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
parser.add_argument("--wandb_project", default="dissertation", type=str)
parser.add_argument("--wandb_group", default="Rudder vs VCBM (two-decision)", type=str)
parser.add_argument("--wandb_entity", default=None, type=str)
parser.add_argument("--wandb_name", default=None, type=str)

# Baseline (RUDDER) exploration
parser.add_argument("--baseline_eps_start", default=0.10, type=float)
parser.add_argument("--baseline_eps_decay", default=0.9995, type=float)
parser.add_argument("--baseline_eps_min", default=0.005, type=float,
                    help="Exploration floor for the RUDDER actor too, so both "
                         "methods share the same lifelong-exploration guarantee.")

# VCBM2 hyperparameters
parser.add_argument("--vcbm2_warmup_transitions", default=2000, type=int)
parser.add_argument("--vcbm2_classifier_z", default=4.0, type=float)
parser.add_argument("--vcbm2_forced_min_samples", default=300, type=int)
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

# Original repo hyperparameters, unchanged
lb_size = 2048
n_lstm = 16
max_time = 50
policy_lr = 0.1
lstm_lr = 1e-2
l2_regularization = 1e-6
avg_window = 750


def build_rudder_components(env):
    """Lazily imports rudder.py and builds the LessonBuffer + adapted LSTM.
    Kept inside a function so that VCBM2-only runs work without rudder.py
    or nn.py present in the folder -- the import only happens when
    --policy_learning RUDDER is actually selected."""
    from rudder import LessonBuffer, RRLSTM, to_one_hot

    class RRLSTM2(RRLSTM):
        """RRLSTM with input preparation for the 6-component two-decision
        state: [repaired, express, cond0, cond1, brand, time]. rudder.py is
        unmodified; only feature slicing differs (state_input_size = 9)."""

        def forward(self, input):
            states, actions = input
            repaired = states[:, :, 0:1]
            express = states[:, :, 1:2]
            transport_cond = states[:, :, 2:4]
            brands = to_one_hot(states[:, :, 4], 4)
            t = states[:, :, 5:] / states.shape[1]
            states = torch.cat([repaired, express, transport_cond, brands, t], 2)
            actions = to_one_hot(actions, self.n_actions)
            actions = torch.cat(
                (actions, torch.zeros((actions.shape[0], 1, self.n_actions))), 1)
            x = torch.cat((states, actions), 2)
            lstm_out = self.lstm.forward(x, return_all_seq_pos=True)
            return self.linear(lstm_out[0])

    lesson_buffer = LessonBuffer(size=lb_size, max_time=max_time,
                                 n_features=env.get_state_shape()[-1])
    rudder_lstm = RRLSTM2(state_input_size=9,
                          n_actions=env.get_n_actions()[-1],
                          buffer=lesson_buffer, n_units=n_lstm,
                          lstm_lr=lstm_lr,
                          l2_regularization=l2_regularization,
                          return_scaling=10, lstm_batch_size=8,
                          continuous_pred_factor=0.5)
    return lesson_buffer, rudder_lstm


class QTableActor:
    """Eps-greedy tabular actor for the two-decision environment (used by the
    RUDDER baseline). No action forcing -- neither method is told which
    timesteps matter."""

    def __init__(self, env, lr):
        self.q_table = np.empty(shape=(env.get_n_actions() + env.get_state_max_values()))
        self.q_table[:] = np.nan
        self.q_table[1, :] = 10
        self.q_table[..., 0] = 10
        self.q_table[..., -1] = 0
        self.env = env
        self.state = env.reset()
        self.lr = lr

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def act(self, eps):
        state = self.state
        q_s = self.q_table[(slice(0, None),) + tuple(state)]
        if len(np.unique(q_s[~np.isnan(q_s)])) > 1:
            a = int(np.nanargmax(q_s))
        else:
            a = int(np.random.choice(2))
        if np.random.random() < eps:
            a = int(np.random.choice(2))
        self.state, reward, done, sc = self.env.step(a)
        return self.state, a, reward, done, sc

    def update_direct_q_estimation(self, states, actions, rewards):
        for i in range(actions.shape[0]):
            idx = tuple([actions[i]] + states[i, :].tolist())
            self.q_table[idx] += self.lr * (rewards[i] - self.q_table[idx])


def run_single_seed(rnd_seed, seed_idx=None, total_seeds=1):
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)

    use_wandb = args.use_wandb and WANDB_AVAILABLE
    display_name = "VCBM2" if update_rule == "VCBM2" else "Rudder"
    base_name = args.wandb_name or f"{display_name} - {total_seeds} seeds (two-decision)"
    run_name = base_name if total_seeds == 1 else f"{base_name} (seed {rnd_seed})"

    if use_wandb:
        wandb.init(project=args.wandb_project, group=args.wandb_group,
                   entity=args.wandb_entity, name=run_name, reinit=True,
                   config=dict(policy_learning=update_rule, target=target,
                               max_episodes=max_episodes, rnd_seed=rnd_seed,
                               env="TwoDecisionWatchRepair",
                               vcbm2_warmup_transitions=args.vcbm2_warmup_transitions,
                               vcbm2_classifier_z=args.vcbm2_classifier_z,
                               vcbm2_forced_min_samples=args.vcbm2_forced_min_samples,
                               vcbm2_conf_z=args.vcbm2_conf_z,
                               vcbm2_eps_min=args.vcbm2_eps_min))

    env = TwoDecisionWatchRepair(avg_window=avg_window, transport_time=max_time)

    use_vcbm2 = update_rule == "VCBM2"
    if use_vcbm2:
        ctrl = VCBM2(n_components=env.get_state_shape()[-1],
                     warmup_transitions=args.vcbm2_warmup_transitions,
                     classifier_z=args.vcbm2_classifier_z,
                     forced_min_samples=args.vcbm2_forced_min_samples,
                     conf_z=args.vcbm2_conf_z,
                     min_samples=args.vcbm2_min_samples,
                     eps_start=args.vcbm2_eps_start,
                     eps_decay=args.vcbm2_eps_decay,
                     eps_min=args.vcbm2_eps_min)
        state = env.reset()
    else:
        actor = QTableActor(env, lr=policy_lr)
        lesson_buffer, rudder_lstm = build_rudder_components(env)

    seed_tag = f"[seed {rnd_seed} ({seed_idx}/{total_seeds})] " if total_seeds > 1 else ""
    print(f"\n{seed_tag}Starting training: {update_rule!r} on TwoDecisionWatchRepair")
    print(f"Target: {target*100:.0f}% good decisions over {avg_window}-decision window")
    print("-" * 63)

    episode = 0
    start_time = time.time()
    pbar = tqdm.tqdm(ncols=0)

    while ((max_episodes is None or episode < max_episodes)
           and (len(env.optimal_actions_list) < avg_window
                or np.mean(env.optimal_actions_list) < target)):
        episode += 1

        if use_vcbm2:
            state = env.reset()
            done = False
            ep_return = 0.0
            opportunity_records = []
            while not done:
                a, was_opp = ctrl.select_action(state, episode)
                if was_opp:
                    opportunity_records.append((state.copy(), a))
                next_state, reward, done, sc = env.step(a)
                ctrl.observe_transition(state, a, next_state)
                ep_return += reward
                state = next_state
            ctrl.update_episode(opportunity_records, ep_return)
        else:
            actor.reset()
            done = False
            rewards_ep, states_ep, actions_ep = [], [actor.state], []
            eps_val = max(args.baseline_eps_min,
                          args.baseline_eps_start * (args.baseline_eps_decay ** episode))
            while not done:
                s, a, r, done, sc = actor.act(eps=eps_val)
                actions_ep.append(a)
                states_ep.append(s)
                rewards_ep.append(r)
            states = np.stack(states_ep)
            actions = np.array(actions_ep)
            rewards = np.array(rewards_ep, dtype=float)
            lesson_buffer.add(states=states, actions=actions, rewards=rewards)
            rewards_for_update = rewards
            if (lesson_buffer.different_returns_encountered()
                    and lesson_buffer.full_enough()):
                if episode % 25 == 0:
                    rudder_lstm.train(episode=episode)
                rewards_for_update = rudder_lstm.redistribute_reward(
                    states=np.expand_dims(states, 0),
                    actions=np.expand_dims(actions, 0))[0, :]
                rewards_for_update = np.asarray(
                    rewards_for_update.detach()) if torch.is_tensor(
                    rewards_for_update) else np.asarray(rewards_for_update)
            actor.update_direct_q_estimation(states, actions, rewards_for_update)

        opt_frac = (float(np.mean(env.optimal_actions_list))
                    if env.optimal_actions_list else 0.0)
        poor = env.n_decisions - env.optimal_choices
        pbar.set_description(f"{seed_tag}{episode:7} | poor={poor:6} | {opt_frac:0.4f}")
        pbar.update(1)

        if use_wandb:
            wandb.log({"episode": episode, "poor_decisions": poor,
                       "moving_avg_optimal": opt_frac})

    pbar.close()
    elapsed = time.time() - start_time
    final_pct = np.mean(env.optimal_actions_list) * 100
    converged = (max_episodes is None) or (episode < max_episodes)

    print(f"\n{seed_tag}Done!  runtime={elapsed:.2f}s  episodes={episode}"
          f"  final_good_pct={final_pct:.2f}%  converged={converged}")
    if not converged:
        print(f"{seed_tag}WARNING: hit --max_episodes cap without reaching target.")

    discovery = None
    if use_vcbm2:
        discovery = ctrl.summary()
        print("\nVCBM2 discovered component labels:",
              discovery["component_labels"])
        print("VCBM2 per-decision-point value estimates:")
        for row in discovery["keys"]:
            print(f"  key={row['key']}: "
                  f"a0 mean={row['mean_a0']:+.2f} (n={row['n_a0']})  "
                  f"a1 mean={row['mean_a1']:+.2f} (n={row['n_a1']})  "
                  f"greedy=a{row['greedy_action']}")

    if use_wandb:
        wandb.log({"final_good_pct": final_pct, "episodes": episode,
                   "elapsed": elapsed, "converged": converged})
        wandb.finish()

    return dict(seed=rnd_seed, episodes=episode, elapsed=elapsed,
                final_pct=final_pct, converged=converged, discovery=discovery)


if __name__ == "__main__":
    results = []
    for i, seed in enumerate(range(1, n_seeds + 1), start=1):
        results.append(run_single_seed(seed, seed_idx=i, total_seeds=n_seeds))

    if n_seeds > 1:
        n_converged = sum(1 for r in results if r["converged"])
        conv = [r for r in results if r["converged"]]
        episodes_arr = np.array([r["episodes"] for r in conv], dtype=float)
        elapsed_arr = np.array([r["elapsed"] for r in conv], dtype=float)
        final_arr = np.array([r["final_pct"] for r in results], dtype=float)

        print("\n" + "=" * 63)
        print(f"MULTI-SEED SUMMARY  ({update_rule} on two-decision env, n_seeds={n_seeds})")
        print("=" * 63)
        print(f"Converged: {n_converged}/{n_seeds} seeds")
        if n_converged:
            print(f"Episodes to converge : {episodes_arr.mean():.1f} +/- {episodes_arr.std():.1f}"
                  f"   (min={episodes_arr.min():.0f}, max={episodes_arr.max():.0f})"
                  f"   [converged seeds only]")
            print(f"Wall-clock time (s)  : {elapsed_arr.mean():.2f} +/- {elapsed_arr.std():.2f}")
        print(f"Final good-decision %: {final_arr.mean():.2f} +/- {final_arr.std():.2f}  [all seeds]")