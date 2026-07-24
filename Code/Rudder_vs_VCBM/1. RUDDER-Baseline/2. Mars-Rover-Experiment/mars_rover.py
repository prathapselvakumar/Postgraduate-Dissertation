"""
mars_rover.py -- RUDDER (vs. MC / TD baselines) on MarsRoverEnvironment.

Runs --n_seeds independent training seeds, all logged to a SINGLE wandb run
(project "Dissertation", group "Rudder : Rudder vs Vcbm", run name
"Rudder - ME"), with each seed's moving-average curve logged under its own
key (moving_avg_optimal/seed_<n>, poor_decisions/seed_<n>) so all 10 seeds
are visible as separate lines on the same charts within that one run.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from mars_rover_environment import MarsRoverEnvironment, TabularActor
from rudder import LessonBuffer
from rudder import RRLSTM
from rudder import to_one_hot
import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Code/Rudder_vs_VCBM/1. RUDDER-Baseline/2. Mars-Rover-Experiment/mars_rover.py
#   -> Code/Rudder_vs_VCBM/3. Outputs/Mars-Rover-Experiment/Rudder/Graphs/...
_OUTPUTS_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "3. Outputs",
    "Mars-Rover-Experiment", "Rudder", "Graphs"))
LOCAL_GRAPH_DIR = os.path.join(_OUTPUTS_ROOT, "Local-Graph")
WANDB_GRAPH_DIR = os.path.join(_OUTPUTS_ROOT, "Wandb")

parser = argparse.ArgumentParser(description='Rudder Demonstration -- Mars Rover')

parser.add_argument('--policy_learning', default="RUDDER", type=str, dest="pl", required=True,
                    help="The policy learning method. Valid options are \"RUDDER\", \"MC\" and \"TD\"",
                    choices=["RUDDER", "TD", "MC"])
parser.add_argument('--n_seeds', default=10, type=int)
parser.add_argument('--use_wandb', action="store_true", default=True)
parser.add_argument('--no_wandb', dest="use_wandb", action="store_false")
parser.add_argument('--wandb_project', default="Dissertation", type=str)
parser.add_argument('--wandb_group', default="Rudder : Rudder vs Vcbm", type=str)
parser.add_argument('--wandb_name', default="Rudder - ME", type=str)
args = parser.parse_args()

update_rule = args.pl
n_seeds = max(1, args.n_seeds)

if update_rule == "RUDDER":
    improve_policy = TabularActor.update_direct_q_estimation
    redistribute_reward = True
elif update_rule == "MC":
    improve_policy = TabularActor.update_monte_carlo
    redistribute_reward = False
elif update_rule == "TD":
    improve_policy = TabularActor.update_q_learning
    redistribute_reward = False
else:
    raise ValueError("Specify an update rule.")

lb_size = 2048
n_lstm = 24
mission_length = 150
policy_lr = 0.1
lstm_lr = 1e-2
l2_regularization = 1e-6
avg_window = 750

use_wandb = args.use_wandb and WANDB_AVAILABLE
if use_wandb:
    wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_name,
               config=dict(policy_learning=update_rule, n_seeds=n_seeds, avg_window=avg_window,
                           mission_length=mission_length, policy_lr=policy_lr, lstm_lr=lstm_lr,
                           l2_regularization=l2_regularization, n_lstm=n_lstm))


class RRLSTMMarsRover(RRLSTM):
    """RRLSTM with input preparation for the 5-component mars-rover state:
    [power_mode, damaged, battery_low, terrain, time]. rudder.py is
    unmodified; only feature slicing differs (state_input_size = 8: 3
    binary flags + 4 one-hot terrain + 1 time)."""

    def forward(self, input):
        states, actions = input
        power_mode = states[:, :, 0:1]
        damaged = states[:, :, 1:2]
        battery_low = states[:, :, 2:3]
        terrains = to_one_hot(states[:, :, 3], 4)
        t = states[:, :, 4:] / states.shape[1]
        states = torch.cat([power_mode, damaged, battery_low, terrains, t], 2)
        actions = to_one_hot(actions, self.n_actions)
        actions = torch.cat((actions, torch.zeros((actions.shape[0], 1, self.n_actions))), 1)
        x = torch.cat((states, actions), 2)
        lstm_out = self.lstm.forward(x, return_all_seq_pos=True)
        return self.linear(lstm_out[0])


def run_single_seed(rnd_seed):
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)

    env = MarsRoverEnvironment(avg_window=avg_window, mission_length=mission_length)
    state = env.reset()

    # Initialize LSTM Actor as well as the LessonBuffer.
    eps_agent = TabularActor(env, lr=policy_lr)
    lesson_buffer, rudder_lstm = None, None
    if redistribute_reward:
        lesson_buffer = LessonBuffer(size=lb_size, max_time=mission_length, n_features=env.get_state_shape()[-1])
        rudder_lstm = RRLSTMMarsRover(state_input_size=8, n_actions=env.get_n_actions()[-1], buffer=lesson_buffer,
                                      n_units=n_lstm, lstm_lr=lstm_lr, l2_regularization=l2_regularization,
                                      return_scaling=20, lstm_batch_size=8, continuous_pred_factor=0.5)

    start_time = time.time()
    all_suboptimal_actions = []
    episode = 0

    print(f"\n[seed {rnd_seed}] Starting training using update rule: \"{update_rule}\"\n")
    print("---------------------------------------------------------------")
    print("Episode |    # poor | terrain/ | % good decisions | runtime stats")
    print("        | decisions |   action |      goal: > 95% |              ")
    pbar = tqdm.tqdm(desc="        |           |        n |                  | ", ncols=0)

    while len(env.optimal_actions_list) < avg_window or np.mean(env.optimal_actions_list) < 0.95:
        episode += 1
        state = eps_agent.reset()
        done = False
        rewards = []
        states = [state]
        actions = []
        while not done:
            state, a, reward, done, state_changed = eps_agent.act()
            actions.append(a)
            states.append(state)
            rewards.append(reward)
            if done:
                eps_agent.reset()
                states = np.stack(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                if redistribute_reward:
                    lesson_buffer.add(states=states, actions=actions, rewards=rewards)
                    if lesson_buffer.different_returns_encountered() and lesson_buffer.full_enough():
                        if episode % 25 == 0:
                            rudder_lstm.train(episode=episode)
                        rewards = rudder_lstm.redistribute_reward(states=np.expand_dims(states, 0),
                                                                  actions=np.expand_dims(actions, 0))[0, :]
                improve_policy(eps_agent, states, actions, rewards)

                poor = episode * 3 - env.optimal_choices
                opt_frac = float(np.mean(env.optimal_actions_list))
                desc = f"{episode:7} | {poor:9} |    {states[0, -2]}/{actions[0]} |           {opt_frac:0.4f} |"
                pbar.set_description(desc)
                pbar.update(1)
                pbar.refresh()

                all_suboptimal_actions.append(poor)

                if use_wandb:
                    wandb.log({f"poor_decisions/seed_{rnd_seed}": poor,
                               f"moving_avg_optimal/seed_{rnd_seed}": opt_frac,
                               f"episode/seed_{rnd_seed}": episode})

    pbar.close()
    elapsed = time.time() - start_time
    final_pct = np.mean(env.optimal_actions_list) * 100
    print(f"\n[seed {rnd_seed}] Done! (runtime: {elapsed:.2f}s, episodes: {episode}, final: {final_pct:.2f}%)")

    return dict(seed=rnd_seed, episodes=episode, elapsed=elapsed, final_pct=final_pct,
               all_suboptimal_actions=all_suboptimal_actions)


if __name__ == "__main__":
    results = [run_single_seed(seed) for seed in range(1, n_seeds + 1)]

    episodes_arr = np.array([r["episodes"] for r in results], dtype=float)
    elapsed_arr = np.array([r["elapsed"] for r in results], dtype=float)
    final_arr = np.array([r["final_pct"] for r in results], dtype=float)

    print("\n" + "=" * 63)
    print(f"MULTI-SEED SUMMARY  ({update_rule} on Mars-Rover, n_seeds={n_seeds})")
    print("=" * 63)
    print(f"Episodes to converge : {episodes_arr.mean():.1f} +/- {episodes_arr.std():.1f}"
          f"   (min={episodes_arr.min():.0f}, max={episodes_arr.max():.0f})")
    print(f"Wall-clock time (s)  : {elapsed_arr.mean():.2f} +/- {elapsed_arr.std():.2f}")
    print(f"Final good-decision %: {final_arr.mean():.2f} +/- {final_arr.std():.2f}")

    run_path = None
    if use_wandb:
        run_path = wandb.run.path
        wandb.log({"summary/episodes_mean": episodes_arr.mean(),
                   "summary/episodes_std": episodes_arr.std(),
                   "summary/elapsed_mean": elapsed_arr.mean(),
                   "summary/elapsed_std": elapsed_arr.std(),
                   "summary/final_pct_mean": final_arr.mean(),
                   "summary/final_pct_std": final_arr.std()})
        wandb.finish()

    # Local graph: all seeds overlaid on one figure, saved (not shown) to
    # Code/Rudder_vs_VCBM/3. Outputs/Mars-Rover-Experiment/Rudder/Graphs/Local-Graph
    os.makedirs(LOCAL_GRAPH_DIR, exist_ok=True)
    fig, ax = plt.subplots()
    for r in results:
        ax.plot(np.arange(len(r["all_suboptimal_actions"])), r["all_suboptimal_actions"],
               label=f"seed {r['seed']}", alpha=0.8)
    ax.set_title(f"Total number of poor decisions ({update_rule}, {n_seeds} seeds)")
    ax.set_ylabel("#poor decisions")
    ax.set_xlabel("#episodes")
    ax.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    local_path = os.path.join(LOCAL_GRAPH_DIR, f"mars_rover_{update_rule.lower()}_all_seeds.png")
    fig.savefig(local_path)
    plt.close(fig)
    print(f"Saved local graph to: {local_path}")

    # Wandb graph: re-plot each logged metric's history (pulled via the
    # wandb API, since wandb has no scripted "download chart PNG" endpoint)
    # and save into Code/Rudder_vs_VCBM/3. Outputs/Mars-Rover-Experiment/Rudder/Graphs/Wandb
    if use_wandb and run_path:
        os.makedirs(WANDB_GRAPH_DIR, exist_ok=True)
        api = wandb.Api()
        run = api.run(run_path)
        # scan_history() returns plain dicts (no pandas dependency). Each
        # wandb.log() call here only logs one seed's keys per step, so a
        # keys= filter (which requires ALL listed keys in the same row)
        # would match nothing -- scan unfiltered and bucket by seed instead.
        all_rows = list(run.scan_history())
        for metric in ["moving_avg_optimal", "poor_decisions"]:
            series_by_seed = {s: [] for s in range(1, n_seeds + 1)}
            for row in all_rows:
                for s in range(1, n_seeds + 1):
                    val = row.get(f"{metric}/seed_{s}")
                    if val is not None:
                        series_by_seed[s].append(val)
            fig, ax = plt.subplots()
            for s in range(1, n_seeds + 1):
                values = series_by_seed[s]
                if values:
                    ax.plot(np.arange(len(values)), values, label=f"seed {s}", alpha=0.8)
            ax.set_title(f"{metric} ({update_rule}, {n_seeds} seeds) -- from wandb")
            ax.set_ylabel(metric)
            ax.set_xlabel("logged step")
            ax.legend(fontsize="small", ncol=2)
            plt.tight_layout()
            wandb_path = os.path.join(WANDB_GRAPH_DIR, f"{metric}_{update_rule.lower()}.png")
            fig.savefig(wandb_path)
            plt.close(fig)
            print(f"Saved wandb graph to: {wandb_path}")
