"""
watch_repair.py -- VCBM (learned bookmark discovery) on the two-decision
Watch-Repair environment.

Unlike the original single-decision Watch-Repair task, this is an honest
test of long-horizon credit assignment: there are TWO meaningful decision
points (t=0 repair/pass, t=25 express/standard for repaired items), and
VCBM is not told where they are.

VCBM must DISCOVER both decision points via bookmark detection (see
vcbm.py) before it can estimate their values.

This script is fully self-contained: the environment (TwoDecisionWatchRepair)
is defined directly below rather than imported from a separate file, and
there is no RUDDER comparison path -- this folder intentionally has no
rudder.py.

Usage:
  python watch_repair.py --target 0.9907 --n_seeds 10
"""

import argparse
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from vcbm import VCBM2

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TwoDecisionWatchRepair:
    """Two-decision extension of the Watch-Repair credit-assignment task.

    Decision structure:
      t=0  : action 0 = repair (charged immediately), action 1 = pass.
             Optimal (same as the original single-decision task): pass
             brands 0, 3; repair brands 1, 2.
      t=25 : only if repaired. action 0 = express shipping (cost charged
             immediately), action 1 = standard.
             Express reduces the brand-related transport cost at delivery.
             Configured so express is worth it for brand 2 but NOT brand 1:
               brand 1: reduction 1.0 < express cost 2.0 -> standard optimal
               brand 2: reduction 4.0 > express cost 2.0 -> express optimal
      All other timesteps: actions have no effect.

    State: [repaired, express, cond0, cond1, brand, time]
    step() returns (state, reward, done, state_changed).
    """

    def __init__(self, avg_window, transport_time=50, express_time=25,
                 express_cost=2.0,
                 express_reduction=np.array([0.0, 1.0, 4.0, 0.0])):
        self.list = []

        self.n_brands = 4
        self.brand_appearance_probability = np.array([0.25, 0.25, 0.25, 0.25])
        self.brand_sale_price = np.array([18, 28, 31, 59])

        self.average_brand_related_transport_cost = np.array([0.5, 2.5, 4.5, 18])
        self.brand_related_transport_cost_variance = np.array([1.5, 1.5, 1.5, 1.5])

        self.average_brand_repair_price = np.array([1, 4, 5, 24.5])
        self.brand_repair_variance = np.array([2, 2, 2, 1])

        self.n_transport_conditions = 2
        self.transport_time = transport_time
        self.transport_cost = np.array([0.1, 7])
        self.transport_condition_probability = np.array([0.1, 0.05])

        # Second decision configuration
        self.express_time = express_time
        self.express_cost = express_cost
        self.express_reduction = express_reduction

        self.brand = -1
        self.repaired = None
        self.express = None
        self.time = None
        self.transport_condition = None

        # Optimal actions for performance tracking:
        #  decision 1 (t=0): same as original
        self.optimal_actions_t0 = np.array([1, 0, 0, 1])
        #  decision 2 (t=express_time, only when repaired):
        #  express (0) iff reduction > cost
        self.optimal_actions_express = np.array(
            [0 if express_reduction[b] > express_cost else 1 for b in range(4)])
        self.optimal_choices = 0
        self.optimal_actions_list = []
        self.n_decisions = 0
        self.avg_window = avg_window

    def reset(self):
        self.brand = np.random.choice(
            self.n_brands, size=1, p=self.brand_appearance_probability)[0]
        self.repaired = 0
        self.express = 0
        self.time = 0
        self.transport_condition = np.zeros(shape=(self.n_transport_conditions,))
        return self._state()

    def _state(self):
        return np.array([self.repaired, self.express]
                        + self.transport_condition.tolist()
                        + [self.brand, self.time], dtype=np.int32)

    def _track(self, action, optimal):
        self.n_decisions += 1
        good = (action == optimal)
        if good:
            self.optimal_choices += 1
        self.optimal_actions_list.append(good)
        if len(self.optimal_actions_list) > self.avg_window:
            del self.optimal_actions_list[0]

    def step(self, action):
        reward = 0
        prev_repaired = self.repaired
        prev_express = self.express

        if self.time == 0:
            self.repaired = int(action == 0)
            if self.repaired:
                repair_price = np.random.normal(
                    self.average_brand_repair_price[self.brand],
                    self.brand_repair_variance[self.brand])
                reward -= repair_price
            self._track(action, self.optimal_actions_t0[self.brand])

        elif self.time == self.express_time and self.repaired:
            self.express = int(action == 0)
            if self.express:
                reward -= self.express_cost
            self._track(action, self.optimal_actions_express[self.brand])

        self.time += 1
        done = self.time == self.transport_time

        transport_cond = self.transport_condition
        transport_cond += np.array(
            [np.random.random() < self.transport_condition_probability[i]
             for i in range(self.n_transport_conditions)])
        self.transport_condition = transport_cond

        if done and self.repaired:
            transport_cost = np.sum(transport_cond * self.transport_cost)
            brand_related = np.random.normal(
                self.average_brand_related_transport_cost[self.brand],
                self.brand_related_transport_cost_variance[self.brand])
            if self.express:
                brand_related -= self.express_reduction[self.brand]
            reward += self.brand_sale_price[self.brand] - transport_cost - brand_related
            self.list.append(transport_cost)

        state_changed = bool(self.repaired != prev_repaired
                             or self.express != prev_express)
        return self._state(), reward, done, state_changed

    def get_state_max_values(self):
        return ([2, 2]
                + np.repeat(self.transport_time, self.n_transport_conditions).tolist()
                + [self.n_brands, self.transport_time + 1])

    def get_state_shape(self):
        return [4 + self.n_transport_conditions]

    def get_n_actions(self):
        return [2]


parser = argparse.ArgumentParser(description="Two-decision Watch Repair — VCBM")
parser.add_argument("--max_episodes", default=20000, type=int,
                    help="Hard cap per seed (0 for unlimited).")
parser.add_argument("--target", default=0.9907, type=float)
parser.add_argument("--n_seeds", default=10, type=int)
parser.add_argument("--use_wandb", action="store_true", default=True)
parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
parser.add_argument("--wandb_project", default="Dissertation", type=str)
parser.add_argument("--wandb_group", default="VCBM: Rudder vs Vcbm", type=str)
parser.add_argument("--wandb_entity", default=None, type=str)
parser.add_argument("--wandb_name", default="VCBM- WR", type=str)

# VCBM hyperparameters
parser.add_argument("--vcbm2_warmup_transitions", default=2000, type=int)
parser.add_argument("--vcbm2_classifier_z", default=4.0, type=float)
parser.add_argument("--vcbm2_forced_min_samples", default=300, type=int)
parser.add_argument("--vcbm2_conf_z", default=2.0, type=float)
parser.add_argument("--vcbm2_min_samples", default=8, type=int)
parser.add_argument("--vcbm2_eps_start", default=0.20, type=float)
parser.add_argument("--vcbm2_eps_decay", default=0.995, type=float)
parser.add_argument("--vcbm2_eps_min", default=0.005, type=float)
args = parser.parse_args()

max_episodes = args.max_episodes if args.max_episodes != 0 else None
target = args.target
n_seeds = max(1, args.n_seeds)

max_time = 50
avg_window = 750


def run_single_seed(rnd_seed, use_wandb, seed_idx=None, total_seeds=1):
    np.random.seed(rnd_seed)

    env = TwoDecisionWatchRepair(avg_window=avg_window, transport_time=max_time)

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

    seed_tag = f"[seed {rnd_seed} ({seed_idx}/{total_seeds})] " if total_seeds > 1 else ""
    print(f"\n{seed_tag}Starting training: 'VCBM' on TwoDecisionWatchRepair")
    print(f"Target: {target*100:.0f}% good decisions over {avg_window}-decision window")
    print("-" * 63)

    episode = 0
    start_time = time.time()
    pbar = tqdm.tqdm(ncols=0)

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
            next_state, reward, done, sc = env.step(a)
            ctrl.observe_transition(state, a, next_state)
            ep_return += reward
            state = next_state
        ctrl.update_episode(opportunity_records, ep_return)

        opt_frac = (float(np.mean(env.optimal_actions_list))
                    if env.optimal_actions_list else 0.0)
        poor = env.n_decisions - env.optimal_choices
        pbar.set_description(f"{seed_tag}{episode:7} | poor={poor:6} | {opt_frac:0.4f}")
        pbar.update(1)

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
                final_pct=final_pct, converged=converged, discovery=discovery)


if __name__ == "__main__":
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, group=args.wandb_group,
                   entity=args.wandb_entity, name=args.wandb_name,
                   config=dict(policy_learning="VCBM", target=target,
                               max_episodes=max_episodes, n_seeds=n_seeds,
                               env="TwoDecisionWatchRepair",
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
        print(f"MULTI-SEED SUMMARY  (VCBM on two-decision env, n_seeds={n_seeds})")
        print("=" * 63)
        print(f"Converged: {n_converged}/{n_seeds} seeds")
        if n_converged:
            print(f"Episodes to converge : {episodes_arr.mean():.1f} +/- {episodes_arr.std():.1f}"
                  f"   (min={episodes_arr.min():.0f}, max={episodes_arr.max():.0f})"
                  f"   [converged seeds only]")
            print(f"Wall-clock time (s)  : {elapsed_arr.mean():.2f} +/- {elapsed_arr.std():.2f}")
        print(f"Final good-decision %: {final_arr.mean():.2f} +/- {final_arr.std():.2f}  [all seeds]")

    if use_wandb:
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
