"""
MarsRoverEnvironment -- a tough, long-horizon credit-assignment stress test
for RUDDER, extending the single-decision Watch-Repair idea to THREE
sequential decisions spread across a much longer mission, with a fully
sparse terminal reward.

Motivation: Watch-Repair has exactly one meaningful decision (repair/pass
at t=0), so any method hardcoded to credit t=0 looks like it "solves"
credit assignment there. This environment goes further:
  - THREE decision points instead of two, spread over a 150-sol mission
    (vs. 50 sols), so the credit-assignment gap between a decision and the
    terminal reward is much larger.
  - The reward is 100% terminal -- no partial costs are paid at the decision
    points themselves (unlike Watch-Repair, where the repair cost is charged
    immediately at t=0). Every consequence is deferred to mission end.
  - The decisions are conditionally entangled three ways: the value of the
    route choice (t=40) depends on the power mode chosen at t=0, and the
    value of the drill choice (t=90) depends on BOTH earlier decisions (via
    remaining battery and whether the rover was damaged). A method must
    discover all three decision points AND their interaction, not just
    treat them as independent single-step credit-assignment problems.

Mission narrative
------------------
A rover is dropped onto one of several terrain types (analogous to "brand"
in Watch-Repair -- a static, decision-relevant context revealed at t=0).

  t=0   (POWER decision): action 0 = high-power mode (fast, enables deep
        drilling later, but drains the battery faster and leaves the rover
        more exposed to dust damage), action 1 = low-power/conservative
        mode (slow drain, safer, but insufficient reserves for deep
        drilling later regardless of what is chosen at t=90).

  t=40  (ROUTE decision, only meaningful if battery remains): action 0 =
        risky shortcut across the crater rim (saves battery, but risks
        dust-storm damage; risk is higher if high-power mode was chosen,
        since the rover's shielding is degraded), action 1 = safe detour
        (costs extra battery, never causes damage).

  t=90  (DRILL decision, only meaningful if battery remains and the rover
        reached this point undamaged): action 0 = deep drill (large
        battery cost, but high science value -- ONLY if the rover has
        enough battery left AND was not damaged by the route choice),
        action 1 = shallow sample (small battery cost, modest guaranteed
        science value).

  All other timesteps: actions have no effect (pure filler, exactly like
  the "all other timesteps" in two_decision_env.py).

Reward structure (entirely terminal, paid at t=mission_length):
  final_reward = science_value - damage_penalty - battery_penalty

  - science_value: from the drill decision, modulated by terrain and by
    whether battery/damage preconditions were met.
  - damage_penalty: incurred if the risky route was taken AND a dust-storm
    event fired (probability depends on power mode).
  - battery_penalty: incurred if the rover's battery went negative at any
    point (harsh terminal penalty), i.e. mode+route+drill choices were too
    aggressive for the available power budget.

State: [power_mode, damaged, battery_low, terrain, time]
  power_mode  : 0/1, set at t=0 (0=unset/high still ambiguous until t=0
                acted; see reset()), part of state from t=0 onward.
  damaged     : 0/1, whether a dust-storm hit on the risky route.
  battery_low : 0/1, coarse discretized signal (derived from continuous
                battery level) so the state stays small enough for a
                tabular Q-table, matching the discretization style of
                transport_condition in Watch-Repair.
  terrain     : static context revealed at reset(), analogous to "brand".
  time        : sol counter.

step() returns the same 4-tuple interface as Watch-Repair / TwoDecisionWatchRepair:
    (state, reward, done, state_changed)
"""

import numpy as np


class MarsRoverEnvironment:

    def __init__(self, avg_window, mission_length=150,
                 power_time=0, route_time=40, drill_time=90):
        self.list = []

        # Terrain types the rover can land on (static per-episode context,
        # analogous to "brand" in Watch-Repair).
        self.n_terrains = 4
        self.terrain_probability = np.array([0.25, 0.25, 0.25, 0.25])

        # Base science value of a successful deep drill per terrain, and of
        # the always-safe shallow sample per terrain.
        self.deep_drill_science = np.array([40.0, 65.0, 90.0, 55.0])
        self.deep_drill_variance = np.array([4.0, 5.0, 6.0, 5.0])
        self.shallow_sample_science = np.array([12.0, 14.0, 13.0, 15.0])
        self.shallow_sample_variance = np.array([1.5, 1.5, 1.5, 1.5])

        # Battery model. Both modes drain every sol; high-power drains
        # faster but is required to reach t=90 with enough reserve for a
        # deep drill on terrains where that is the better play.
        self.battery_capacity = 100.0
        self.drain_per_sol = {0: 0.55, 1: 0.30}  # keyed by power_mode
        self.route_extra_cost = np.array([2.0, 9.0])  # [risky, safe]
        self.drill_cost = np.array([28.0, 6.0])  # [deep, shallow]

        # Dust-storm damage risk on the risky route, depends on power mode:
        # high-power mode degrades shielding, so risk is higher.
        self.damage_probability_by_power_mode = np.array([0.35, 0.12])

        # Timesteps of the three decision points.
        self.power_time = power_time
        self.route_time = route_time
        self.drill_time = drill_time
        self.mission_length = mission_length

        # Instance variables which will be part of the state.
        self.terrain = -1
        self.power_mode = None
        self.route_taken = None
        self.damaged = None
        self.battery = None
        self.time = None

        # For tracking performance across all three decision points.
        # Optimal power mode per terrain: high-power (0) needed to afford
        # a deep drill on the two terrains where that pays off (1, 2);
        # low-power (1) is safer on terrains where deep drilling never
        # pays off net of risk (0, 3).
        self.optimal_power = np.array([1, 0, 0, 1])
        # Optimal route per terrain: only terrains where deep drilling is
        # the long-run plan (1, 2) benefit from banking battery via the
        # safe route (1); otherwise the risky shortcut (0) is free upside
        # since the shallow-sample plan does not need the saved battery.
        self.optimal_route = np.array([0, 1, 1, 0])
        # Optimal drill choice per terrain, conditioned on the rover
        # actually having enough reserve and being undamaged (checked
        # dynamically in step(); this array records the "ideal world"
        # optimum used only for the performance-tracking heuristic).
        self.optimal_drill = np.array([1, 0, 0, 1])

        self.optimal_choices = 0
        self.optimal_actions_list = []
        self.n_decisions = 0
        self.avg_window = avg_window

    def reset(self):
        self.terrain = np.random.choice(
            self.n_terrains, size=1, p=self.terrain_probability)[0]
        self.power_mode = 0  # placeholder until acted upon at t=0
        self.route_taken = 0
        self.damaged = 0
        self.battery = self.battery_capacity
        self.time = 0
        return self._state()

    def _state(self):
        battery_low = int(self.battery < 35.0)
        return np.array(
            [self.power_mode, self.damaged, battery_low, self.terrain,
             self.time], dtype=np.int32)

    def _track(self, action, optimal):
        self.n_decisions += 1
        good = bool(action == optimal)
        if good:
            self.optimal_choices += 1
        self.optimal_actions_list.append(good)
        if len(self.optimal_actions_list) > self.avg_window:
            del self.optimal_actions_list[0]

    def step(self, action):
        reward = 0.0
        prev_state = self._state()

        if self.time == self.power_time:
            # action 0 = high-power, action 1 = low-power/conservative.
            self.power_mode = int(action == 0)
            self._track(action, self.optimal_power[self.terrain])

        elif self.time == self.route_time:
            # action 0 = risky shortcut, action 1 = safe detour.
            self.route_taken = int(action == 0)
            if self.route_taken:
                if np.random.random() < self.damage_probability_by_power_mode[self.power_mode]:
                    self.damaged = 1
            self.battery -= self.route_extra_cost[0 if self.route_taken else 1]
            self._track(action, self.optimal_route[self.terrain])

        elif self.time == self.drill_time:
            # action 0 = deep drill, action 1 = shallow sample.
            deep = int(action == 0)
            can_deep_drill = (not self.damaged) and (
                    self.battery - self.drill_cost[0] >= 0)
            if deep and can_deep_drill:
                science = np.random.normal(
                    self.deep_drill_science[self.terrain],
                    self.deep_drill_variance[self.terrain])
                self.battery -= self.drill_cost[0]
            else:
                science = np.random.normal(
                    self.shallow_sample_science[self.terrain],
                    self.shallow_sample_variance[self.terrain])
                self.battery -= self.drill_cost[1]
            self._pending_science = science
            self._track(action, self.optimal_drill[self.terrain])

        # Battery drains every sol regardless of action.
        self.battery -= self.drain_per_sol[self.power_mode]

        self.time += 1
        done = self.time == self.mission_length

        if done:
            science_value = getattr(self, "_pending_science", 0.0)
            damage_penalty = 30.0 if self.damaged else 0.0
            battery_penalty = 50.0 if self.battery < 0 else 0.0
            reward += science_value - damage_penalty - battery_penalty
            self.list.append(science_value)

        new_state = self._state()
        state_changed = bool(np.any(new_state[:-1] != prev_state[:-1]))
        return new_state, reward, done, state_changed

    def get_state_max_values(self):
        return [2, 2, 2, self.n_terrains, self.mission_length + 1]

    def get_state_shape(self):
        return [5]

    def get_n_actions(self):
        return [2]


class TabularActor:

    def __init__(self, env, lr):
        self.q_table = np.empty(shape=(env.get_n_actions() + env.get_state_max_values()))
        self.q_table[:] = np.nan
        self.q_table[1, :] = 10  # Optimistic initialization
        self.q_table[..., 0] = 10  # Optimistic initialization
        self.q_table[..., -1] = 0
        self.env = env
        self.state = env.reset()
        self.lr = lr

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def act(self, eps=0.10):
        state = self.state
        q_s = self.q_table[(slice(0, None),) + tuple(state)]
        valid = ~np.isnan(q_s)
        if np.any(valid) and len(np.unique(q_s[valid])) > 1:
            a = int(np.nanargmax(q_s))
        else:
            a = int(np.random.choice(2))

        if np.random.random() < eps:
            a = int(np.random.choice(2))

        self.state, reward, done, state_changed = self.env.step(a)
        return self.state, a, reward, done, state_changed

    # Three different update rules to change the policy:

    # "RUDDER learning"
    # Direct Q-Value estimation, using redistributed reward of RUDDER
    def update_direct_q_estimation(self, states, actions, rewards):
        for i in range(actions.shape[0]):
            idx = tuple([actions[i]] + states[i, :].tolist())
            self.q_table[idx] += self.lr * (rewards[i] - self.q_table[idx])

    # Q-Learning update
    def update_q_learning(self, states, actions, rewards):
        indices = np.concatenate([np.expand_dims(np.array(actions.tolist() + [0]), 1), states], axis=1)
        maxq = [self.q_table[tuple([slice(0, None), *indices[i, 1:]])] for i in range(indices.shape[0])]
        maxq = np.nanmax(np.array(maxq), axis=1)
        for t in range(actions.shape[0]):
            self.q_table[tuple(indices[t, :])] = (1 - self.lr) * self.q_table[tuple(indices[t, :])] + self.lr * (
                    rewards[t] + maxq[t + 1])

    # Monte-Carlo control update
    def update_monte_carlo(self, states, actions, rewards):
        gt = rewards[::-1].cumsum()[::-1]
        for i in range(actions.shape[0]):
            idx = tuple([actions[i]] + states[i, :].tolist())
            self.q_table[idx] += self.lr * (gt[i] - self.q_table[idx])
