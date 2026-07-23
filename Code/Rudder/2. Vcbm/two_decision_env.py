"""
TwoDecisionWatchRepair -- an extension of the RUDDER demonstration
Watch-Repair environment with a SECOND, mid-episode decision point.

Purpose: the original environment has exactly one meaningful decision
(repair/pass at t=0), so any method hardcoded to credit t=0 looks like it
"solves" credit assignment there. This variant adds a shipping decision at
t=25 (express vs standard, only meaningful for repaired watches), so a
method must DISCOVER both decision points to reach optimal behaviour.

Decision structure:
  t=0  : action 0 = repair (charged immediately), action 1 = pass.
         Optimal (same as original): pass brands 0,3; repair brands 1,2.
  t=25 : only if repaired. action 0 = express shipping (cost charged
         immediately), action 1 = standard.
         Express reduces the brand-related transport cost at delivery.
         Configured so express is worth it for brand 2 but NOT brand 1:
           brand 1: reduction 1.0 < express cost 2.0 -> standard optimal
           brand 2: reduction 4.0 > express cost 2.0 -> express optimal
  All other timesteps: actions have no effect.

State: [repaired, express, cond0, cond1, brand, time]
step() returns (state, reward, done, state_changed) -- the same 4-tuple
interface as the modified single-decision environment used by watch_pair.py.
"""

import numpy as np


class TwoDecisionWatchRepair:

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