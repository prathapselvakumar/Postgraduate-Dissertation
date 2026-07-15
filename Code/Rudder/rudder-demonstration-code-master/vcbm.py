"""
VCBM (Visual/Value Causal Chain Bookmarking) -- a lightweight alternative to
RUDDER's LSTM-based reward redistribution for single-decision-point episodes
like the Watch-Repair environment.

Structured to mirror rudder.py's interface (a stats/buffer class + a
redistributor-style class with train()/act()-adjacent methods) so it can be
swapped in alongside RUDDER in the same training loop.

Core idea: instead of learning a general-purpose sequence model to figure out
which timestep "caused" the final reward (RUDDER's approach), VCBM assumes
the credit-assignment structure is simple -- one meaningful decision per
episode, identified by a discrete key (here: the brand) -- and tracks
per-key running statistics (mean, variance) of the repair return directly.
This is far cheaper than training an LSTM, but only works because that
single-decision-point assumption happens to hold for this environment.
"""

import numpy as np


class VCBMStats:
    """
    Tracks per-key (e.g. per-brand) running mean and variance of the
    "repair" return using Welford's algorithm, plus a sample counter.
    Analogous role to rudder.py's LessonBuffer, but stores compact
    running statistics instead of raw trajectories.
    """

    def __init__(self, n_keys):
        self.n_keys = n_keys
        self.counts = np.zeros(n_keys)
        self.mean = np.zeros(n_keys)
        self.m2 = np.zeros(n_keys)  # Welford's running sum of squared deviations

    def update(self, key, value):
        """Incorporate one new observed return for the given key."""
        n = self.counts[key]
        n_new = n + 1
        delta = value - self.mean[key]
        self.mean[key] += delta / n_new
        delta2 = value - self.mean[key]
        self.m2[key] += delta * delta2
        self.counts[key] = n_new

    def standard_error(self, key):
        """
        Standard error of the mean estimate for this key. Returns +inf when
        too few samples exist to estimate variance, so confidence checks
        correctly refuse to fire on unreliable estimates.
        """
        n = self.counts[key]
        if n < 2:
            return float('inf')
        variance = self.m2[key] / (n - 1)
        return float(np.sqrt(variance / n))

    def z_score(self, key):
        """|mean| / standard_error -- how many standard errors the mean is
        from zero. Used as a statistically-grounded confidence measure,
        instead of judging every key against the same fixed magnitude
        regardless of how reliable that magnitude is."""
        se = self.standard_error(key)
        if se <= 0 or se == float('inf'):
            return 0.0 if se == float('inf') else float('inf')
        return abs(self.mean[key]) / se


class VCBM:
    """
    VCBM redistribution/exploration controller. Mirrors rudder.py's RRLSTM
    class role (something the training loop calls to get an action-selection
    policy and to update its internal model after each episode), but with no
    neural network: it directly tracks per-brand repair-return statistics
    and uses them both to bias the Q-table (via counterfactual updates) and
    to decide when a brand's evidence is statistically reliable enough for
    the policy to stop exploring it.

    Parameters
    ----------
    n_keys : int
        Number of distinct decision-relevant keys (brands in Watch-Repair).
    eps_start, eps_decay, eps_min : float
        Standard decaying epsilon-greedy schedule used while a key's
        evidence is not yet considered reliable.
    conf_z : float
        z-score threshold (see VCBMStats.z_score) above which a key's
        mean-repair-return sign is treated as statistically reliable.
    min_samples : int
        Minimum repair observations for a key before the confidence lock
        (conf_z check) is even consulted, and before counterfactual Q
        updates / Q-table broadcast are applied. Prevents acting on a
        single noisy sample.
    forced_min_samples : int
        Forces the repair action for a key until it has this many
        observations, regardless of eps or confidence. Fixes sample
        starvation for keys whose true repair-vs-pass margin is small.
    confident_explore_prob, confident_explore_decay : float
        Even once "confident", keep a small (decaying) chance of
        re-sampling, so a mistaken lock-in from an unlucky early estimate
        remains correctable rather than permanent.
    """

    def __init__(self, n_keys, eps_start=0.20, eps_decay=0.995, eps_min=0.0,
                 conf_z=2.0, min_samples=8, forced_min_samples=51,
                 confident_explore_prob=0.08, confident_explore_decay=0.9985):
        self.stats = VCBMStats(n_keys)
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.conf_z = conf_z
        self.min_samples = min_samples
        self.forced_min_samples = forced_min_samples
        self.confident_explore_prob = confident_explore_prob
        self.confident_explore_decay = confident_explore_decay

    def current_eps(self, episode):
        return max(self.eps_min, self.eps_start * (self.eps_decay ** episode))

    def current_confident_explore_prob(self, episode):
        return self.confident_explore_prob * (self.confident_explore_decay ** episode)

    def select_action(self, q_s, key, episode, at_decision_point):
        """
        Choose an action given the current Q-values for this state, the
        decision-relevant key (brand), and the episode index (for eps
        decay). Only meaningful at the actual decision point (t=0);
        callers should force the deterministic "pass" action afterward,
        same as the original TabularActor.act() behaviour.
        """
        if not at_decision_point:
            return 1  # forced pass after t=0, matching TabularActor.act()

        # Forced exploration: guarantee this key gets a reliable minimum
        # sample count before any confidence logic (eps-greedy OR the
        # z-score lock) is even consulted.
        if self.stats.counts[key] < self.forced_min_samples:
            return 0  # force repair

        enough_samples = self.stats.counts[key] >= self.min_samples
        z = self.stats.z_score(key) if enough_samples else 0.0
        eps = self.current_eps(episode)
        cur_confident_explore = self.current_confident_explore_prob(episode)

        valid = ~np.isnan(q_s)
        if not np.any(valid) or len(np.unique(q_s[valid])) <= 1:
            return int(np.random.choice(2))
        elif enough_samples and z > self.conf_z:
            if np.random.random() < cur_confident_explore:
                return int(np.random.choice(2))
            return int(np.nanargmax(q_s))
        elif np.random.random() < eps:
            return int(np.random.choice(2))
        else:
            return int(np.nanargmax(q_s))

    def redistribute_and_update(self, q_table, key, a_taken, episode_return, lr, state_index):
        """
        VCBM's analogue of RUDDER's redistribute_reward() + Q update: no
        sequence model needed, since the whole episode's credit belongs to
        the single decision at t=0. Updates the running per-key statistics,
        applies a direct Q update for the action taken, and a counterfactual
        Q update for the action not taken (0 for "pass", the running mean
        for "repair" once enough samples exist).

        Parameters
        ----------
        q_table : np.ndarray
            The agent's Q-table (mutated in place, same convention as
            TabularActor's update_* methods).
        key : int
            The decision-relevant key (brand) for this episode.
        a_taken : int
            Action actually taken at the decision point (0=repair, 1=pass).
        episode_return : float
            Total return for the episode.
        lr : float
            Learning rate for the Q update.
        state_index : tuple
            Index into q_table (excluding the leading action dimension) for
            the decision-point state, e.g. (0, 0, 0, brand, 0).
        """
        if a_taken == 0:
            self.stats.update(key, episode_return)

        idx_taken = (a_taken,) + state_index
        q_table[idx_taken] += lr * (episode_return - q_table[idx_taken])

        a_other = 1 - a_taken
        if a_other == 1:
            cf_return, do_cf = 0.0, True
        else:
            do_cf = self.stats.counts[key] >= self.min_samples
            cf_return = self.stats.mean[key]

        if do_cf:
            idx_other = (a_other,) + state_index
            if not np.isnan(q_table[idx_other]):
                q_table[idx_other] += lr * (cf_return - q_table[idx_other])

        # Broadcast the running mean across the (cond0, cond1) subspace once
        # enough samples exist, so states that differ only in transport
        # conditions (which don't affect the decision) share the estimate.
        if self.stats.counts[key] == self.min_samples:
            try:
                q_table[0, 0, :, :, key, 0] = self.stats.mean[key]
            except (IndexError, ValueError):
                pass

    def summary(self, optimal_actions=None):
        """
        Returns a list of per-key dicts (mean, count, learned decision, and
        -- if optimal_actions is provided -- whether that decision matches
        the true optimum), mirroring the per-brand printout in the training
        script.
        """
        out = []
        for k in range(self.stats.n_keys):
            mu = self.stats.mean[k]
            n = int(self.stats.counts[k])
            learned = "REPAIR" if mu > 0 else "PASS"
            entry = dict(key=k, mean_repair=mu, n=n, learned=learned)
            if optimal_actions is not None:
                optimal = "PASS" if optimal_actions[k] == 1 else "REPAIR"
                entry["optimal"] = optimal
                entry["match"] = (learned == optimal)
            out.append(entry)
        return out