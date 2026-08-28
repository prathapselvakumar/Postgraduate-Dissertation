"""
VCBM -- Visual Causal-chain BookMarking with LEARNED bookmark detection.

VCBM makes no assumption that credit belongs to a fixed, known decision
point (e.g. t=0): it DISCOVERS candidate causal decision points
("bookmarks") from data, then estimates per-decision values only at those
points.

Pipeline:
1. TransitionClassifier watches (state, action, next_state) transitions and
   classifies each state component as:
     - "clock":      changes every step regardless of action (e.g. time)
     - "static":     never changes within an episode (e.g. brand)
     - "controlled": change probability depends significantly on the action
                     (e.g. the `repaired` flag -- only flips under repair)
     - "noise":      changes randomly, independent of action (e.g. transport
                     condition counters)
   Classification uses a two-proportion z-test on per-action change rates,
   so it is data-driven, not hardcoded.

2. Bookmark opportunities: whenever a controlled component changed, the
   classifier stores the context it changed FROM. A later timestep is an
   "opportunity" (candidate decision point) if its projection onto the
   (controlled + clock) components matches a stored change context. This is
   how t=0 (and, in multi-decision environments, later decision points) are
   discovered rather than assumed.

3. Value estimation: at each opportunity, VCBM keeps Welford running
   mean/variance of the episode return per (context key, action), where the
   context key is the state projected onto (static + controlled + clock)
   components -- noise components are excluded, which is the sample-sharing
   step that a plain tabular Q-table cannot do.

4. Action selection at opportunities: forced balanced sampling until each
   (key, action) has enough observations, then a two-sample z-test gated
   greedy policy with a permanent exploration floor (a lifelong-exploration
   guarantee, avoiding permanent lock-in onto a wrong action).
   At non-opportunity timesteps the action is chosen uniformly at random:
   before classification converges this supplies the exploration data that
   discovery needs, and after convergence these timesteps are exactly the
   ones the environment ignores.
"""

import numpy as np


class TransitionClassifier:
    """Classifies state components and records bookmark change-contexts."""

    CLOCK = "clock"
    STATIC = "static"
    CONTROLLED = "controlled"
    NOISE = "noise"

    def __init__(self, n_components, n_actions=2, warmup_transitions=2000,
                 z_threshold=4.0, clock_rate=0.9):
        self.n_components = n_components
        self.n_actions = n_actions
        self.warmup_transitions = warmup_transitions
        self.z_threshold = z_threshold
        self.clock_rate = clock_rate

        self.n_by_action = np.zeros(n_actions)
        self.changes = np.zeros((n_components, n_actions))
        # Raw contexts (full prev-state tuples + which comp changed);
        # projected lazily once classification is known.
        self.change_contexts = [set() for _ in range(n_components)]
        self.total = 0
        self._labels = None
        self._opportunity_set = None  # projected contexts, rebuilt on classify

    def update(self, prev_state, action, next_state):
        self.total += 1
        self.n_by_action[action] += 1
        for i in range(self.n_components):
            if prev_state[i] != next_state[i]:
                self.changes[i, action] += 1
                self.change_contexts[i].add(tuple(int(v) for v in prev_state))
        # Refresh classification periodically
        if self.total >= self.warmup_transitions and self.total % 500 == 0:
            self._classify()

    def _classify(self):
        labels = []
        n0, n1 = self.n_by_action[0], self.n_by_action[1]
        for i in range(self.n_components):
            c0, c1 = self.changes[i, 0], self.changes[i, 1]
            r0 = c0 / n0 if n0 > 0 else 0.0
            r1 = c1 / n1 if n1 > 0 else 0.0
            if min(r0, r1) > self.clock_rate:
                labels.append(self.CLOCK)
                continue
            if c0 + c1 == 0:
                labels.append(self.STATIC)
                continue
            # Two-proportion z-test on change rates between actions
            p = (c0 + c1) / (n0 + n1)
            se = np.sqrt(p * (1 - p) * (1 / max(n0, 1) + 1 / max(n1, 1)))
            z = abs(r0 - r1) / se if se > 0 else 0.0
            labels.append(self.CONTROLLED if z > self.z_threshold else self.NOISE)
        self._labels = labels
        self._rebuild_opportunities()

    def _rebuild_opportunities(self):
        """Project stored change contexts onto (controlled + clock) components."""
        proj_idx = self._projection_indices(include_static=False)
        opp = set()
        for i in range(self.n_components):
            if self._labels[i] != self.CONTROLLED:
                continue
            for ctx in self.change_contexts[i]:
                opp.add(tuple(ctx[j] for j in proj_idx))
        self._opportunity_set = opp

    def _projection_indices(self, include_static):
        idx = []
        for i in range((self.n_components)):
            lab = self._labels[i]
            if lab == self.CONTROLLED or lab == self.CLOCK:
                idx.append(i)
            elif include_static and lab == self.STATIC:
                idx.append(i)
        return idx

    def ready(self):
        return self._labels is not None and self._opportunity_set is not None

    def labels(self):
        return list(self._labels) if self._labels else None

    def is_opportunity(self, state):
        """Is this state a discovered candidate decision point?"""
        if not self.ready():
            return False
        proj_idx = self._projection_indices(include_static=False)
        key = tuple(int(state[j]) for j in proj_idx)
        return key in self._opportunity_set

    def stats_key(self, state):
        """Context key for value stats: static + controlled + clock components
        (noise components excluded -- this is the sample-sharing step)."""
        proj_idx = self._projection_indices(include_static=True)
        return tuple(int(state[j]) for j in proj_idx)


class VCBM:
    """Bookmark-discovering VCBM controller."""

    def __init__(self, n_components, n_actions=2,
                 warmup_transitions=2000, classifier_z=4.0,
                 forced_min_samples=60, conf_z=2.0, min_samples=8,
                 eps_start=0.20, eps_decay=0.995, eps_min=0.005):
        self.classifier = TransitionClassifier(
            n_components, n_actions,
            warmup_transitions=warmup_transitions, z_threshold=classifier_z)
        self.n_actions = n_actions
        self.forced_min_samples = forced_min_samples
        self.conf_z = conf_z
        self.min_samples = min_samples
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        # Welford stats per (key, action)
        self.counts = {}
        self.means = {}
        self.m2 = {}

    # ---------------- stats helpers ----------------
    def _get(self, key, a):
        return (self.counts.get((key, a), 0.0),
                self.means.get((key, a), 0.0),
                self.m2.get((key, a), 0.0))

    def _update_stat(self, key, a, value):
        n, mean, m2 = self._get(key, a)
        n += 1
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
        self.counts[(key, a)] = n
        self.means[(key, a)] = mean
        self.m2[(key, a)] = m2

    def _se(self, key, a):
        n, _, m2 = self._get(key, a)
        if n < 2:
            return float("inf")
        return float(np.sqrt((m2 / (n - 1)) / n))

    def current_eps(self, episode):
        return max(self.eps_min, self.eps_start * (self.eps_decay ** episode))

    # ---------------- interface ----------------
    def observe_transition(self, prev_state, action, next_state):
        self.classifier.update(prev_state, action, next_state)

    def select_action(self, state, episode):
        """Returns (action, was_opportunity)."""
        if not self.classifier.ready() or not self.classifier.is_opportunity(state):
            # Unknown/non-decision timestep: uniform random. Pre-discovery this
            # supplies the data classification needs; post-discovery these are
            # the timesteps the environment ignores.
            return int(np.random.choice(self.n_actions)), False

        key = self.classifier.stats_key(state)

        # Forced balanced sampling: guarantee both actions get a reliable
        # minimum sample size before any confidence logic runs.
        ns = [self._get(key, a)[0] for a in range(self.n_actions)]
        if min(ns) < self.forced_min_samples:
            return int(np.argmin(ns)), True

        # Two-sample z-test between action means
        m = [self._get(key, a)[1] for a in range(self.n_actions)]
        se = [self._se(key, a) for a in range(self.n_actions)]
        denom = float(np.sqrt(se[0] ** 2 + se[1] ** 2))
        z = abs(m[0] - m[1]) / denom if denom > 0 else float("inf")

        eps = self.current_eps(episode)
        enough = min(ns) >= self.min_samples
        if enough and z > self.conf_z:
            # Confident: greedy, but the eps floor below still applies on the
            # eps branch -- lifelong exploration is guaranteed via eps_min.
            if np.random.random() < self.eps_min:
                return int(np.random.choice(self.n_actions)), True
            return int(np.argmax(m)), True
        if np.random.random() < eps:
            return int(np.random.choice(self.n_actions)), True
        return int(np.argmax(m)), True

    def update_episode(self, opportunity_records, episode_return):
        """opportunity_records: list of (state, action) at opportunity steps."""
        for state, action in opportunity_records:
            key = self.classifier.stats_key(state)
            self._update_stat(key, int(action), float(episode_return))

    def summary(self):
        labels = self.classifier.labels()
        keys = sorted(set(k for (k, _) in self.counts.keys()))
        rows = []
        for key in keys:
            entry = {"key": key}
            for a in range(self.n_actions):
                n, mean, _ = self._get(key, a)
                entry[f"n_a{a}"] = int(n)
                entry[f"mean_a{a}"] = mean
            entry["greedy_action"] = int(np.argmax(
                [self._get(key, a)[1] for a in range(self.n_actions)]))
            rows.append(entry)
        return {"component_labels": labels, "keys": rows}