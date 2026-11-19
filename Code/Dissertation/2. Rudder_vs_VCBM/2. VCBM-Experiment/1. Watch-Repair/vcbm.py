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

Classifier modes (TransitionClassifier(mode=...)):
   - "legacy" (default): the behaviour above, unchanged, so published results
     reproduce exactly. A fixed bar (z_threshold) on a pooled two-proportion
     z-test of per-action change rates, recomputed from scratch at every check.
   - "sequential": fixes three weaknesses of the legacy test.
       1. The bar grows with the sample size, max(z_min, sqrt(2 ln n)), so
          re-checking cumulative counts no longer makes a false "controlled"
          label certain in a long run, while a real effect (z ~ sqrt(n)) still
          crosses it.
       2. Failures are visible and recoverable: a "controlled" label is sticky
          (demoted only after `demote_checks` consecutive checks with
          z < demote_z); a watchdog provisionally promotes the strongest
          candidate after `watchdog_checks` checks with nothing controlled;
          value-statistics keys carry a schema version so a relabel cannot
          merge returns from different projections.
       3. The test sees more kinds of control: besides the change-rate test it
          runs a G-test of action vs next value, stratified by the component's
          previous value (catches control of direction or of the value set).
          Only transitions whose action was drawn uniformly at random are
          counted, so the greedy policy cannot make state-dependent noise look
          action-dependent.
   Both modes warn when no component is labelled controlled (no decision
   points), and record per-check telemetry in `history` and `label_changes`.
   Neither mode draws random numbers, so the RNG stream is untouched.
"""

import math
import warnings

import numpy as np

CLASSIFIER_MODES = ("legacy", "sequential")


def _stratified_g_test(tables, min_per_action):
    """G-test of independence between action and next value, summed over strata.

    tables: {prev_value: {(action, next_value): count}} for one component.
    A stratum is skipped unless both actions were observed at least
    `min_per_action` times in it and it has at least two distinct next values.

    Returns (G, df, z): z is the Wilson-Hilferty normal equivalent of the
    chi-square(df) upper tail, so it is on the same scale as the change-rate z.
    Returns (0.0, 0, 0.0) when no stratum qualifies.
    """
    g_total, df_total = 0.0, 0
    for cells in tables.values():
        actions = sorted({a for a, _ in cells})
        nexts = sorted({v for _, v in cells})
        if len(actions) < 2 or len(nexts) < 2:
            continue
        obs = np.array([[cells.get((a, v), 0) for v in nexts] for a in actions],
                       dtype=float)
        row = obs.sum(axis=1)
        if row.min() < min_per_action:
            continue
        exp = np.outer(row, obs.sum(axis=0)) / obs.sum()
        mask = obs > 0
        g_total += 2.0 * float(np.sum(obs[mask] * np.log(obs[mask] / exp[mask])))
        df_total += (len(actions) - 1) * (len(nexts) - 1)
    if df_total == 0:
        return 0.0, 0, 0.0
    k = 2.0 / (9.0 * df_total)
    z = ((g_total / df_total) ** (1.0 / 3.0) - (1.0 - k)) / math.sqrt(k)
    return g_total, df_total, float(z)


class TransitionClassifier:
    """Classifies state components and records bookmark change-contexts."""

    CLOCK = "clock"
    STATIC = "static"
    CONTROLLED = "controlled"
    NOISE = "noise"

    def __init__(self, n_components, n_actions=2, warmup_transitions=2000,
                 z_threshold=4.0, clock_rate=0.9, mode="legacy",
                 check_interval=500, z_min=3.0, demote_z=2.0, demote_checks=3,
                 watchdog_checks=3, min_stratum_per_action=5):
        if n_actions != 2:
            # Both tests and VCBM.select_action compare exactly two actions; fail
            # loudly instead of silently ignoring actions >= 2.
            raise ValueError(f"TransitionClassifier supports exactly 2 actions, got {n_actions}")
        if mode not in CLASSIFIER_MODES:
            raise ValueError(f"mode must be one of {CLASSIFIER_MODES}, got {mode!r}")
        self.n_components = n_components
        self.n_actions = n_actions
        self.warmup_transitions = warmup_transitions
        self.z_threshold = z_threshold
        self.clock_rate = clock_rate
        self.mode = mode
        self.check_interval = check_interval
        self.z_min = z_min
        self.demote_z = demote_z
        self.demote_checks = demote_checks
        self.watchdog_checks = watchdog_checks
        self.min_stratum_per_action = min_stratum_per_action

        # Sequential-mode evidence, from randomised-action transitions only.
        self.n_rand_by_action = np.zeros(n_actions)
        self.rand_changes = np.zeros((n_components, n_actions))
        self.value_tables = [dict() for _ in range(n_components)]
        self._weak_streak = np.zeros(n_components, dtype=int)
        self._provisional = [False] * n_components

        # Telemetry (both modes; never affects behaviour in legacy mode).
        self.n_checks = 0
        self.history = []        # one record per (check, component)
        self.label_changes = []  # one record per check whose labels changed
        self.schema_version = 0  # bumped on every label change
        self._zero_controlled_streak = 0

        self.n_by_action = np.zeros(n_actions)
        self.changes = np.zeros((n_components, n_actions))
        # Raw contexts (full prev-state tuples + which comp changed);
        # projected lazily once classification is known.
        self.change_contexts = [set() for _ in range(n_components)]
        self.total = 0
        self._labels = None
        self._opportunity_set = None  # projected contexts, rebuilt on classify

    def update(self, prev_state, action, next_state, randomised=True):
        """Record one transition.

        randomised: True if `action` was drawn uniformly at random, independent
        of the state. Sequential mode only tests on such transitions; legacy
        mode ignores the flag and counts every transition, as before.
        """
        self.total += 1
        self.n_by_action[action] += 1
        for i in range(self.n_components):
            if prev_state[i] != next_state[i]:
                self.changes[i, action] += 1
                self.change_contexts[i].add(tuple(int(v) for v in prev_state))
        if self.mode == "sequential" and randomised:
            a = int(action)
            self.n_rand_by_action[a] += 1
            for i in range(self.n_components):
                prev_v, next_v = int(prev_state[i]), int(next_state[i])
                if prev_v != next_v:
                    self.rand_changes[i, a] += 1
                cells = self.value_tables[i].setdefault(prev_v, {})
                cells[(a, next_v)] = cells.get((a, next_v), 0) + 1
        # Refresh classification periodically
        if self.total >= self.warmup_transitions and self.total % self.check_interval == 0:
            self._classify()

    @staticmethod
    def _change_rate_z(c0, c1, n0, n1):
        """Two-proportion z-test on change rates between the two actions."""
        r0 = c0 / n0 if n0 > 0 else 0.0
        r1 = c1 / n1 if n1 > 0 else 0.0
        if n0 + n1 == 0:
            return r0, r1, 0.0
        p = (c0 + c1) / (n0 + n1)
        se = np.sqrt(p * (1 - p) * (1 / max(n0, 1) + 1 / max(n1, 1)))
        z = abs(r0 - r1) / se if se > 0 else 0.0
        return r0, r1, float(z)

    def _classify(self):
        self.n_checks += 1
        old = self._labels
        if self.mode == "legacy":
            labels, records, bar = self._classify_legacy()
        else:
            labels, records, bar = self._classify_sequential(old)
        for rec in records:
            rec.update(check=self.n_checks, transitions=self.total)
            self.history.append(rec)
        if old is not None and list(old) != labels:
            self.schema_version += 1
            self.label_changes.append(dict(check=self.n_checks, transitions=self.total,
                                           old=list(old), new=list(labels)))
        self._labels = labels
        self._rebuild_opportunities()
        self._warn_if_no_controlled(records, bar)

    def _classify_legacy(self):
        """Original rule: fixed bar, all transitions, no memory between checks."""
        labels, records = [], []
        n0, n1 = self.n_by_action[0], self.n_by_action[1]
        for i in range(self.n_components):
            c0, c1 = self.changes[i, 0], self.changes[i, 1]
            r0, r1, z = self._change_rate_z(c0, c1, n0, n1)
            if min(r0, r1) > self.clock_rate:
                lab = self.CLOCK
            elif c0 + c1 == 0:
                lab = self.STATIC
            else:
                lab = self.CONTROLLED if z > self.z_threshold else self.NOISE
            labels.append(lab)
            records.append(dict(component=i, n0=int(n0), n1=int(n1), c0=int(c0), c1=int(c1),
                                r0=r0, r1=r1, z_change=z, z_value=None, z=z,
                                bar=self.z_threshold, label=lab, provisional=False))
        return labels, records, self.z_threshold

    def _classify_sequential(self, old):
        """Growing bar, randomised transitions only, value test, sticky labels."""
        n0, n1 = self.n_rand_by_action[0], self.n_rand_by_action[1]
        bar = max(self.z_min, math.sqrt(2.0 * math.log(max(n0 + n1, 2.0))))
        labels, records = [], []
        for i in range(self.n_components):
            c0, c1 = self.rand_changes[i, 0], self.rand_changes[i, 1]
            r0, r1, z_change = self._change_rate_z(c0, c1, n0, n1)
            _, _, z_value = _stratified_g_test(self.value_tables[i], self.min_stratum_per_action)
            z = max(z_change, z_value)
            was_controlled = old is not None and old[i] == self.CONTROLLED
            if self.changes[i].sum() == 0:
                lab = self.STATIC  # never changed in any transition
            elif z >= bar:
                lab = self.CONTROLLED
                self._weak_streak[i] = 0
                self._provisional[i] = False
            elif was_controlled:
                # Hysteresis: a miss is the costly error, so only demote after
                # repeated clearly weak evidence.
                self._weak_streak[i] = self._weak_streak[i] + 1 if z < self.demote_z else 0
                lab = self.CONTROLLED
                if self._weak_streak[i] >= self.demote_checks:
                    lab = self.CLOCK if min(r0, r1) > self.clock_rate else self.NOISE
            elif min(r0, r1) > self.clock_rate:
                lab = self.CLOCK  # changes almost every step, not action-dependent
            else:
                lab = self.NOISE
            if lab != self.CONTROLLED:
                self._weak_streak[i] = 0
                self._provisional[i] = False
            labels.append(lab)
            records.append(dict(component=i, n0=int(n0), n1=int(n1), c0=int(c0), c1=int(c1),
                                r0=r0, r1=r1, z_change=z_change, z_value=z_value, z=z,
                                bar=bar, label=lab, provisional=self._provisional[i]))

        # Watchdog: "no decision points" must not be a state VCBM stays in.
        if self.CONTROLLED in labels:
            self._zero_controlled_streak = 0
        else:
            self._zero_controlled_streak += 1
            if self._zero_controlled_streak >= self.watchdog_checks:
                cands = [r for r in records
                         if r["label"] == self.NOISE and r["z"] >= self.demote_z]
                if cands:
                    best = max(cands, key=lambda r: r["z"])
                    i = best["component"]
                    labels[i] = self.CONTROLLED
                    self._provisional[i] = True
                    best.update(label=self.CONTROLLED, provisional=True)
                    warnings.warn(
                        f"VCBM classifier check {self.n_checks}: nothing reached the bar "
                        f"({bar:.2f}) for {self._zero_controlled_streak} checks; component {i} "
                        f"(z={best['z']:.2f}) is provisionally labelled controlled.",
                        RuntimeWarning)
        return labels, records, bar

    def _warn_if_no_controlled(self, records, bar):
        if self.CONTROLLED in self._labels:
            if self.mode == "legacy":
                self._zero_controlled_streak = 0
            return
        if self.mode == "legacy":
            self._zero_controlled_streak += 1
        streak = self._zero_controlled_streak
        if streak == 1 or streak % 10 == 0:
            closest = sorted((r for r in records if r["label"] == self.NOISE),
                             key=lambda r: r["z"], reverse=True)[:2]
            near = ", ".join(f"component {r['component']} z={r['z']:.2f}" for r in closest)
            warnings.warn(
                f"VCBM classifier check {self.n_checks} ({self.total} transitions): no component "
                f"is labelled controlled, so there are no decision points and every action is "
                f"uniformly random. Bar={bar:.2f}; closest: {near or 'none'}.",
                RuntimeWarning)

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
        key = tuple(int(state[j]) for j in proj_idx)
        if self.mode == "sequential":
            # Version the key so statistics gathered under one projection can
            # never be merged with those of another after a relabel.
            return (self.schema_version,) + key
        return key


class VCBM:
    """Bookmark-discovering VCBM controller."""

    def __init__(self, n_components, n_actions=2,
                 warmup_transitions=2000, classifier_z=4.0,
                 forced_min_samples=60, conf_z=2.0, min_samples=8,
                 eps_start=0.20, eps_decay=0.995, eps_min=0.005,
                 classifier_mode="legacy"):
        self.classifier = TransitionClassifier(
            n_components, n_actions,
            warmup_transitions=warmup_transitions, z_threshold=classifier_z,
            mode=classifier_mode)
        # Whether the last action from select_action was drawn uniformly at
        # random (independent of the state); passed on to the classifier.
        self._last_action_random = True
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
        self.classifier.update(prev_state, action, next_state,
                               randomised=self._last_action_random)

    def select_action(self, state, episode):
        """Returns (action, was_opportunity)."""
        # Every uniform-random branch below sets this back to True; no extra
        # random numbers are drawn, so the RNG stream matches the original.
        self._last_action_random = False
        if not self.classifier.ready() or not self.classifier.is_opportunity(state):
            # Unknown/non-decision timestep: uniform random. Pre-discovery this
            # supplies the data classification needs; post-discovery these are
            # the timesteps the environment ignores.
            self._last_action_random = True
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
                self._last_action_random = True
                return int(np.random.choice(self.n_actions)), True
            return int(np.argmax(m)), True
        if np.random.random() < eps:
            self._last_action_random = True
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
        clf = self.classifier
        orphaned = 0
        if clf.mode == "sequential":
            # Entries stored under an older label schema are no longer read.
            orphaned = sum(1 for (k, _) in self.counts if k and k[0] != clf.schema_version)
        return {"component_labels": labels, "keys": rows,
                "classifier_mode": clf.mode,
                "classifier_checks": clf.n_checks,
                "label_changes": list(clf.label_changes),
                "orphaned_stat_entries": orphaned}