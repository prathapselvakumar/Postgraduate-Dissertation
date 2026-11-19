"""
classifier_check.py -- stress-test TransitionClassifier on components with known roles.

Each synthetic episode is 50 steps with uniformly random actions (0/1). Components:
  0 flag       controlled: set to 1 by action 0 at t=0 only (like watch-repair's `repaired`)
  1 position   controlled by direction: +1 under action 0, -1 under action 1, every step (mod 10).
               The legacy test only asks "did it change?", so it sees a clock here.
  2 noise_rare noise: +1 with probability 0.05 each step, whatever the action
  3 noise_coin noise: flips with probability 0.5 each step, whatever the action
  4 clock      clock: the timestep
  5 static     static: fixed for the episode

For each mode and seed it reports, per component, the label at the end and whether a
noise/clock/static component was EVER labelled controlled at any check (a false alarm),
and the first check at which each controlled component was found.

Usage:
  python classifier_check.py                       # 10 seeds x 200,000 transitions
  python classifier_check.py --seeds 20 --transitions 1000000
"""

import argparse
import warnings
from collections import Counter

import numpy as np

from vcbm import TransitionClassifier

NAMES = ["flag", "position", "noise_rare", "noise_coin", "clock", "static"]
TRUE_ROLE = ["controlled", "controlled", "noise", "noise", "clock", "static"]
EPISODE_LEN = 50


def run(mode, seed, n_transitions):
    rng = np.random.default_rng(seed)
    clf = TransitionClassifier(n_components=len(NAMES), mode=mode)
    first_found = [None] * len(NAMES)
    ever_false = [False] * len(NAMES)
    t_total = 0
    while t_total < n_transitions:
        state = np.array([0, rng.integers(10), 0, rng.integers(2), 0, rng.integers(4)])
        for t in range(EPISODE_LEN):
            a = int(rng.integers(2))
            nxt = state.copy()
            if t == 0 and a == 0:
                nxt[0] = 1
            nxt[1] = (state[1] + (1 if a == 0 else -1)) % 10
            nxt[2] = state[2] + (rng.random() < 0.05)
            if rng.random() < 0.5:
                nxt[3] = 1 - state[3]
            nxt[4] = t + 1
            n_checks_before = clf.n_checks
            clf.update(state, a, nxt, randomised=True)
            if clf.n_checks != n_checks_before:
                for i, lab in enumerate(clf.labels()):
                    if lab == "controlled":
                        if TRUE_ROLE[i] == "controlled" and first_found[i] is None:
                            first_found[i] = clf.n_checks
                        if TRUE_ROLE[i] != "controlled":
                            ever_false[i] = True
            state = nxt
            t_total += 1
    return clf.labels(), first_found, ever_false, clf.n_checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--transitions", type=int, default=200_000)
    args = p.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)  # zero-controlled warnings are expected early

    for mode in ("legacy", "sequential"):
        final = [Counter() for _ in NAMES]
        found = [[] for _ in NAMES]
        false_alarm_seeds = [0] * len(NAMES)
        for seed in range(1, args.seeds + 1):
            labels, first_found, ever_false, n_checks = run(mode, seed, args.transitions)
            for i in range(len(NAMES)):
                final[i][labels[i]] += 1
                if first_found[i] is not None:
                    found[i].append(first_found[i])
                false_alarm_seeds[i] += ever_false[i]
        print(f"\n=== mode={mode}  seeds={args.seeds}  transitions/seed={args.transitions:,}"
              f"  checks/seed={n_checks} ===")
        print(f"{'component':<11} {'true role':<11} {'final labels':<34} detail")
        for i, name in enumerate(NAMES):
            labs = ", ".join(f"{k}:{v}" for k, v in final[i].most_common())
            if TRUE_ROLE[i] == "controlled":
                det = (f"found in {len(found[i])}/{args.seeds} seeds"
                       + (f", median first check {int(np.median(found[i]))}" if found[i] else ""))
            else:
                det = f"ever falsely 'controlled' in {false_alarm_seeds[i]}/{args.seeds} seeds"
            print(f"{name:<11} {TRUE_ROLE[i]:<11} {labs:<34} {det}")


if __name__ == "__main__":
    main()
