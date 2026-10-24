"""
analyze_results.py -- summary statistics for run_fair_comparison.py CSVs.

Reports, per (method, schedule, forced_min_samples) group: converged/total, mean and
sample SD (ddof=1), median and IQR of episodes-to-converge. Capped (non-converged) seeds
count at the cap value, so the statistics are conservative for methods that fail to converge.

Pairwise comparisons against a reference group use a two-sided Mann-Whitney U test
(normal approximation with tie correction, no continuity correction, matching the paper)
and rank-biserial r = 1 - 2U / (n1 * n2). No scipy dependency.

Usage:
  python analyze_results.py results/*.csv --reference VCBM:vcbm:60
"""

import argparse
import csv
import math
from collections import defaultdict

import numpy as np


def mann_whitney(x, y):
    """Two-sided Mann-Whitney U (normal approximation, tie-corrected). Returns U, z, p, r."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(len(combined))
    sorted_vals = combined[order]
    i = 0
    tie_term = 0.0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    n = n1 + n2
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u - n1 * n2 / 2.0) / sigma if sigma > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    r = 1.0 - 2.0 * u / (n1 * n2)
    return u, z, p, r


def load(paths):
    groups = defaultdict(list)
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = f"{row['method']}:{row['schedule']}:{row['forced_min_samples'] or '-'}"
                groups[key].append((int(row["seed"]), int(row["episodes"]), int(row["converged"]),
                                    row["vcbm_labels"]))
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+")
    p.add_argument("--reference", default=None,
                   help="Group key to compare all others against, e.g. VCBM:vcbm:60")
    args = p.parse_args()
    groups = load(args.csvs)

    print(f"{'group':28} {'conv':>6} {'mean':>9} {'sd':>9} {'median':>8} {'IQR':>15} {'min':>6} {'max':>6}")
    for key in sorted(groups):
        eps = np.array([e for _, e, _, _ in groups[key]], float)
        conv = sum(c for _, _, c, _ in groups[key])
        q1, q3 = np.percentile(eps, [25, 75])
        sd = eps.std(ddof=1) if len(eps) > 1 else float("nan")
        print(f"{key:28} {conv:>3}/{len(eps):<2} {eps.mean():9.1f} {sd:9.1f} {np.median(eps):8.1f} "
              f"{q1:7.0f}-{q3:<7.0f} {eps.min():6.0f} {eps.max():6.0f}")
        labels = {lab for _, _, _, lab in groups[key] if lab}
        if labels:
            print(f"{'':28} VCBM labels: {sorted(labels)}")

    if args.reference and args.reference in groups:
        ref = [e for _, e, _, _ in groups[args.reference]]
        print(f"\nMann-Whitney vs {args.reference} (two-sided, tie-corrected normal approximation)")
        for key in sorted(groups):
            if key == args.reference:
                continue
            other = [e for _, e, _, _ in groups[key]]
            u, z, pval, r = mann_whitney(ref, other)
            print(f"  {key:28} U={u:7.1f}  z={z:6.2f}  p={pval:.2e}  rank-biserial r={r:.2f}")


if __name__ == "__main__":
    main()
