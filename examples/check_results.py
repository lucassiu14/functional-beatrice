#!/usr/bin/env python3
"""Score Functional BEATRICE's output on the example against the known truth.

    python examples/check_results.py examples/results_joint

Reads each region's pip.csv, credible_set.txt and feature_importance.csv and
reports, per region, whether the causal variants were found and whether the
prior head selected the annotations that actually drove causality.

Only numpy and pandas are needed.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
TRUTH = os.path.join(HERE, "data", "truth.tsv")
# Must match make_example_data.py.
N_INFORMATIVE = 2
N_ANNOT = 10


def score_region(region, results_dir, truth):
    pip = pd.read_csv(os.path.join(results_dir, "pip.csv"))
    causal = sorted(truth.loc[truth.is_causal == 1, "variant_index"])

    ranked = pip.sort_values("pip", ascending=False)
    rank_of = {int(v): i + 1 for i, v in enumerate(ranked.variant_index)}
    pip_of = dict(zip(pip.variant_index, pip.pip))

    cs_path = os.path.join(results_dir, "credible_set.txt")
    sets = []
    if os.path.exists(cs_path):
        for line in open(cs_path):
            idx = [int(t) for t in line.split()]
            if idx:
                sets.append(idx)

    covered = sum(1 for s in sets if any(c in s for c in causal))
    in_a_set = [c for c in causal if any(c in s for s in sets)]

    print(f"  {region}")
    print(f"    causal variants      {causal}")
    print("    their PIPs           " +
          ", ".join(f"{c}:{pip_of[c]:.3f} (rank {rank_of[c]})" for c in causal))
    print(f"    top-ranked variant   {int(ranked.variant_index.iloc[0])} "
          f"(PIP {ranked.pip.iloc[0]:.3f})"
          f"{'  <- causal' if int(ranked.variant_index.iloc[0]) in causal else ''}")
    if sets:
        print(f"    credible sets        {sets}")
        print(f"    sets containing a causal variant   {covered}/{len(sets)}")
        print(f"    causal variants captured by a set  {len(in_a_set)}/{len(causal)}")
    else:
        print("    credible sets        none reported")
    return rank_of, causal


def score_annotations(results_dir):
    path = os.path.join(results_dir, "feature_importance.csv")
    if not os.path.exists(path):
        return
    fi = pd.read_csv(path).sort_values("importance", ascending=False)
    ranks = {int(a): i + 1 for i, a in enumerate(fi.annotation_index)}
    informative = list(range(N_INFORMATIVE))
    got = [ranks[a] for a in informative]
    n_zero = int((fi.importance == 0).sum())

    print("\n  Annotation selection (the shared prior head)")
    print(f"    informative columns  {informative} of 0..{N_ANNOT - 1}")
    print("    their ranks          " +
          ", ".join(f"col {a}: rank {r} of {N_ANNOT}" for a, r in zip(informative, got)))
    print(f"    columns driven to exactly zero by the L1 penalty  {n_zero}/{N_ANNOT}")
    hits = sum(1 for r in got if r <= N_INFORMATIVE)
    # Under a random ranking, the chance of landing both informative columns in
    # the top N_INFORMATIVE slots is N_INFORMATIVE! / (N_ANNOT * (N_ANNOT - 1)).
    chance = N_INFORMATIVE / N_ANNOT
    print(f"    informative columns in the top {N_INFORMATIVE}: {hits}/{N_INFORMATIVE} "
          f"(expected {chance * N_INFORMATIVE:.1f} by chance)")


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_joint")
    truth = pd.read_csv(TRUTH, sep="\t")

    print(f"Scoring {results} against {os.path.relpath(TRUTH, HERE)}\n")
    regions = sorted(r for r in truth.region.unique()
                     if os.path.isdir(os.path.join(results, r)))
    if not regions:
        sys.exit(f"no region subdirectories found under {results}")

    all_ranks = []
    for region in regions:
        rank_of, causal = score_region(
            region, os.path.join(results, region), truth[truth.region == region])
        all_ranks += [rank_of[c] for c in causal]
        print()

    # The prior head is shared, so any region's copy is the same file.
    score_annotations(os.path.join(results, regions[0]))

    print("\n  Overall")
    print(f"    causal variants ranked first in their region   "
          f"{sum(1 for r in all_ranks if r == 1)}/{len(all_ranks)}")
    print(f"    ranked in the top 5                            "
          f"{sum(1 for r in all_ranks if r <= 5)}/{len(all_ranks)}")
    print(f"    median rank of a causal variant (of 300)       "
          f"{int(np.median(all_ranks))}")


if __name__ == "__main__":
    main()
