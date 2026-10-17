# A worked example

A complete run on simulated data, from nothing to scored output, in about half a
minute on a laptop CPU. Every number below is from an actual run of these
commands; the data generator is seeded, so you should get the same ones.

Run everything from the repository root.

## 1. Make the data

```bash
python examples/make_example_data.py
```

Three regions of 300 variants, each with ten binary annotation columns. **Only
columns 0 and 1 carry any signal** — the other eight are noise. Causal variants
(three per region) are drawn with probability driven by those two columns, so
there is a real annotation-to-causality map for the prior to find, and a known
answer to check against.

LD is block-structured with correlation decaying inside each block, which is
what makes fine-mapping hard: a causal variant is nearly indistinguishable from
its neighbours. The strongest marginal signal is often *not* the causal variant.

```
region1: causal at [5, 72, 74]; largest |z| at [70, 71, 72, 73, 74]
region2: causal at [30, 151, 210]; largest |z| at [28, 29, 30, 31, 32]
region3: causal at [89, 251, 266]; largest |z| at [264, 265, 266, 267, 268]
```

This writes `examples/data/`: one `.z`, `.ld` and `.annot` per region, a
`manifest.tsv` for the joint model, and `truth.tsv` recording which variants are
really causal. The files are not in git — regenerate them with the command above.

## 2. One locus

```bash
python beatrice_annot.py \
    --z examples/data/region1.z \
    --LD examples/data/region1.ld \
    --annot examples/data/region1.annot \
    --N 50000 --n_caus 3 --max_iter 1500 --lambda_l1 0.2 \
    --target examples/results_single/region1
```

The prior is a LassoNet, so the L1 penalty can remove an annotation from the
prior entirely rather than merely shrinking it. Here it keeps one column and
zeroes the other nine:

```
Top 10 most important annotations:
 annotation_index  importance
                1    0.002588
                0    0.000000
                ...  0.000000
```

Column 1 is one of the two that really matter — but column 0 was dropped, and
nothing distinguishes it from the eight noise columns. **That is the limitation
this method's joint model exists to address**: one region contains three causal
variants, which is not enough evidence to identify a genome-wide enrichment.

## 3. All three loci, one shared prior

```bash
python beatrice_joint.py \
    --manifest examples/data/manifest.tsv \
    --N 50000 --n_caus 3 --max_iter 1500 --lambda_l1 0.2 \
    --prior_head lassonet \
    --save_prior_weights examples/results_joint/prior_head.pt
```

One prior head is trained across all three regions at once, while each region
keeps its own posterior. Every optimiser step sees all three regions' evidence.

## 4. Score it

```bash
python examples/check_results.py examples/results_joint
```

```
  region1
    causal variants      [5, 72, 74]
    their PIPs           5:0.007 (rank 29), 72:0.093 (rank 2), 74:0.530 (rank 1)
    top-ranked variant   74 (PIP 0.530)  <- causal
    credible sets        [[74, 72]]
    causal variants captured by a set  2/3

  region2
    causal variants      [30, 151, 210]
    their PIPs           30:0.960 (rank 1), 151:0.188 (rank 2), 210:0.010 (rank 38)
    top-ranked variant   30 (PIP 0.960)  <- causal
    credible sets        [[30]]

  region3
    causal variants      [89, 251, 266]
    their PIPs           89:0.045 (rank 4), 251:0.169 (rank 2), 266:0.965 (rank 1)
    top-ranked variant   266 (PIP 0.965)  <- causal
    credible sets        [[266]]

  Annotation selection (the shared prior head)
    informative columns  [0, 1] of 0..9
    their ranks          col 0: rank 2 of 10, col 1: rank 1 of 10
    columns driven to exactly zero by the L1 penalty  6/10
    informative columns in the top 2: 2/2 (expected 0.4 by chance)

  Overall
    causal variants ranked first in their region   3/9
    ranked in the top 5                            7/9
    median rank of a causal variant (of 300)       2
```

Two things to take from this.

**The fine-mapping works.** Each region's top-ranked variant is genuinely causal,
and the median causal variant ranks 2nd of 300. Every credible set reported
contains a causal variant, and none is spurious. The causals that are missed —
variant 5 in region 1, 210 in region 2 — are the ones with weak marginal signal,
which is the expected failure mode, not a bug.

**The shared prior recovers the annotations.** Pooling three regions puts both
informative columns at ranks 1 and 2 of 10, against 0.4 expected by chance, and
drops six of the ten entirely. The single-region run in step 2 found one of the
two and could not separate the other from noise.

## On `--lambda_l1`

The example passes `--lambda_l1 0.2`, not the default `0.01`. That is deliberate:
in a sweep of this penalty the shared head selected annotations **no better than
chance at λ ≤ 0.06**, and selection was best in the range 0.12–0.25. The default
is too weak to drive the skip connections to zero, so the prior uses all ten
columns a little and identifies none of them. Try it and compare:

```bash
# Output paths come from the manifest, so this overwrites examples/results_joint.
python beatrice_joint.py --manifest examples/data/manifest.tsv \
    --N 50000 --n_caus 3 --max_iter 1500 --lambda_l1 0.01 --prior_head lassonet
python examples/check_results.py examples/results_joint
```

At `--lambda_l1 0.01` the two informative columns come out at ranks 5 and 7 of
10, with nothing driven to zero. Re-run step 3 to get the good run back.

If you care about which annotations the prior used, tune this penalty. If you
only care about the PIPs, it matters much less.

## What each output file is

| File | Contents |
|---|---|
| `pip.csv` | `variant_index`, `pip`, `variant_names` — the posterior inclusion probability per variant |
| `credible_set.txt` | one credible set per line, as variant indices |
| `conditional_credible_variants_probability.txt` | within-set selection probabilities, aligned with `credible_set.txt` |
| `feature_importance.csv` | `annotation_index`, `importance` — the LassoNet skip-connection magnitude per annotation, so a zero means the prior does not use that annotation |
| `res` | the full fit, pickled |
| `prior_head.pt` | the trained shared prior head, reusable via `--prior_weights` |

## Scaling up

`--max_iter 1500` and three 300-variant regions keep this fast. For real loci
expect a few thousand iterations and correspondingly longer runs; set
`BEATRICE_DEVICE=cuda` to use a GPU.
