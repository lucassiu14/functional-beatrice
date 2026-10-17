# Functional BEATRICE

Fine-mapping that learns the map from functional annotations to the prior
probability of causality **jointly with the causal posterior**, rather than
fixing that map in a separate upstream stage or constraining it to a
pre-chosen parametric form.

Two models:

- **`beatrice_annot.py`** — one locus. The uniform prior of BEATRICE is
  replaced by a neural prior that maps each variant's annotation vector to its
  prior causal probability, optimised in the same variational objective as the
  inference network. The prior is a LassoNet: a linear skip connection in
  parallel with a multilayer branch, with an L1 penalty and a hierarchy
  constraint, so annotations irrelevant to the trait are removed from the prior
  rather than merely down-weighted.
- **`beatrice_joint.py`** — many loci. One prior head is shared across all
  regions and trained jointly, so every optimiser step aggregates evidence from
  every region, while each region keeps its own posterior. This addresses the
  central weakness of learning an enrichment at a single locus: one region
  supplies too few causal variants to estimate a genome-wide quantity.

---

## Provenance and licence

This is a **fork of BEATRICE** (Ghosal et al. 2024,
<https://github.com/sayangsep/Beatrice-Finemapping>). The variational
formulation, the inference network, the likelihood evaluation, the
configuration scoring and the credible-set construction are theirs. The
annotation-conditioned prior, the LassoNet parameterisation and the
cross-region joint trainer are the additions here.

> **Before this repository is made public, the upstream licence must be
> established and included.** No `LICENSE` file was carried over from the fork
> and none is present in this history. Check the terms upstream, add the
> appropriate licence file and attribution notice, and make sure the additions
> here are licensed compatibly.

If you use this, cite the original BEATRICE paper as well:

> Ghosal S, et al. (2024). BEATRICE: Bayesian fine-mapping from summary data
> using deep variational inference. *Bioinformatics*.

---

## Requirements

See `requirements.txt`: PyTorch, NumPy, pandas, `absl-py`, and — because the
single-locus trainer imports them unconditionally for its loss plots —
matplotlib, seaborn, imageio and tqdm. A GPU is optional; the models run on
CPU, and `scripts/device.py` handles placement.

```bash
pip install -r requirements.txt
```

The `--z` / `--LD` / `--annot` flags default to paths under `example_data/`,
which is not in this repository, so pass your own files rather than relying on
the defaults.

## Running one locus

```bash
python beatrice_annot.py \
    --z region.z --LD region.ld --annot region.annot \
    --N 5000 --target results/
```

`--z`, `--LD` and `--annot` are the marginal z-scores, the LD matrix and the
per-variant annotation matrix. Results are written to `--target`: posterior
inclusion probabilities, and credible sets when `--get_cred` is set.

The prior is controlled by `--lambda_l1` (the L1 penalty on the skip
connections, which decides how many annotations the prior may use),
`--hierarchy_M` (how much non-linearity an annotation is allowed relative to
its linear effect) and `--prior_neural_network` (the hidden layer widths).
`--return_weights` saves the trained prior so `--prior_weights` can reuse it.

## Running many loci jointly

```bash
python beatrice_joint.py \
    --manifest scenario.tsv --N 5000 \
    --prior_head lassonet --save_prior_weights prior.pt
```

`--manifest` is a tab-separated file with one region per line, giving that
region's z-scores, LD matrix, annotation matrix, output directory and sample
size. Every region is fitted under one shared prior head, and per-region
posterior inclusion probabilities and credible sets are written to each
region's own target directory.

`--prior_head` chooses the shared head: `lassonet`, or `linear` for a prior
linear in the annotations.

---

## Relationship to fine-mapping-benchmark

The evaluation of this method lives in a separate repository,
<https://github.com/lucassiu14/fine-mapping-benchmark>, which runs it against
other fine-mapping methods on common simulated data. That package calls this
code as an external tool: point its `beatrice_dir` argument at a clone of this
repository.

Nothing here depends on the benchmark.
