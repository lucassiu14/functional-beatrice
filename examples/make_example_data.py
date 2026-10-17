#!/usr/bin/env python3
"""Generate a small, self-contained example for Functional BEATRICE.

Three regions of 300 variants each. Ten annotation columns per variant, of
which only the first two carry any signal - the other eight are noise. Causal
variants are drawn with probability driven by those two columns, so a method
that learns the annotation-to-prior map should (a) find the causal variants
more easily than one with a uniform prior, and (b) say which annotations it
used.

The same two columns are informative in every region. That is the point of the
joint model: no single region has enough causal variants to identify a
genome-wide enrichment, but three regions sharing one prior head do better.

Only numpy is needed - no torch - so the data can be made before the
environment is fully set up.

    python examples/make_example_data.py

Writes into examples/data/:
    region{1,2,3}.z        variant name, z-score            (space separated)
    region{1,2,3}.ld       300 x 300 LD matrix              (space separated)
    region{1,2,3}.annot    variant name, 10 annotations     (tab separated)
    manifest.tsv           z, LD, annot, target, N          (for beatrice_joint)
    truth.tsv              which variants are really causal
"""
import os

import numpy as np

# --- the experiment ----------------------------------------------------------
N_REGIONS       = 3
P               = 300      # variants per region
N_SAMPLES       = 50_000   # GWAS sample size the z-scores imply
N_ANNOT         = 10
N_INFORMATIVE   = 2        # leading annotation columns that carry signal
LOG_ENRICHMENT  = 1.6      # effect of an informative annotation on log-odds causal
N_CAUSAL        = 3        # causal variants per region
HERITABILITY    = 0.01     # variance explained per region
LD_BLOCK        = 25       # variants per LD block
LD_DECAY        = 0.92     # within-block AR(1) correlation
SEED            = 20_251_017

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "data")


def block_ld(p, block, decay):
    """Block-diagonal AR(1) correlation, nudged to be positive definite.

    Real LD is roughly block structured with correlation decaying with
    distance, which is what makes fine-mapping hard: a causal variant is
    nearly indistinguishable from its neighbours.
    """
    ld = np.eye(p)
    for start in range(0, p, block):
        stop = min(start + block, p)
        idx = np.arange(start, stop)
        d = np.abs(idx[:, None] - idx[None, :])
        ld[start:stop, start:stop] = decay ** d
    ld = (ld + ld.T) / 2
    # shrink towards the identity until the smallest eigenvalue is comfortably > 0
    for lam in (0.0, 0.01, 0.05, 0.1, 0.2):
        cand = (1 - lam) * ld + lam * np.eye(p)
        if np.linalg.eigvalsh(cand).min() > 1e-4:
            return cand
    raise RuntimeError("could not make the LD matrix positive definite")


def simulate_region(rng):
    """One region: annotations, causal variants chosen from them, and z-scores."""
    ld = block_ld(P, LD_BLOCK, LD_DECAY)

    # Annotations: binary, 20% prevalence. Only the first N_INFORMATIVE matter.
    annot = (rng.random((P, N_ANNOT)) < 0.20).astype(float)

    # Prior probability of being causal, log-linear in the informative columns.
    log_w = LOG_ENRICHMENT * annot[:, :N_INFORMATIVE].sum(axis=1)
    prob = np.exp(log_w - log_w.max())
    prob /= prob.sum()
    causal = rng.choice(P, size=N_CAUSAL, replace=False, p=prob)

    # Effect sizes scaled so the region explains HERITABILITY of the phenotype.
    beta = np.zeros(P)
    beta[causal] = rng.normal(0, 1, size=N_CAUSAL)
    beta[causal] *= np.sqrt(HERITABILITY / N_CAUSAL) / np.std(beta[causal], ddof=0)

    # Marginal z-scores under LD:  z ~ MVN(sqrt(N) * LD @ beta, LD)
    chol = np.linalg.cholesky(ld)
    z = np.sqrt(N_SAMPLES) * (ld @ beta) + chol @ rng.standard_normal(P)
    return ld, annot, causal, z


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)

    manifest = ["z\tLD\tannot\ttarget\tN"]
    truth = ["region\tvariant\tvariant_index\tis_causal"]

    for r in range(1, N_REGIONS + 1):
        ld, annot, causal, z = simulate_region(rng)
        names = [f"r{r}_snp{i:03d}" for i in range(P)]

        z_path     = os.path.join(OUT, f"region{r}.z")
        ld_path    = os.path.join(OUT, f"region{r}.ld")
        annot_path = os.path.join(OUT, f"region{r}.annot")

        with open(z_path, "w") as f:
            for nm, zi in zip(names, z):
                f.write(f"{nm} {zi:.6f}\n")

        np.savetxt(ld_path, ld, fmt="%.6f", delimiter=" ")

        with open(annot_path, "w") as f:
            for nm, row in zip(names, annot):
                f.write(nm + "\t" + "\t".join(f"{v:.0f}" for v in row) + "\n")

        # Paths in the manifest are relative to where you run beatrice_joint.py,
        # i.e. the repository root.
        rel = os.path.join("examples", "data")
        manifest.append("\t".join([
            os.path.join(rel, f"region{r}.z"),
            os.path.join(rel, f"region{r}.ld"),
            os.path.join(rel, f"region{r}.annot"),
            os.path.join("examples", "results_joint", f"region{r}"),
            str(N_SAMPLES),
        ]))

        for i, nm in enumerate(names):
            truth.append(f"region{r}\t{nm}\t{i}\t{int(i in set(causal))}")

        top = np.argsort(-np.abs(z))[:5]
        print(f"region{r}: causal at {sorted(int(c) for c in causal)}; "
              f"largest |z| at {sorted(int(t) for t in top)}")

    with open(os.path.join(OUT, "manifest.tsv"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    with open(os.path.join(OUT, "truth.tsv"), "w") as f:
        f.write("\n".join(truth) + "\n")

    print(f"\nWrote {N_REGIONS} regions of {P} variants to {OUT}")
    print(f"Annotation columns 0..{N_INFORMATIVE - 1} are informative; "
          f"{N_INFORMATIVE}..{N_ANNOT - 1} are noise.")


if __name__ == "__main__":
    main()
