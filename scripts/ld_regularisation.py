#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical LD regularisation.

This lived as two byte-identical copies, in trainer_annot.py and
generate_credible_sets.py. That is the shape of a bug we have now paid for
twice: the memo-normalisation fix landed in reformat_memo and not in abf, and
the two halves of the memo then sat on different scales for months. One
definition, imported by both.
"""
import numpy as np
import torch

from scripts.convert_to_gpu import gpu
from scripts.convert_to_cpu import cpu

# Directions whose eigenvalue the panel reports as negative carry no
# information: the value is error. Clipping them to a FIXED 1e-3 asserts real
# information at 1e-3 there, and the ABF's quadratic form
# 0.5 * sum_j (q_j' z)^2 / lambda_j then amplifies whatever z-noise projects
# onto them by up to 1/1e-3 = 1000x. The panel's own error scale is a far
# better floor, and |min eig| estimates it directly.
#
# Measured on UKBB cholesterol ldEUR_08_0045 (p=3,665, min eig -5.044e-02):
# the winning 36-variant configuration scored a quadratic term of 745.88
# against 85.12 for the single true lead at |Z|=13.05, and took PIP 1.0 on a
# variant at |Z|=0.04 while the lead got 0. Flooring at |min eig| bounds the
# amplification at ~20x instead of ~1000x.
#
# Pass eig_floor=1e-3 to reproduce the previous behaviour exactly.
ADAPTIVE = None


def regularize_ld(LD, eig_floor=ADAPTIVE):
    """Project LD onto the nearest PD correlation matrix.

    eig_floor
        None (default): adaptive, max(1e-3, |min eig|). See the module comment
        for why a fixed 1e-3 is not safe at large n.
        A float: that floor, used as given. 1e-3 is the historical value.
    """
    LD = (LD + LD.T)/2
    # The fast path only needs to know whether min(eig) > 1e-3, and
    # A - 1e-3*I is positive definite EXACTLY when it is. Cholesky settles
    # that far faster than any eigendecomposition: at m_r = 2000, 0.022s
    # against 0.273s for eigvalsh and 172.6s for the np.linalg.eig this
    # originally called (measured on a CX3 core). For LD built from a few
    # thousand samples the fast path is the common one - no shift is needed at
    # all - so the eigendecomposition is skipped entirely.
    #
    # It is also exactly right under the adaptive floor: min(eig) > 1e-3 means
    # there are no negative eigenvalues, so |min eig| contributes nothing and
    # the adaptive floor is itself 1e-3, making the clip a no-op.
    _A = cpu(LD).data.numpy()
    if eig_floor is None or eig_floor <= 10**-3:
        try:
            np.linalg.cholesky(_A - (10**-3) * np.eye(_A.shape[0], dtype=_A.dtype))
            return LD
        except np.linalg.LinAlgError:
            pass

    # Only reached when the matrix is not positive definite, which on real
    # reference LD is always: the UKBB panels are float32 and computed on a
    # different sample, so min(eig) runs -0.017 to -0.069 across LDetect
    # blocks. Simulated LD is PD (min(eig) ~ +0.042) and takes the fast path
    # above, untouched.
    #
    # Project onto the nearest PD CORRELATION matrix - clip the eigenvalues
    # and renormalise the diagonal back to one - rather than adding
    # (1e-3 - min(eig)) to the diagonal as this used to.
    #
    # The diagonal shift leaves the off-diagonals alone while inflating the
    # diagonal, so every implied correlation is scaled by 1/(1 + c). At
    # c = 0.0703 that turned r = 0.996 into 0.931, and the approximate Bayes
    # factor has a crossover near r = 0.955: above it, including two
    # collinear variants together is penalised (-11.1 nats at r = 0.996);
    # below it, FAVOURED (+8.2 nats at r = 0.931). The shift therefore pushed
    # every strongly-correlated pair across that line - 100% of sampled pairs
    # with raw |r| > 0.955 on one block - and the posterior then preferred to
    # include both members of a perfectly-tagged pair. Measured consequence:
    # sum(PIP) of 1.97 over four variants at r > 0.99, where a coherent
    # posterior can spend at most about 1.
    #
    # Clipping preserves the structure instead: on that block r went
    # 0.9962 -> 0.9961 (against 0.9308 under the shift), max |change| in any
    # correlation 0.0040 against 0.0657, and co-inclusion is penalised again.
    # It costs an eigh plus two matmuls rather than eigvalsh - 59s against 43s
    # at p = 8,161 - in a step that is already O(p^3).
    A = _A.astype(np.float64)
    s, Q = np.linalg.eigh(A)
    floor = (max(10**-3, abs(min(float(s.min()), 0.0)))
             if eig_floor is None else float(eig_floor))
    A = (Q * np.clip(s, floor, None)) @ Q.T
    d = np.sqrt(np.diag(A))
    A /= np.outer(d, d)
    print("\n Projected LD onto the nearest PD correlation matrix "
          "(min eigenvalue was {:.3e}, floor {:.3e}, max |change| {:.4f})".format(
              float(s.min()), floor, float(np.abs(A - _A).max())))
    return gpu(torch.from_numpy(A.astype(_A.dtype)))
