"""
joint_trainer.py  -  CROSS-REGION JOINT prior training for (Functional) BEATRICE.

Motivation (Iteration 003, 2026-07-22). Functional BEATRICE fits its annotation
prior (`LassoNetPrior`) from a SINGLE locus, an underpowered estimation problem
that over-fits annotation noise and worsens calibration. This module trains ONE
shared prior head jointly across ALL regions of a scenario, exactly as agreed with
the user:

    L(phi, {psi_r}) = sum_r  ELBO_r( psi_r ; p0_r = f_phi(v_r) )

  - phi  = the ONE shared prior head (annotations -> per-SNP prior). It sees every
           region's evidence at every optimiser step.
  - psi_r= region-specific variational finemapping posterior (its own `network`).
           These stay SEPARATE - which SNPs are causal differs per region.
  - only p0 = f_phi(v) is shared; the likelihood (Z_r, LD_r) is per-region.

Each optimiser step: accumulate the (equal-weighted) mean of the per-region losses,
ONE backward, ONE step -> phi's gradient is (1/R) sum_r d ELBO_r / d phi. This is the
JOINT scheme, NOT the sequential `--prior_weights` warm-start chain (order-dependent,
catastrophic forgetting). Gradient aggregation over regions is EQUAL WEIGHT PER
REGION (user-confirmed 2026-07-22): each locus is one exchangeable draw of the shared
enrichment, so we do NOT weight by region size.

Two swappable prior heads select the two user model ideas:
  - prior_head='lassonet' -> the existing `LassoNetPrior`      (idea #2)
  - prior_head='linear'   -> `LinearPrior` below (logistic map) (idea #1, flagship)

The per-region ELBO math (`_region_elbo`, `_abf`) is copied verbatim from
`finemapper_lassonet.train`/`.abf` in trainer_annot.py (single-region, tested, in
production) so the JOINT and single-region paths use identical likelihood/KL/reg
terms - only the optimisation is cross-region. trainer_annot.py is left UNTOUCHED.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from scripts.convert_to_gpu import gpu
from scripts.convert_to_gpu_and_tensor import gpu_t
from scripts.convert_to_gpu_scalar import gpu_ts
from scripts.convert_to_cpu import cpu
from scripts.trainer_annot import (
    network, LassoNetPrior, regularize_ld, calculate_pip, reformat_memo,
    save_object,
)
import scripts.generate_credible_sets as gen_cred


# ---------------------------------------------------------------------------
# Prior head for idea #1: a simple SHARED logistic map on annotations.
# Same forward interface as LassoNetPrior (returns ..., imp_o = p0), so the joint
# loop is head-agnostic. No hidden layers, no L1 / hierarchy (proximal ops are
# no-ops), giving a funmap/PolyFun-style global-coefficient prior.
# ---------------------------------------------------------------------------
class LinearPrior(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.skip = nn.Linear(m, 2, bias=True)   # 2-logit softmax, matches LassoNet
        self.imp = []
        self.feature_importance = None

    def compute_feature_importance(self):
        theta = self.skip.weight.T               # (m, 2)
        contrast = theta[:, 1] - theta[:, 0]     # identifiable logit contrast
        self.feature_importance = torch.abs(contrast).detach().cpu().numpy()
        return torch.abs(contrast)

    # L1 / hierarchy are meaningless for a plain linear head -> no-ops so the joint
    # loop can call them unconditionally.
    def proximal_l1(self, lambda_l1, lr):
        return
    def apply_hierarchy_constraint(self):
        return
    def l1_penalty(self):
        return gpu_ts(0.0)

    def gumbel(self, alpha, t):
        u = (-torch.log(-torch.log(torch.rand_like(alpha) + 1e-10) + 1e-10) + alpha) / t
        return torch.nn.functional.softmax(u, dim=1)

    def forward(self, X, T, samples):
        eps = 1e-7
        out = self.skip(X)                       # (bp, 2)
        imp = torch.exp(out)
        imp_o = imp[:, 1] / (torch.sum(imp, dim=1) + eps)
        self.imp = imp_o.detach().cpu().numpy()
        self.compute_feature_importance()
        if self.training:
            z_N = self.gumbel(torch.log(imp.repeat(samples, 1) + eps), T)
            bin_concrete = z_N[:, 1].reshape(samples, len(imp_o))
            return bin_concrete, None, None, imp_o
        return None, None, None, imp_o


# ---------------------------------------------------------------------------
# Per-region ELBO. Copied from finemapper_lassonet.train / .abf (trainer_annot.py
# lines ~554-668) so JOINT and single-region share identical math. `S` and `memo`
# are per-region (passed in); the shared prior head supplies p_0 = f_phi(v).
# Returns the differentiable region loss; populates `memo` in place (for gen_cred).
# ---------------------------------------------------------------------------
def _abf(S, z, ld, memo, n_sub, sigma_sq, cc, p0, K_C, eps):
    id_sort = np.argsort(cpu(cc).data.numpy())[::-1][:K_C]
    cc_t = cc[list(id_sort)]
    ind = sorted(id_sort[cpu(torch.where(cc_t > eps)[0]).data.numpy()])
    ind_m = tuple(ind)
    cc = gpu(torch.ones(len(z)))
    if len(ind) > 0:
        if ind_m in memo:
            return memo[ind_m]
        U = n_sub * torch.diag(sigma_sq * cc)[:, ind]
        V = ld[ind, :]
        inv = torch.inverse(gpu(torch.eye(len(ind))) + torch.mm(V, U))
        sigma_inv = torch.mm(torch.mm(U, inv), V)
        sigma = gpu(torch.eye(len(ind))) + torch.mm(V, U)
        sigma2 = torch.matmul(torch.matmul(z.T, sigma_inv), S) / 2
        prior = 1 - p0
        prior[ind] = p0[ind]
        res = -torch.logdet(sigma) / 2 + sigma2 + torch.sum(torch.log(prior))
        memo[ind_m] = float(res.detach().cpu().numpy().squeeze())
        return res
    return None


def _region_elbo(model, prior_head, S, Z, LD, v, temp, n_samples, sigma_sq,
                 n_sub, K_C, gamma, lambda_reg, memo):
    """One region's differentiable loss (lik + kl + reg), populating `memo`.
    Mirrors finemapper_lassonet.train's inner body; NO optimiser step here - the
    joint loop sums these across regions and steps ONCE."""
    eps = gpu_ts(1e-7)
    M = len(Z)
    c, c1, c2, imp = model(temp, n_samples)          # region posterior psi_r
    _, _, _, p_0 = prior_head(v, temp, 1)            # SHARED prior phi -> p0_r

    loss = gpu_ts(0.0)
    lik_loss = gpu_ts(0.0)
    Zc = Z.unsqueeze(1)
    valid_samples = 0
    best_u, best_ind = None, None

    for i in range(n_samples):
        u, ind = torch.topk(c[i], max(1, K_C))
        if ind.numel() == 0:
            continue
        valid_samples += 1
        K_C_eff = ind.numel()
        cc = c[i]
        if gamma > 0:
            _abf(S, Zc, LD, memo, n_sub, sigma_sq, cc, p_0, K_C_eff, gamma)
        U = n_sub * torch.diag(sigma_sq * cc)[:, ind]
        V = LD[ind, :]
        eyeKC = gpu(torch.eye(K_C_eff))
        eyeM = gpu(torch.eye(M))
        inv = torch.inverse(eyeKC + torch.mm(V, U))
        sigma_inv = eyeM - torch.mm(torch.mm(U, inv), V)
        sigma = eyeKC + torch.mm(V, U) + 1e-6 * eyeKC
        sigma2 = -torch.matmul(torch.matmul(Zc.T, sigma_inv), S) / 2
        log_likelihood = -torch.logdet(sigma) / 2 + sigma2
        if not torch.isfinite(log_likelihood):
            continue
        lik_loss += -log_likelihood.squeeze()
        loss += -log_likelihood.squeeze()
        if (best_u is None) or (u.max() > best_u.max()):
            best_u, best_ind = u, ind

    if valid_samples == 0:
        best_ind = torch.topk(c[0], 1).indices
        valid_samples = 1

    x2 = imp[best_ind]
    x1 = p_0[best_ind]
    kl_loss = torch.sum(x2 * (torch.log(x2 + eps) - torch.log(x1 + eps))) + \
              torch.sum((1 - x2) * (torch.log(1 - x2 + eps) - torch.log(1 - x1 + eps)))
    reg_loss = lambda_reg * torch.sum(p_0 ** 2)

    region_loss = (loss / valid_samples) + kl_loss + reg_loss
    return region_loss, p_0


# ---------------------------------------------------------------------------
# The joint driver.
# ---------------------------------------------------------------------------
def _load_region(z_path, ld_path, annot_path):
    names = list(pd_read_col0(z_path))
    Z = gpu_t(pd_read(z_path)[:, 1].astype(float))
    if torch.max(Z) == torch.inf:
        Z[torch.where(Z == torch.inf)[0]] = 200
    LD = gpu_t(pd_read(ld_path))
    v = gpu_t(pd_read_annot(annot_path)[:, 1:].astype(float))
    if torch.max(v) == torch.inf:
        v[torch.where(v == torch.inf)[0]] = 200
    return names, Z, regularize_ld(LD), v


# small pandas readers kept local to avoid re-importing in hot code
import pandas as pd
def pd_read(path):
    return pd.read_table(path, sep=' ', header=None).to_numpy()
def pd_read_col0(path):
    return pd.read_table(path, sep=' ', header=None).to_numpy()[:, 0]
def pd_read_annot(path):
    return pd.read_table(path, sep=None, engine='python', header=None).to_numpy()


def run_joint(options):
    """
    options keys:
      regions            : list of dicts {z, LD, annot, target, N}
      prior_head         : 'lassonet' | 'linear'
      NN                 : finemapping-net hidden dims  (default [10,200,10])
      prior_neural_network : LassoNet hidden dims       (default [10,200,10])
      max_iter, sigma_sq, sparse_concrete, MCMC_samples, temp_lower_bound,
      gamma, prior_regularisation, lambda_l1, hierarchy_M,
      n_causal, coverage_ths, selection_prob, key_thres, allow_duplicates,
      save_prior_weights (path or '')
    """
    torch.manual_seed(1)
    import random; random.seed(1)

    regions_cfg = options['regions']
    R = len(regions_cfg)
    if R == 0:
        raise ValueError("run_joint: no regions supplied")

    sigma_sq = options['sigma_sq']
    n_samples = options['MCMC_samples']
    n_epochs = options['max_iter']
    temp_lb = gpu_ts(options['temp_lower_bound'])
    gamma = options['gamma']
    lambda_reg = options['prior_regularisation']
    lambda_l1 = options.get('lambda_l1', 0.01)
    hierarchy_M = options.get('hierarchy_M', 10.0)
    NN = options['NN']
    prior_hidden = options['prior_neural_network']

    # --- load every region + build region-specific state --------------------
    reg = []
    m_annots = None
    for rc in regions_cfg:
        names, Z, LD, v = _load_region(rc['z'], rc['LD'], rc['annot'])
        if m_annots is None:
            m_annots = v.shape[1]
        elif v.shape[1] != m_annots:
            raise ValueError(f"annotation count differs across regions: "
                             f"{v.shape[1]} vs {m_annots}")
        model = gpu(network(len(Z), [1] + NN, 3, LD, Z))
        S = torch.matmul(torch.inverse(LD), Z.unsqueeze(1))
        reg.append(dict(names=names, Z=Z, LD=LD, v=v, model=model, S=S,
                        memo={}, bp=len(Z), N=rc['N'], target=rc['target'],
                        z=rc['z'], LD_path=rc['LD']))

    # --- ONE shared prior head phi ------------------------------------------
    if options['prior_head'] == 'linear':
        prior_head = gpu(LinearPrior(m=m_annots))
        head_desc = "LinearPrior (idea #1: shared logistic annotation prior)"
    else:
        prior_head = gpu(LassoNetPrior(m=m_annots, hidden_dims=prior_hidden, M=hierarchy_M))
        head_desc = f"LassoNetPrior (idea #2: shared, hidden={prior_hidden}, M={hierarchy_M})"
    print(f"[joint] {R} regions, {m_annots} annotations, head={head_desc}")

    # --- ONE optimiser over all region posteriors + the shared prior --------
    params = list(prior_head.parameters())
    for r in reg:
        params += list(r['model'].parameters())
    opt = optim.Adam(params, lr=0.002, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)

    prior_head.train()
    for r in reg:
        r['model'].train()

    # --- JOINT training loop: equal-weight mean over regions, ONE step ------
    from tqdm.auto import tqdm
    for n in tqdm(range(n_epochs + 1), desc="joint"):
        temp = torch.max(temp_lb, gpu_ts(np.exp(-0.0001 * n)))
        opt.zero_grad()
        total = gpu_ts(0.0)
        for r in reg:
            K_C = min(r['bp'], options['sparse_concrete'])
            region_loss, _ = _region_elbo(
                r['model'], prior_head, r['S'], r['Z'], r['LD'], r['v'],
                temp, n_samples, gpu_ts(sigma_sq), r['N'], K_C, gamma,
                lambda_reg, r['memo'])
            total = total + region_loss / R          # EQUAL weight per region
        total.backward()
        opt.step()
        scheduler.step()
        # proximal L1 + hierarchy on the SHARED head, once per step (no-ops for linear)
        prior_head.proximal_l1(lambda_l1, opt.param_groups[0]['lr'])
        prior_head.apply_hierarchy_constraint()

    # --- per-region outputs: save res + generate credible sets --------------
    prior_head.eval()
    for r in reg:
        with torch.no_grad():
            _, _, _, p_0 = prior_head(r['v'], temp, 1)
        mean_memo = reformat_memo(r['memo'], p_0)
        res = {'memo': r['memo'], 'mean_memo': mean_memo,
               'names': r['names'], 'pip': calculate_pip(r['memo'], r['bp']),
               'feature_importance': prior_head.feature_importance}
        os.makedirs(r['target'], exist_ok=True)
        save_object(res, os.path.join(r['target'], 'res'))
        gen_opts = {
            'z': r['z'], 'LD': r['LD_path'], 'prior_location': '',
            'sigma_sq': sigma_sq, 'n_sub': r['N'], 'target': r['target'],
            'names': r['names'], 'key_thres': options['key_thres'],
            'n_causal': options['n_causal'], 'allow_duplicates': options['allow_duplicates'],
            'coverage_ths': options['coverage_ths'], 'selection_prob': options['selection_prob'],
            'purity': options.get('purity', 0.0),
        }
        gen_cred.main(gen_opts)

    # --- save the shared prior head (for inspection / reuse) ----------------
    if options.get('save_prior_weights'):
        ckpt = {'state_dict': prior_head.state_dict(),
                'n_annotations': m_annots, 'hidden_dims': prior_hidden,
                'hierarchy_M': hierarchy_M, 'prior_head': options['prior_head'],
                'feature_importance': prior_head.feature_importance}
        torch.save(ckpt, options['save_prior_weights'])
        print(f"[joint] saved shared prior head -> {options['save_prior_weights']}")
