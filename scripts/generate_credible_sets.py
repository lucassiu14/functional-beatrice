import math
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scripts.convert_to_gpu import gpu
from scripts.convert_to_gpu_and_tensor import gpu_t
from scripts.convert_to_gpu_scalar import gpu_ts
from scripts.convert_to_cpu import cpu
from scripts.ld_regularisation import regularize_ld
import os

matplotlib.use('Agg')


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 4)

def load_object(filename):
    with open(filename, 'rb') as input:   # Overwrites any existing file.
        obj = pickle.load(input)
    return obj


def cond_prob(M, K, bp):
    total = 0
    for k in M:
        if len(k)==K:
            total += np.squeeze(M[k])
    pip = np.zeros(bp)

    for k in M:
        if len(k)==K:
            for j in k:
                pip[j] += np.squeeze(M[k])
    pip = pip/total

    return pip

def find_cond_prob_constrained_causal_no_dup(given, n_caus, M, bp, cred_set):

    cred = []
    for c in cred_set:
        cred+=c
    cred = set(cred)
    total = 0
    ss = given[:]
    # float64, NOT np.array([0]*bp): that is int64, and these accumulate
    # unnormalised Bayes factors. Adding a float to an int64 array silently
    # truncates it (so a BF below 1 becomes 0), and a value past int64
    # raises OverflowError - which is what a larger sparse_concrete does,
    # because it explores more configurations and pushes the memo higher.
    p = np.zeros(bp, dtype=np.float64)

    for k in M:
        tag = True
        if tag:
            if all([s in k for s in ss]):
                total+=M[k]
                for kk in k:
                    if kk not in given and kk not in cred:
                        p[kk] += M[k]


    p = p/total

    p = p.squeeze()
    index_max = np.argmax(p)

    return index_max, p[index_max],p

def find_cond_prob_constrained_causal(given, n_caus, M, bp):

    total = 0
    ss = given[:]
    # float64, NOT np.array([0]*bp): that is int64, and these accumulate
    # unnormalised Bayes factors. Adding a float to an int64 array silently
    # truncates it (so a BF below 1 becomes 0), and a value past int64
    # raises OverflowError - which is what a larger sparse_concrete does,
    # because it explores more configurations and pushes the memo higher.
    p = np.zeros(bp, dtype=np.float64)

    for k in M:
            if all([s in k for s in ss]):
                total+=M[k]
                for kk in k:
                    if kk not in given:
                        p[kk] += M[k]


    p = p/total

    p = p.squeeze()
    index_max = np.argmax(p)

    return index_max, p[index_max],p


def cond_stepwise_causal(M, mean_memo, pip, prior_causal, threshold, Z, LD,\
                         n_sub, sigma_sq, p0, S, bp, allow_dup):
    """ Create key set.
    """

    ind       = np.argsort(pip)[::-1]

    start_set = sorted(ind[:1])

    if pip[start_set[0]]<threshold:
        return []

    for nn in range(prior_causal-1):

        add(M, mean_memo, Z, LD, n_sub, sigma_sq, p0, S, sorted(start_set[:]), range(bp))

        if allow_dup:
            new_index_snp, index_prob,_ = find_cond_prob_constrained_causal(start_set, nn+2, M, bp)

        else:

            new_index_snp, index_prob,_ = find_cond_prob_constrained_causal_no_dup(start_set, nn+2, M, bp, [])

        if index_prob>threshold:
            start_set.append(new_index_snp)
        else:
            return start_set#sorted(start_set)
    return start_set        #sorted(start_set)



def find_credible_set(LD, M, start_set, bp, ths, prob_ths, allow_dup, options):
    cred_prob = []
    cred_set =  []
    start_set = list(start_set)

    for i, s1 in enumerate(start_set)  :

        ss = start_set[:]
        ss.pop(i)


        # local cred sets
        cr_set = [s1]
        p_set = []

        # posterior cond prob.
        if allow_dup:
            _,_, p = find_cond_prob_constrained_causal(ss, 0, M, bp)
        else:

            _,_, p = find_cond_prob_constrained_causal_no_dup(ss, 0, M, bp, cred_set)

        ind_cred = np.argsort(p)[::-1]

        p_set.append(np.round(p[s1], decimals=3))

        # threshold tracker.
        prob = p[s1]

        for it in ind_cred:

            if prob > ths:
                break
            else:
                if it != s1 and p[it]>prob_ths:

                    if np.min(np.abs(cpu(LD[cr_set, it]).data.numpy()))> options['purity']:
                        cr_set.append(it)
                        p_set.append(np.round(p[it], decimals=3))
                        prob += p[it]

        LLD  = LD[:,cr_set]
        LLD = LLD[cr_set,:]
        LLD =np.abs(cpu(LLD).data.numpy())

        cred_set.append(cr_set)
        cred_prob.append(p_set)
    return cred_set, cred_prob


def abf( z, ld, memo, mean_memo, n_sub, sigma_sq, p0, ind, S):
    '''
    Aproximate bayes factor calculation.

    Parameters
    ----------
    z : z scores.
    ld : LD Matrix.
    memo : Dynamic programming dict.
    mean_memo : log(exp(constant))
    n_sub : Number of subjects
    sigma_sq : Variance
    p0 : Prior
    ind : indices of causal configurations
    S : \Sigma^{-1}z

    '''
    z = z.unsqueeze(1)
    ind_m  = tuple(sorted(ind[:]))
    cc = gpu(torch.ones(len(z)))
    if len(ind)>0:
        if ind_m in memo:
            return memo[ind_m]

        # Contracted Woodbury (results.tex eq:contract): U is n*diag(sigma^2 c)
        # restricted to |ind| columns and cc is all-ones here, so U carries only
        # |ind| non-zeros. Building it dense as (m_r x m_r) and forming
        # sigma_inv = U inv V dense, only to contract to a scalar, costs
        # 2 m_r^2 |ind| flops for a result obtainable in O(m_r |ind| + |ind|^3).
        # Algebraically identical; float32 results differ in the last bits.
        w              = n_sub * sigma_sq * cc[ind]           # (K,)
        VU             = ld[ind][:, ind] * w.unsqueeze(0)     # (K, K) == V @ U
        zU             = (z[ind].squeeze(-1) * w).unsqueeze(0)  # (1, K) == z^T @ U
        VS             = torch.mm(ld[ind, :], S)              # (K, 1) == V @ S
        eyeK           = gpu(torch.eye(len(ind)))
        inv            = torch.inverse(eyeK + VU)
        sigma          = eyeK + VU
        sigma2         = torch.mm(torch.mm(zU, inv), VS)/2

        prior = 1 - p0
        prior[ind] = p0[ind]
        ex = -torch.logdet(sigma)/2 + sigma2 + torch.sum(torch.log(prior)) - mean_memo
        # Was: min(torch.tensor(1e15), torch.exp(min(torch.log(torch.tensor(1e15)), ex)))
        # Python's min() on two tensors does an element-wise comparison, and
        # those literals build CPU 0-dim tensors while `ex` may live on an
        # accelerator. clamp expresses the same ceiling with no cross-device
        # comparison and no host scalar, so it is device-agnostic. Identical
        # numerically: both cap the value at 1e15.
        _CAP = 10**15
        res = torch.clamp(torch.exp(torch.clamp(ex, max=math.log(_CAP))), max=_CAP)


        memo[ind_m] = float(res.detach().cpu().numpy().squeeze())

        return res
    else:
        return

def delete(memo, mean_memo, z, ld, n_sub, sigma_sq, p0, S, st_Set, search_Set):

        abf(z, ld, memo, mean_memo, n_sub, sigma_sq, p0, st_Set, S)

        K = list(st_Set)

        for i in range(len(K)):
            kk = K[:]
            kk.pop(i)
            kk = sorted(kk)
            abf(z, ld, memo, mean_memo, n_sub, sigma_sq, p0, kk,S)
        return

def add(memo, mean_memo, z, ld, n_sub, sigma_sq, p0, S, st_Set, search_set):

        abf(z, ld, memo, mean_memo, n_sub, sigma_sq, p0, st_Set, S)

        K = list(st_Set)
        for i in search_set:
            kk = K[:]
            if i not in K:
                kk.append(i)
            kk = sorted(kk)

            abf(z, ld, memo, mean_memo, n_sub, sigma_sq, p0, kk, S)
        return

def change(memo, mean_memo, z, ld, n_sub, sigma_sq, p0, S, st_Set,search_set):

        abf(z, ld, memo, mean_memo, n_sub, sigma_sq, p0, st_Set, S)

        K = list(st_Set)

        for j, k_con in enumerate(K):

            tem_K = K[:]
            tem_K.pop(j)
            for i in search_set:
                kk = tem_K[:]
                if i not in K and i !=k_con:
                    kk.append(i)
                elif i!=k_con:
                    rem_id = kk.index(i)
                    kk.pop(rem_id)
                kk = sorted(kk)

                abf(z, ld, memo, mean_memo, n_sub, sigma_sq, p0, kk, S)
        return

def calculate_pip(memo,bp):

    pip = np.zeros(bp)
    tot = 0.0
    for k in memo:
        val = float(np.asarray(memo[k]).squeeze())
        tot += val
        for i in k:
            pip[i] += val

    if tot <= 0:
        return pip
    return np.squeeze(pip/tot)



def unique_sets(cred_set, cred_prob):
    D = []
    P = []
    C = []
    for i, xx in enumerate(cred_set):
        ids = np.argsort(xx)[::-1]
        x = np.array(xx)[ids]
        if  list(x) not in D:
            D += [list(x)]
            C += [xx]
            P += [cred_prob[i]]
    return C, P


    x = D

def main(options):
    print('GENERATING CREDIBLE SETS\n')
    ###############################################################################################################################################

    # The caller may hand over the tensors it has already built. The joint
    # trainer has Z, the regularised LD and S in memory when it calls us, so
    # re-deriving them here re-parsed an m_r x m_r TEXT matrix, ran a second
    # eigendecomposition and a second m_r x m_r inverse per region - all O(m^3)
    # or worse, and all duplicates. Identical values, so nothing changes
    # numerically; it is the same objects.
    if options.get('Z_t') is not None:
        Z, LD, S = options['Z_t'], options['LD_t'], options['S_t']
    else:
        Z  = gpu_t(pd.read_table(options['z'],  sep=' ', header=None).to_numpy()[:,1].astype(float))
        # .npy as well as whitespace text; see read_ld in joint_trainer.py for why
        LD = gpu_t(np.load(options['LD']) if str(options['LD']).endswith('.npy')
                   else pd.read_table(options['LD'], sep=' ', header=None).to_numpy())
        LD = regularize_ld(LD)
        # solve(A, b) rather than inverse(A) @ b: same result, ~3x fewer flops
        # and better conditioned.
        S = torch.linalg.solve(LD, Z.unsqueeze(1))
    bp = len(Z)
    # The learned per-variant prior, when the caller holds it. The joint
    # trainer evaluates its shared prior head to get p_0, scores every
    # TRAINING configuration with it and normalises the memo by it
    # (reformat_memo), so the sweep must score with the SAME p_0. Otherwise
    # the two halves of one memo sit on different scales and calculate_pip
    # normalises them together: measured on UKBB cholesterol ldEUR_08_0045 the
    # gap was 96.09 nats, +2.67 per variant, systematically favouring whatever
    # training produced.
    #
    # Worse, `prior_location` is '' on every path that reaches here -
    # joint_trainer passes it literally, and beatrice_annot/beatrice_joint
    # both default it to '' - so the bare `except` this replaces turned every
    # run into a UNIFORM prior, and the learned annotation prior never reached
    # a single reported PIP. In the 232 of 254 regions where the sweep's
    # configurations hold the posterior mass, the published PIPs were
    # annotation-free.
    if options.get('p0_t') is not None:
        p0 = options['p0_t']
        # Fail fast and say which prior is in force. A 21 h run that silently
        # used the wrong prior is the failure mode this whole fix is about, so
        # the log must record it. reshape rather than assert on the shape:
        # trainer_annot's abf does the same `prior[ind] = p0[ind]` and already
        # works with whatever the prior head returns, so a stricter check here
        # could reject a run that would have been fine.
        if p0.numel() != bp:
            raise ValueError(f"p0_t has {p0.numel()} entries for {bp} variants")
        p0 = p0.reshape(-1)
        _lo, _hi = float(p0.min()), float(p0.max())
        if not (0.0 < _lo and _hi < 1.0):
            raise ValueError(f"p0_t outside (0,1): min {_lo:.3e} max {_hi:.3e}")
        print("\n Prior: the caller's LEARNED per-variant p0 "
              "(min {:.3e}, median {:.3e}, max {:.3e}, uniform would be {:.3e})"
              .format(_lo, float(p0.median()), _hi, 1.0/bp))
    elif options.get('prior_location'):
        # A path was given, so failing to read it is an error. Falling back to
        # uniform here is precisely what hid the bug above for months.
        p0 = gpu_t(pd.read_table(options['prior_location'], sep=' ',
                                 header=None).to_numpy()[:, 1].astype(float))
    else:
        p0 = gpu_t(np.array([1/len(Z)]*len(Z)))
        print("\n Prior: UNIFORM 1/{} - no p0_t and no prior_location. The "
              "learned annotation prior is NOT in use.".format(bp))

    sigma_sq =  gpu_ts(options['sigma_sq'])
    m  = load_object(os.path.join(options['target'],'res'))['memo']
    mean_memo = load_object(os.path.join(options['target'],'res'))['mean_memo']


    ###############################################################################################################################################
    pip = calculate_pip(m, bp)
    threshold_causal = options['key_thres']
    prior_n_causal   = min(bp,options['n_causal'] )
    n_sub = options['n_sub']
    allow_dup = options['allow_duplicates']
    start_set = cond_stepwise_causal(m, mean_memo, pip, prior_n_causal, threshold_causal, Z, LD, n_sub, sigma_sq, p0, S, bp, allow_dup)

    ################################################################################################################################################

    for ij, ji in enumerate(start_set):
        ss = start_set[:]
        ss.pop(ij)
        add(m, mean_memo, Z, LD, n_sub, sigma_sq, p0, S,ss, range(bp))

    #########################################################################################################################################################
    # PIPs are computed and written BEFORE the credible-set search, for two
    # reasons. (1) find_credible_set can fail - at a large sparse_concrete it
    # overflowed - and the PIPs are the expensive part to recompute, so losing
    # them to a crash in machinery the caller may not even want is wrong.
    # (2) It keeps pip.csv holding the same quantity whether or not sets are
    # built. Set `options['skip_cred_sets']` to stop here.
    pip = calculate_pip(m, bp)
    df = pd.DataFrame({'variant_index': list(range(bp)), 'pip': pip,
                       'variant_names': options['names']})
    df.to_csv(os.path.join(options['target'], 'pip.csv'), index=False)
    if options.get('skip_cred_sets', False):
        return

    ths = options['coverage_ths']
    prob_ths = options['selection_prob']
    cred_set, cred_prob = find_credible_set(LD, m, start_set, bp, ths, prob_ths, allow_dup, options)

    cred_str = []
    for item in cred_set:
        s = ''
        for ii in item:
            s += str(ii) + ' '
        cred_str.append(s[:-1])

    cred_p = []
    for item in cred_prob:
        s = ''
        for ii in item:
            s += str(ii) + ' '
        cred_p.append(s)


    cred_set, cred_prob = unique_sets(cred_set, cred_prob)
    f = open(os.path.join(options['target'],'credible_set.txt'),'w')
    f.write('\n'.join(cred_str))
    f.close()


    f = open(os.path.join(options['target'],'conditional_credible_variants_probability.txt'),'w')
    f.write('\n'.join(cred_p))
    f.close()

    C= [np.zeros(3)]
    for i in range(len(cred_set)):
        C.append(np.random.rand(3))
    c =[C[0] for i in range(bp)]
    for ii, i in enumerate(cred_set):
        for j in i:
            c[j] = C[ii+1]

    plt.scatter(range(bp), pip, c=c)
    plt.xlabel('variants')
    plt.ylabel('PIP')
    plt.savefig(os.path.join(options['target'],'credible_set.pdf'))
    plt.close()
