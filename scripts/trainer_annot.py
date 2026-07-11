import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import shutil
import os
import glob
from scripts.convert_to_gpu import gpu
from scripts.convert_to_gpu_and_tensor import gpu_t
from scripts.convert_to_gpu_scalar import gpu_ts
from scripts.convert_to_cpu import cpu
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import imageio
import seaborn as sn
from tqdm.auto import tqdm
import scripts.generate_credible_sets as gen_cred
import pandas as pd
import time

matplotlib.use('Agg')



def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 4)
        
def load_object(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        obj = pickle.load(input)
    return obj

def calculate_pip(M,bp):
    """Calculate posterior inclusion probabilities.
    """
    pip = np.zeros(bp)
    tot = 0.0
    for k in M:
        val = float(np.asarray(M[k]).squeeze())
        tot += val
        for i in k:
            pip[i] += val

    if tot <= 0:
        return pip
    return np.squeeze(pip/tot)

def make_gif(M, bp, loc, store, ep):
    tr      = np.zeros(bp)
    tr[loc] = 1
    pip = []
    for k in M:   
        pip+= list(k)
    sn.histplot(pip, stat='count', bins=100,element='poly')
    #plt.stem(tr*(ep/2))
    plt.savefig(store+'/'+str(ep)+'.png')
    plt.close()
    
    list_of_files = filter( os.path.isfile,
                        glob.glob(store+ '/*.png') )
    # Sort list of files based on last modification time in ascending order
    filenames = sorted( list_of_files,
                        key = os.path.getmtime)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(store+'/movie.gif', images)
    return 
    
def regularize_ld(LD):
    """Regularize LD to make it non-singular
    """
    LD = (LD + LD.T)/2
    s, w = np.linalg.eig(cpu(LD).data.numpy())
    s = np.real(s)
    s_new = torch.zeros(len(s))
    if min(s)<10**-3:
        s_new = torch.ones(len(s))*(min(s)-10**-3)   
        print("\n Adding a constant {} to regularize LD".format(-min(s)+10**-3))
    LD = LD - gpu(torch.diag(s_new))    
    return LD
    
    
def reformat_memo(memo, p0):
    memo[tuple([])] = float(torch.sum(torch.log(1 - p0)).item())
    m0 = float(np.mean([float(np.asarray(val).squeeze()) for val in memo.values()]))
    for key in memo:
        val = float(np.asarray(memo[key]).squeeze())
        memo[key] = float(min(10**15, np.exp(min(np.log(10**15), val - m0))))
    return m0
    return m0

class network(nn.Module):
    def __init__(self,K,f_dim,n_l,A,x):
        """Initialization of the neural network.
        """
        super(network, self).__init__()
        
  
        self.softmax  = nn.Softmax(dim=1)
        self.imp      =[]
        self.sig = nn.Sigmoid()
        self.rel = nn.ReLU()
        self.tan = nn.Tanh()
        
        self.sig           = nn.Sigmoid()

        
        self.L1 = nn.Linear(f_dim[0], f_dim[1], bias=False)
        self.N1 = nn.LayerNorm(K)
        self.A1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(f_dim[1], f_dim[2], bias=False)
            )
        self.N2 = nn.LayerNorm(K)
        self.A2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(f_dim[2], f_dim[3], bias=False)
            )
        self.N3 =  nn.LayerNorm(K)
        
        self.A3 = nn.Sequential(
            nn.ReLU()           
            )
        

        self.conc = nn.Linear(f_dim[n_l], 2,bias=False)

        
        self.A = A
        self.x = x
        self.degree = []
        self.variance = nn.Parameter(torch.rand(1))
        
        
    def gumbel(self,alpha,t):
        """ Generate Binary Concrete Vectors."""
        u = (-torch.log(-torch.log(gpu(torch.rand(alpha.size())))) + alpha)/t
        return F.softmax(u,dim=1)
    
        
    def forward(self, T, samples):
        """ The inference module which generates the parameters of the binary 
        concrete distribution and generate samples of binary concrete vectors.      
        """

        eps = gpu_ts(10**-7)
        X   = self.x.unsqueeze(1)        
        out = self.conc(self.A3(self.N3(self.A2(self.N2(self.A1(self.N1(self.L1(X).T).T).T).T).T).T)  )

        imp     = gpu(torch.exp(out))
        imp_o   = imp[:,1]/torch.sum(imp,dim=1)
        self.imp = cpu(imp_o.detach()).data.numpy()
        
        eps = gpu_ts(10**-6)
        if self.training:
                 z_N     = self.gumbel(torch.log(imp.repeat(samples, 1)+eps), T) 
                 z_N1    = self.gumbel(torch.log(imp.repeat(1, 1)+eps), T) 
                 z_N2    = self.gumbel(torch.log(imp.repeat(1, 1)+eps), T)
                 if torch.isnan(torch.max(z_N)):
                     print(torch.max(z_N))

                 bin_concrete =  z_N[:,1].reshape(samples,len(imp_o))
                 bin_concrete1 = z_N1[:,1].reshape(1,len(imp_o))
                 bin_concrete2 = z_N2[:,1].reshape(1,len(imp_o))
        return bin_concrete,bin_concrete1,bin_concrete2, imp_o

class priornetwork(nn.Module):
    def __init__(self, bp, m, hidden_dims=[16, 8]):
        """
        bp: number of variants
        m: number of annotations per variant
        hidden_dims: list of hidden layer sizes for the shared MLP
        """
        super().__init__()
        # Shared MLP for each variant's annotation vector
        layers = []
        last_dim = m
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 2, bias=False))  # Output 2 logits per variant
        self.mlp = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
        self.imp = []

    def gumbel(self, alpha, t):
        """ Generate Binary Concrete Vectors. """
        u = (-torch.log(-torch.log(torch.rand_like(alpha))) + alpha) / t
        return F.softmax(u, dim=1)

    def forward(self, X, T, samples):
        """
        X: (bp, m) annotation matrix
        T: temperature
        samples: number of MC samples
        """
        eps = 1e-7
        out = self.mlp(X)  # (bp, 2)
        imp = torch.exp(out)
        imp_o = imp[:, 1] / torch.sum(imp, dim=1)
        self.imp = imp_o.detach().cpu().numpy()

        if self.training:
            z_N = self.gumbel(torch.log(imp.repeat(samples, 1) + eps), T)
            z_N1 = self.gumbel(torch.log(imp.repeat(1, 1) + eps), T)
            z_N2 = self.gumbel(torch.log(imp.repeat(1, 1) + eps), T)
            bin_concrete = z_N[:, 1].reshape(samples, len(imp_o))
            bin_concrete1 = z_N1[:, 1].reshape(1, len(imp_o))
            bin_concrete2 = z_N2[:, 1].reshape(1, len(imp_o))
        else:
            bin_concrete = bin_concrete1 = bin_concrete2 = None

        return bin_concrete, bin_concrete1, bin_concrete2, imp_o


class LassoNetPrior(nn.Module):
    """
    LassoNet-based prior network for annotation feature selection.

    Implements the LassoNet architecture with:
    - Skip connections from input features directly to output (θ)
    - Hidden layers with hierarchy constraint
    - L1 penalty on skip connections induces feature sparsity

    The hierarchy constraint ensures: ||W^(1)_j||_2 ≤ M * |θ_j|
    If a feature's skip connection is zero, it cannot contribute through hidden layers.

    Reference: LeMeur et al., "LassoNet: A Neural Network with Feature Sparsity"
    """

    def __init__(self, m, hidden_dims=[32, 16], M=10.0):
        """
        Args:
            m: number of input annotations per variant
            hidden_dims: list of hidden layer sizes
            M: hierarchy constraint multiplier (larger M = looser constraint)
        """
        super().__init__()
        self.m = m  # number of input features
        self.hidden_dims = hidden_dims
        self.M = M

        # Skip connection: direct path from input features to output
        # θ has shape (m, 2) - each feature contributes directly to 2 output logits
        self.skip = nn.Linear(m, 2, bias=False)

        # First hidden layer - weights constrained by hierarchy
        # W^(1) has shape (m, hidden_dims[0])
        self.first_hidden = nn.Linear(m, hidden_dims[0], bias=True)

        # Remaining hidden layers (unconstrained)
        layers = []
        last_dim = hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h

        # Final layer from hidden to output
        layers.append(nn.Linear(last_dim, 2, bias=False))
        self.hidden_layers = nn.Sequential(*layers)

        self.relu = nn.ReLU()
        self.imp = []

        # Track feature importance (based on |θ|)
        self.feature_importance = None

    def get_skip_weights(self):
        """Return the skip connection weights θ (m x 2)."""
        return self.skip.weight.T  # Transpose to get (m, 2)

    def get_first_layer_weights(self):
        """Return the first hidden layer weights W^(1) (hidden_dim x m)."""
        return self.first_hidden.weight  # Shape: (hidden_dims[0], m)

    def compute_feature_importance(self):
        """
        Compute feature importance from the identifiable logit contrast.

        The two-output softmax parameterisation is invariant to adding the same
        annotation-dependent function to both logits: softmax(out + c) ==
        softmax(out), so the causal probability imp_o is unchanged. However
        the naive L2-norm importance ||theta_j||_2 IS affected by that shift,
        making it non-identifiable from the causal probability. The identifiable
        quantity is the logit contrast delta_j = theta[j, 1] - theta[j, 0];
        its magnitude reports how much annotation j moves the log-odds of the
        causal class, independent of any common shift.

        Returns tensor of shape (m,).
        """
        theta = self.get_skip_weights()          # (m, 2)
        contrast = theta[:, 1] - theta[:, 0]     # identifiable logit contrast
        importance = torch.abs(contrast)         # magnitude of contribution
        self.feature_importance = importance.detach().cpu().numpy()
        return importance

    def l1_penalty(self):
        """
        Compute L1 penalty on skip connection weights.
        Returns scalar L1 norm of θ (for logging only, not backpropagated).
        """
        theta = self.get_skip_weights()
        return torch.sum(torch.abs(theta))

    def proximal_l1(self, lambda_l1, lr):
        """
        Apply proximal operator for L1 on skip weights (soft-thresholding).

        Instead of backpropagating through |θ| (which has a discontinuous
        gradient at zero and can produce NaN via Adam momentum), this applies
        the closed-form proximal step:
            θ ← sign(θ) * max(|θ| - λ·lr, 0)

        Must be called after optimizer.step() and before apply_hierarchy_constraint().
        """
        with torch.no_grad():
            w = self.skip.weight.data          # (2, m)
            threshold = lambda_l1 * lr
            self.skip.weight.data = torch.sign(w) * torch.clamp(torch.abs(w) - threshold, min=0.0)

    def apply_hierarchy_constraint(self):
        """
        Apply the LassoNet hierarchy constraint via proximal operator.

        For each feature j: ||W^(1)_j||_2 ≤ M * |θ_j|

        This should be called after each optimizer step.
        """
        with torch.no_grad():
            theta = self.get_skip_weights()  # (m, 2)
            W1 = self.get_first_layer_weights()  # (hidden_dims[0], m)

            # For each feature j
            for j in range(self.m):
                theta_j_norm = torch.norm(theta[j], p=2)  # L2 norm of θ_j
                W1_j = W1[:, j]  # Column j of W^(1)
                W1_j_norm = torch.norm(W1_j, p=2)

                # Hierarchy constraint: ||W^(1)_j||_2 ≤ M * ||θ_j||_2
                max_allowed = self.M * theta_j_norm

                if W1_j_norm > max_allowed and W1_j_norm > 0:
                    # Scale down W^(1)_j to satisfy constraint
                    scale = max_allowed / W1_j_norm
                    self.first_hidden.weight[:, j] = W1_j * scale

    def gumbel(self, alpha, t):
        """Generate Binary Concrete Vectors."""
        u = (-torch.log(-torch.log(torch.rand_like(alpha) + 1e-10) + 1e-10) + alpha) / t
        return F.softmax(u, dim=1)

    def forward(self, X, T, samples):
        """
        Forward pass through LassoNet.

        Args:
            X: (bp, m) annotation matrix
            T: temperature for Gumbel-Softmax
            samples: number of MC samples

        Returns:
            bin_concrete, bin_concrete1, bin_concrete2, imp_o
        """
        eps = 1e-7

        # Skip connection path
        skip_out = self.skip(X)  # (bp, 2)

        # Hidden layer path
        h = self.relu(self.first_hidden(X))  # (bp, hidden_dims[0])
        hidden_out = self.hidden_layers(h)  # (bp, 2)

        # Combine skip and hidden paths
        out = skip_out + hidden_out  # (bp, 2)

        # Convert to probabilities
        imp = torch.exp(out)
        imp_o = imp[:, 1] / (torch.sum(imp, dim=1) + eps)
        self.imp = imp_o.detach().cpu().numpy()

        # Update feature importance
        self.compute_feature_importance()

        if self.training:
            z_N = self.gumbel(torch.log(imp.repeat(samples, 1) + eps), T)
            z_N1 = self.gumbel(torch.log(imp.repeat(1, 1) + eps), T)
            z_N2 = self.gumbel(torch.log(imp.repeat(1, 1) + eps), T)
            bin_concrete = z_N[:, 1].reshape(samples, len(imp_o))
            bin_concrete1 = z_N1[:, 1].reshape(1, len(imp_o))
            bin_concrete2 = z_N2[:, 1].reshape(1, len(imp_o))
        else:
            bin_concrete = bin_concrete1 = bin_concrete2 = None

        return bin_concrete, bin_concrete1, bin_concrete2, imp_o

class finemapper():
    def __init__(self, model, opt, sch):
        self.model = model
        self.opt = opt
        self.scheduler = sch
    

    def abf(self, z, ld, memo, n_sub, sigma_sq, cc, p0, K_C, eps):

        id_sort = np.argsort(cpu(cc).data.numpy())[::-1]
        id_sort = id_sort[:K_C]
        
        cc_t = cc[list(id_sort)]
        
        ind = sorted(id_sort[cpu(torch.where(cc_t>eps)[0]).data.numpy()])
        ind_m  = tuple(ind)
        cc = gpu(torch.ones(len(z)))
        
        if len(ind)>0:
            if ind_m in memo:
                return memo[ind_m]
        
            U =  n_sub*torch.diag(sigma_sq*cc)[:,ind]
            V = ld[ind,:]
                
            inv            = torch.inverse(gpu(torch.eye(len(ind))) + torch.mm(V,U))
                
            sigma_inv      = torch.mm(torch.mm(U,inv),V)
                
            sigma          = gpu(torch.eye(len(ind))) + torch.mm(V,U)
                
            sigma2         = torch.matmul(torch.matmul(z.T, sigma_inv),self.S)/2
        
            prior = 1 - p0
            prior[ind] = p0[ind]
        
            res =  -torch.logdet(sigma)/2 + sigma2 + torch.sum(torch.log(prior)) 
        
        
            memo[ind_m] = float(res.detach().cpu().numpy().squeeze())
        
            return res
        else:
            return
        
        
    def train(self, z_score, ld, temp, n_samples, sigma_sq, n_sub, p_0, num_iter, memo, epp, K_C, gamma):
        self.model.train()  # make training mode explicit

        sigma_sq = gpu_ts(sigma_sq)
        eps = gpu_ts(1e-7)
        ll_lik, ll_kl, ll_total = [], [], []
        M = len(z_score)

        for n_b in range(num_iter):
            Z  = Variable(z_score)
            LD = Variable(ld)

            self.opt.zero_grad()
            for p in self.model.parameters():
                p.requires_grad = True

            c, c1, c2, imp = self.model(temp, n_samples)
            loss = gpu_ts(0.0)
            lik_loss = gpu_ts(0.0)

            Z = Z.unsqueeze(1)

            valid_samples = 0
            best_u, best_ind = None, None  # for stable KL index choice

            for i in range(n_samples):
                # always keep at least one index; remove hard 0.01 gate
                u, ind = torch.topk(c[i], max(1, K_C))
                if ind.numel() == 0:
                    continue

                valid_samples += 1
                K_C_eff = ind.numel()
                cc = c[i]

                # optional memo usage
                if epp > 0:
                    self.abf(Z, ld, memo, n_sub, sigma_sq, cc, p_0, K_C_eff, gamma)

                U = n_sub * torch.diag(sigma_sq * cc)[:, ind]
                V = LD[ind, :]

                eyeKC = gpu(torch.eye(K_C_eff))
                eyeM  = gpu(torch.eye(M))

                inv = torch.inverse(eyeKC + torch.mm(V, U))
                sigma_inv = eyeM - torch.mm(torch.mm(U, inv), V)

                sigma = eyeKC + torch.mm(V, U)
                # tiny jitter for stability
                sigma = sigma + 1e-6 * eyeKC

                sigma2 = -torch.matmul(torch.matmul(Z.T, sigma_inv), self.S) / 2
                log_likelihood = -torch.logdet(sigma) / 2 + sigma2

                if not torch.isfinite(log_likelihood):
                    continue  # skip this sample if it’s numerically bad

                lik_loss += -log_likelihood.squeeze()
                loss     += -log_likelihood.squeeze()

                if (best_u is None) or (u.max() > best_u.max()):
                    best_u, best_ind = u, ind

            # if no valid samples, fall back to top-1 from the first sample
            if valid_samples == 0:
                best_ind = torch.topk(c[0], 1).indices
                valid_samples = 1  # to avoid div by zero

            # KL on a stable index set
            x2 = imp[best_ind]
            x1 = p_0[best_ind]
            kl_loss = torch.sum(x2 * (torch.log(x2 + eps) - torch.log(x1 + eps))) + \
                  torch.sum((1 - x2) * (torch.log(1 - x2 + eps) - torch.log(1 - x1 + eps)))

            loss_f = (loss / valid_samples) + kl_loss
            loss_f.backward()
            self.opt.step()

            ll_lik.append(cpu(lik_loss.detach()).data.numpy() / valid_samples)
            ll_kl.append(cpu(kl_loss.detach()).data.numpy())
            ll_total.append(cpu(loss_f.detach()).data.numpy())

        # always return scalars so your histories grow by exactly one per epoch
        return [np.mean(ll_total)], [np.mean(ll_lik)], [np.mean(ll_kl)]
    
        
        
    


class finemapper_lassonet():
    """
    Fine-mapper using LassoNet prior network for annotation feature selection.
    """
    def __init__(self, model, prior_network, opt, sch, lambda_l1, lambda_reg):
        self.model = model
        self.prior_network = prior_network  # LassoNetPrior instance
        self.opt = opt
        self.scheduler = sch
        self.lambda_l1 = lambda_l1  # L1 penalty weight for skip connections
        self.lambda_reg = lambda_reg  # Additional regularization

    def abf(self, z, ld, memo, n_sub, sigma_sq, cc, p0, K_C, eps):
        id_sort = np.argsort(cpu(cc).data.numpy())[::-1]
        id_sort = id_sort[:K_C]

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
            sigma2 = torch.matmul(torch.matmul(z.T, sigma_inv), self.S) / 2

            prior = 1 - p0
            prior[ind] = p0[ind]

            res = -torch.logdet(sigma) / 2 + sigma2 + torch.sum(torch.log(prior))
            memo[ind_m] = float(res.detach().cpu().numpy().squeeze())

            return res
        else:
            return

    def train(self, z_score, ld, v, temp, n_samples, sigma_sq, n_sub, num_iter, memo, epp, K_C, gamma):
        """Training loop with LassoNet prior (with hierarchy constraint and L1 penalty)."""
        self.model.train()
        self.prior_network.train()

        sigma_sq = gpu_ts(sigma_sq)
        eps = gpu_ts(1e-7)
        ll_lik, ll_kl, ll_l1, ll_reg, ll_total = [], [], [], [], []
        M = len(z_score)

        for n_b in range(num_iter):
            Z = Variable(z_score)
            LD = Variable(ld)
            v = Variable(v)

            self.opt.zero_grad()
            for p in self.model.parameters():
                p.requires_grad = True
            for p in self.prior_network.parameters():
                p.requires_grad = True

            c, c1, c2, imp = self.model(temp, n_samples)
            _, _, _, p_0 = self.prior_network(v, temp, 1)

            loss = gpu_ts(0.0)
            lik_loss = gpu_ts(0.0)
            Z = Z.unsqueeze(1)

            valid_samples = 0
            best_u, best_ind = None, None

            for i in range(n_samples):
                u, ind = torch.topk(c[i], max(1, K_C))
                if ind.numel() == 0:
                    continue

                valid_samples += 1
                K_C_eff = ind.numel()
                cc = c[i]

                if epp > 0:
                    self.abf(Z, LD, memo, n_sub, sigma_sq, cc, p_0, K_C_eff, gamma)

                U = n_sub * torch.diag(sigma_sq * cc)[:, ind]
                V = LD[ind, :]

                eyeKC = gpu(torch.eye(K_C_eff))
                eyeM = gpu(torch.eye(M))

                inv = torch.inverse(eyeKC + torch.mm(V, U))
                sigma_inv = eyeM - torch.mm(torch.mm(U, inv), V)

                sigma = eyeKC + torch.mm(V, U)
                sigma = sigma + 1e-6 * eyeKC

                sigma2 = -torch.matmul(torch.matmul(Z.T, sigma_inv), self.S) / 2
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

            # KL divergence on stable indices
            x2 = imp[best_ind]
            x1 = p_0[best_ind]
            kl_loss = torch.sum(x2 * (torch.log(x2 + eps) - torch.log(x1 + eps))) + \
                      torch.sum((1 - x2) * (torch.log(1 - x2 + eps) - torch.log(1 - x1 + eps)))

            # Additional regularization on p_0
            reg_p0 = torch.sum((p_0) ** 2)
            reg_loss = self.lambda_reg * reg_p0

            # L1 is NOT in the differentiable loss — applied as proximal step below
            loss_f = (loss / valid_samples) + kl_loss + reg_loss
            loss_f.backward()
            self.opt.step()

            # Proximal L1 on skip connections (soft-thresholding, no NaN risk)
            current_lr = self.opt.param_groups[0]['lr']
            self.prior_network.proximal_l1(self.lambda_l1, current_lr)

            # Apply hierarchy constraint after proximal step
            self.prior_network.apply_hierarchy_constraint()

            # Log L1 for monitoring (detached, not backpropagated)
            with torch.no_grad():
                l1_loss = self.lambda_l1 * self.prior_network.l1_penalty()

            ll_lik.append(cpu(lik_loss.detach()).data.numpy() / valid_samples)
            ll_kl.append(cpu(kl_loss.detach()).data.numpy())
            ll_l1.append(cpu(l1_loss.detach()).data.numpy())
            ll_reg.append(cpu(reg_loss.detach()).data.numpy())
            ll_total.append(cpu(loss_f.detach()).data.numpy() + cpu(l1_loss.detach()).data.numpy())

        return [np.mean(ll_total)], [np.mean(ll_lik)], [np.mean(ll_kl)], [np.mean(ll_l1)], [np.mean(ll_reg)]


class finemapper_annot():
    def __init__(self, model, prior_network, opt, sch, lambda_reg):
        self.model = model
        self.prior_network = prior_network
        self.opt = opt
        self.scheduler = sch
        self.lambda_reg = lambda_reg
    
    def abf(self, z, ld, memo, n_sub, sigma_sq, cc, p0, K_C, eps):

        id_sort = np.argsort(cpu(cc).data.numpy())[::-1]
        id_sort = id_sort[:K_C]
        
        cc_t = cc[list(id_sort)]
        
        ind = sorted(id_sort[cpu(torch.where(cc_t>eps)[0]).data.numpy()])
        ind_m  = tuple(ind)
        cc = gpu(torch.ones(len(z)))
        
        if len(ind)>0:
            if ind_m in memo:
                return memo[ind_m]
        
            U =  n_sub*torch.diag(sigma_sq*cc)[:,ind]
            V = ld[ind,:]
                
            inv            = torch.inverse(gpu(torch.eye(len(ind))) + torch.mm(V,U))
                
            sigma_inv      = torch.mm(torch.mm(U,inv),V)
                
            sigma          = gpu(torch.eye(len(ind))) + torch.mm(V,U)
                
            sigma2         = torch.matmul(torch.matmul(z.T, sigma_inv),self.S)/2
        
            prior = 1 - p0
            prior[ind] = p0[ind]
        
            res =  -torch.logdet(sigma)/2 + sigma2 + torch.sum(torch.log(prior)) 
        
        
            memo[ind_m] = float(res.detach().cpu().numpy().squeeze())
        
            return res
        else:
            return
        
    def train(self, z_score, ld, v, temp, n_samples, sigma_sq, n_sub, num_iter, memo, epp, K_C, gamma):
        """Training loop with annotations (robust: no silent early-returns)."""
        self.model.train()
        self.prior_network.train()

        sigma_sq = gpu_ts(sigma_sq)
        eps = gpu_ts(1e-7)
        ll_lik, ll_kl, ll_reg, ll_total = [], [], [], []
        M = len(z_score)

        for n_b in range(num_iter):
            Z  = Variable(z_score)
            LD = Variable(ld)
            v  = Variable(v)

            self.opt.zero_grad()
            for p in self.model.parameters():
                p.requires_grad = True
            for p in self.prior_network.parameters():
                p.requires_grad = True

            c, c1, c2, imp  = self.model(temp, n_samples)
            _, _, _, p_0    = self.prior_network(v, temp, 1)

            loss = gpu_ts(0.0)
            lik_loss = gpu_ts(0.0)
            Z = Z.unsqueeze(1)

            valid_samples = 0
            best_u, best_ind = None, None

            for i in range(n_samples):
                u, ind = torch.topk(c[i], max(1, K_C))
                if ind.numel() == 0:
                    continue

                valid_samples += 1
                K_C_eff = ind.numel()
                cc = c[i]

                if epp > 0:
                    self.abf(Z, LD, memo, n_sub, sigma_sq, cc, p_0, K_C_eff, gamma)

                U = n_sub * torch.diag(sigma_sq * cc)[:, ind]
                V = LD[ind, :]

                eyeKC = gpu(torch.eye(K_C_eff))
                eyeM  = gpu(torch.eye(M))

                inv = torch.inverse(eyeKC + torch.mm(V, U))
                sigma_inv = eyeM - torch.mm(torch.mm(U, inv), V)

                sigma = eyeKC + torch.mm(V, U)
                sigma = sigma + 1e-6 * eyeKC

                sigma2 = -torch.matmul(torch.matmul(Z.T, sigma_inv), self.S) / 2
                log_likelihood = -torch.logdet(sigma) / 2 + sigma2

                if not torch.isfinite(log_likelihood):
                    continue

                lik_loss += -log_likelihood.squeeze()
                loss     += -log_likelihood.squeeze()

                if (best_u is None) or (u.max() > best_u.max()):
                    best_u, best_ind = u, ind

            if valid_samples == 0:
                best_ind = torch.topk(c[0], 1).indices
                valid_samples = 1

            # KL on stable indices
            x2 = imp[best_ind]
            x1 = p_0[best_ind]
            kl_loss = torch.sum(x2 * (torch.log(x2 + eps) - torch.log(x1 + eps))) + \
                    torch.sum((1 - x2) * (torch.log(1 - x2 + eps) - torch.log(1 - x1 + eps)))

            # regularization on p_0
            reg_p0 = torch.sum((p_0) ** 2)
            reg_loss = self.lambda_reg * reg_p0

            loss_f = (loss / valid_samples) + kl_loss + reg_loss
            loss_f.backward()
            self.opt.step()

            ll_lik.append(cpu(lik_loss.detach()).data.numpy() / valid_samples)
            ll_kl.append(cpu(kl_loss.detach()).data.numpy())
            ll_reg.append(cpu(reg_loss.detach()).data.numpy())
            ll_total.append(cpu(loss_f.detach()).data.numpy())

        return [np.mean(ll_total)], [np.mean(ll_lik)], [np.mean(ll_kl)], [np.mean(ll_reg)]

##################################################################

def main(options):    
    """
    options: A dictionary of hyper-parameters. 
    """
    start_time = time.time()
    ###################
    # Creat a folder to store figures.
    fig_location = os.path.join(options['target'],'figures')
    if os.path.exists(fig_location):
            shutil.rmtree(fig_location)
    os.mkdir(fig_location)
    ###################
    try:
        names = list(pd.read_table(options['z'],  sep=' ', header=None).to_numpy()[:,0])
        options['names'] = names
        Z  = gpu_t(pd.read_table(options['z'],  sep=' ', header=None).to_numpy()[:,1].astype(float))
        if torch.max(Z)==torch.inf:
            print('Z vector has inf as an element, converting it to 200')
            Z[torch.where(Z==torch.inf)[0]] = 200
            
        LD = gpu_t(pd.read_table(options['LD'], sep=' ', header=None).to_numpy())

        if options['annot_given']:
            v  = gpu_t(pd.read_table(options['annot'],  sep=None, engine = 'python', header=None).to_numpy()[:,1:].astype(float))
            #v  = gpu_t(pd.read_table(options['annot'],  sep=' ', header=None).to_numpy()[:,1:].astype(float))

            if torch.max(v)==torch.inf:
                print('v vector has inf as an element, converting it to 200')
                v[torch.where(v==torch.inf)[0]] = 200
            print(f"v shape {v.shape}")

    except BaseException as be:
        print(be)
        return
    
    print(f"Annot given? {options['annot_given']}")

    if LD.size()[0]!=LD.size()[1]:
        return print('\n LD is not a square matrix')
    if LD.size()[0]!=Z.size()[0]:
        return print("\n Dimension of Z and dimension of LD are not same. Dim of Z = {}, dim of LD = {}".format(list(Z.size()), list(LD.size()) ))
    if options['annot_given']:
        if Z.size()[0]!=v.size()[0]:
            return print("\n Dimension of Z and dimension of v are not same. Dim of Z = {}, dim of v = {}".format(list(Z.size()), list(v.size()) ))
    

    bp  = len(Z) # Number of variants..
    LD = regularize_ld(LD) 
    
    n_sub    = gpu_ts(options['n_sub'])
    
    if len(options['loc_true'])!=0:
        loc      = options['loc_true']
    else:
        loc = []
    

    ## Hyperparamters
    n_samples = options['MCMC_samples']
    sigma_sq =  options['sigma_sq']
    n_epochs = options['max_iter']
    temp_lower_bound = gpu_ts(options['temp_lower_bound'])
    K_C = min(bp,options['sparsity_cl'])
    gamma_sp = options['gamma']
    num_iter = 1
    prior_neural_network_layers = options['prior_neural_network']
    lambda_reg = options['prior_regularisation']
    lambda_l1 = options.get('lambda_l1', 0.01)  # L1 penalty for LassoNet
    hierarchy_M = options.get('hierarchy_M', 10.0)  # Hierarchy constraint multiplier

    model = gpu(network(len(Z), [1]+options['NN'], 3, LD, Z))

    if options['annot_given']:
        # Use LassoNet prior for feature selection
        prior_network = gpu(LassoNetPrior(
            m=v.shape[1],
            hidden_dims=prior_neural_network_layers,
            M=hierarchy_M
        ))
        print(f"Using LassoNet prior with {v.shape[1]} annotations, hidden_dims={prior_neural_network_layers}, M={hierarchy_M}, lambda_l1={lambda_l1}")

        # Load pre-trained prior weights if provided
        if options['prior_weights'] and os.path.exists(options['prior_weights']):
            print(f"Loading prior network weights from: {options['prior_weights']}")
            checkpoint = torch.load(options['prior_weights'], map_location='cpu')

            # Check architecture compatibility
            if checkpoint.get('n_annotations') != v.shape[1]:
                raise ValueError(f"Annotation dimension mismatch: weights expect {checkpoint.get('n_annotations')} annotations, but data has {v.shape[1]}")
            if checkpoint.get('hidden_dims') != prior_neural_network_layers:
                raise ValueError(f"Hidden dims mismatch: weights expect {checkpoint.get('hidden_dims')}, but config has {prior_neural_network_layers}")

            prior_network.load_state_dict(checkpoint['state_dict'])
            print("Prior network weights loaded successfully.")

        params = list(model.parameters()) + list(prior_network.parameters())
    else:
        try:
            prior_loc = options['prior_location']
            p_0 = gpu_t(pd.read_table(prior_loc,  sep=' ', header=None).to_numpy()[:,1].astype(float))
        except:
            p_0 = gpu_t(np.array([1/len(Z)]*len(Z)))
        params = list(model.parameters())

    opt_j = optim.Adam(params, lr=0.002, betas=(0.9, 0.999), weight_decay=0)
    scheduler_j = torch.optim.lr_scheduler.StepLR(opt_j, step_size=1000, gamma=0.5)

    if options['annot_given']:
        # Use LassoNet finemapper with L1 penalty and hierarchy constraint
        F_map = finemapper_lassonet(model, prior_network, opt_j, scheduler_j, lambda_l1, lambda_reg)
    else:
        F_map = finemapper(model, opt_j, scheduler_j)
    F_map.S = torch.matmul(torch.inverse(LD),Z.unsqueeze(1))
    F_map.logdetLD = torch.logdet(LD)

    Loss = []
    Loss_lik=[]
    Loss_kl=[]
    Loss_l1 = []  # L1 penalty loss for LassoNet
    Loss_reg = []
    memo ={}

    pip = np.zeros(len(Z))
    for n in tqdm(range(n_epochs+1)):
        temp  = torch.max(temp_lower_bound,gpu_ts(np.exp(-0.0001*n)))
        if options['annot_given']:
            # LassoNet returns 5 values: total, lik, kl, l1, reg
            ll, ll_lik, ll_kl, ll_l1, ll_reg = F_map.train(Z, LD, v, temp, n_samples, sigma_sq,\
                                       n_sub, num_iter, memo, n, K_C, gamma_sp)
        else:
            ll, ll_lik,ll_kl = F_map.train(Z, LD, temp, n_samples, sigma_sq,\
                                       n_sub, p_0, num_iter, memo, n, K_C, gamma_sp)
        F_map.scheduler.step()
        Loss.extend(ll)
        Loss_lik.extend(ll_lik)
        Loss_kl.extend(ll_kl)
        if options['annot_given']:
            Loss_l1.extend(ll_l1)
            Loss_reg.extend(ll_reg)


        if n==n_epochs:
            if options['annot_given']:
                _, _, _, p_0 = prior_network(v, temp, 1)
            mean_memo = reformat_memo(memo, p_0)
            if options['annot_given']:
                # Include feature importance from LassoNet
                feature_importance = prior_network.feature_importance
                res_to_save={
                    'loss': Loss,
                    'lik_loss': Loss_lik,
                    'kl_loss': Loss_kl,
                    'l1_loss': Loss_l1,
                    'reg_loss': Loss_reg,
                    'imp': F_map.model.imp,
                    'loc': loc,
                    'pip': pip,
                    'memo': memo,
                    'mean_memo': mean_memo,
                    'feature_importance': feature_importance
                }
            else:
                res_to_save={'loss':Loss,'lik_loss':Loss_lik,'kl_loss':Loss_kl, 'imp':F_map.model.imp,'loc':loc, 'pip':pip,'memo':memo, 'mean_memo':mean_memo}
            pip = calculate_pip(memo, bp)
            save_object(res_to_save, os.path.join(options['target'],'res'))


        if (n==(n_epochs//2) and n>0 and options['plot_loss'])or (n==n_epochs):
            real = np.zeros(len(Z))
            if len(options['loc_true'])!=0:
                real[loc] = 1
                plt.stem(real, linefmt='r-', markerfmt='ro')
            plt.stem(real, linefmt='r-', markerfmt='ro')
            plt.stem(pip)
            plt.xlabel('variants')
            plt.ylabel('PIP')
            plt.savefig(os.path.join(fig_location, 'pip.pdf'))
            plt.close()

            plt.plot(Loss)
            plt.xlabel('epochs')
            plt.ylabel('Total Loss')
            plt.title('total loss')
            plt.savefig(os.path.join(fig_location,'total_loss.pdf'))
            plt.close()

            plt.plot(Loss_lik)
            plt.xlabel('epochs')
            plt.ylabel('Likelihood Loss')
            plt.title('lik loss')
            plt.savefig(os.path.join(fig_location,'lik_loss.pdf'))
            plt.close()

            plt.plot(Loss_kl)
            plt.title('kl loss')
            plt.xlabel('epochs')
            plt.ylabel('KL Regularization Loss')
            plt.savefig(os.path.join(fig_location,'kl_loss.pdf'))
            plt.close()

            if options['annot_given']:
                # Plot L1 loss
                plt.plot(Loss_l1)
                plt.title('L1 Sparsity Loss (LassoNet)')
                plt.xlabel('epochs')
                plt.ylabel('L1 Loss')
                plt.savefig(os.path.join(fig_location,'l1_loss.pdf'))
                plt.close()

                plt.plot(Loss_reg)
                plt.title('reg loss')
                plt.xlabel('epochs')
                plt.ylabel('Regularization Loss')
                plt.savefig(os.path.join(fig_location,'reg_loss.pdf'))
                plt.close()

                # Plot feature importance from LassoNet
                if prior_network.feature_importance is not None:
                    feat_imp = prior_network.feature_importance
                    plt.figure(figsize=(12, 4))
                    plt.bar(range(len(feat_imp)), feat_imp)
                    plt.xlabel('Annotation Index')
                    plt.ylabel('Feature Importance (|θ|)')
                    plt.title('LassoNet Feature Importance')
                    plt.savefig(os.path.join(fig_location,'feature_importance.pdf'))
                    plt.close()

            real = np.zeros(len(Z))
            if len(options['loc_true'])!=0:
                real[loc] = 1
                plt.stem(real, linefmt='r-', markerfmt='ro')
            plt.stem(F_map.model.imp)
            plt.savefig(os.path.join(fig_location,'binary_concrete_prob.pdf'))
            plt.close()

    # Save prior network weights if requested
    if options['annot_given'] and options['return_weights']:
        weights_path = os.path.join(options['target'], 'prior_network_weights.pt')
        checkpoint = {
            'state_dict': prior_network.state_dict(),
            'n_annotations': v.shape[1],
            'hidden_dims': prior_neural_network_layers,
            'hierarchy_M': hierarchy_M,
            'feature_importance': prior_network.feature_importance
        }
        torch.save(checkpoint, weights_path)
        print(f"Prior network weights saved to: {weights_path}")

    # Save feature importance to CSV
    if options['annot_given'] and prior_network.feature_importance is not None:
        feat_imp_df = pd.DataFrame({
            'annotation_index': range(len(prior_network.feature_importance)),
            'importance': prior_network.feature_importance
        })
        feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)
        feat_imp_df.to_csv(os.path.join(options['target'], 'feature_importance.csv'), index=False)
        print(f"Feature importance saved to: {os.path.join(options['target'], 'feature_importance.csv')}")

        # Print top 10 important features
        print("\nTop 10 most important annotations:")
        print(feat_imp_df.head(10).to_string(index=False))

    if options['get_cred']:
        gen_cred.main(options) 
    else:
        df = {'variant_index':list(range(bp)),'pip':pip, 'variant_names':names}
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(options['target'],'pip.csv'), index=False)
    
    finish_time = time.time()
    
    f = open(os.path.join(options['target'],'time'),'w')
    f.write(str(finish_time-start_time))
    f.close()