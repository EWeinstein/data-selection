import argparse
import configparser
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import MultiStepLR


class Jorganizer(nn.Module):
    """Construct structured J matrix."""

    def __init__(self, N):
        super().__init__()
        self.N = N
        self._make_transfer()

    def _make_transfer(self):
        N = self.N
        self.register_buffer("J_transf", torch.zeros(N*N, dtype=torch.long))
        pos = 0
        offset1 = int(N * (N - 1) / 2)
        offset2 = int(N * (N + 1) / 2)
        for j in range(N-1):
            for jp in range(j+1, N):
                self.J_transf[j * N + jp] = pos
                self.J_transf[jp * N + j] = offset2 + pos
                pos += 1
        for j in range(N):
            self.J_transf[j * N + j] = offset1 + j

    def forward(self, Jraw):
        diag = torch.zeros((self.N, 2, 2))
        flip = torch.transpose(Jraw, 1, 2)
        concat = torch.cat([Jraw, diag, flip])
        reorder = concat[self.J_transf, :, :]
        Jmat = reorder.reshape([self.N, self.N, 2, 2])
        return Jmat


class SteinGlass(TorchDistribution):
    """
    Continuous glass model with a normalized kernelized Stein discrepancy
    instead of a log probability.
    """
    arg_constraints = {}

    def __init__(self, gene_scale, gene_mean, glass_h, glass_J,
                 on_mean, on_scale, off_scale, onoff_shapes=True,
                 kernel_c=1., kernel_beta=-0.5, kernel_l=-0.5,
                 nksd_T=1., validate_args=None):

        self.gene_scale = gene_scale
        self.gene_mean = gene_mean
        self.ngenes = glass_h.shape[0]
        self.glass_h = glass_h
        self.glass_J = glass_J
        self.on_mean = on_mean
        self.on_scale = on_scale
        self.off_scale = off_scale
        self.onoff_shapes = onoff_shapes
        self.kernel_c = kernel_c
        self.kernel_beta = torch.tensor(kernel_beta)
        self.kernel_l = kernel_l
        self.nksd_T = nksd_T

        super(SteinGlass, self).__init__(
            torch.Size([1]), torch.Size([self.ngenes]),
            validate_args=validate_args
        )

    def stein_score(self, value):
        """Stein score function for model."""
        # Compute inverse logits (effective spins).
        invlogit = 1/(1 + torch.exp(-self.gene_scale * (
                                value - self.gene_mean)))
        sigma = torch.cat([(1-invlogit.unsqueeze(-1)), invlogit.unsqueeze(-1)],
                          -1)
        # Compute derivative of inverse logits.
        dinvlogit = self.gene_scale * torch.prod(sigma, -1)
        dsigma = torch.cat([-dinvlogit.unsqueeze(-1), dinvlogit.unsqueeze(-1)],
                           -1)
        # Order-one term.
        term_one = torch.einsum('ijk,jk->ij', dsigma, self.glass_h)
        # Order-two term
        term_two = torch.einsum('ijk,jakb,iab->ij', dsigma, self.glass_J,
                                sigma)
        if self.onoff_shapes:
            onterm = (- dinvlogit * (value - self.on_mean)**2
                      - invlogit * 2 * (value - self.on_mean)
                      ) / (2 * self.on_scale**2)
            offterm = (dinvlogit * value**2 - (1 - invlogit) * 2 * value
                       ) / (2 * self.off_scale**2)
            return term_one + term_two + onterm + offterm
        else:
            return term_one + term_two

    def kernel_terms(self, value):
        """Compute kernel terms in NKSD."""
        c, beta = self.kernel_c, self.kernel_beta / self.ngenes
        xmy = value[:, None, :] - value[None, :, :]
        fl2 = xmy**2
        K = torch.prod((c*c + fl2)**beta, dim=2)
        K = K - torch.diag(torch.diag(K))

        # First derivative.
        Kp = 2 * beta * xmy * (1/(c*c + fl2)) * K[:, :, None]
        # Trace of second derivative.
        Kpp = torch.sum(- 2 * beta / (c*c + fl2)
                        - 4 * beta * (beta - 1) * fl2 / (c*c + fl2)**2,
                        dim=2) * K
        # Normalization.
        Kbar = torch.sum(K)

        # Domain-limited correction.
        if self.kernel_l is not None:
            xl = value[:, None, :] - self.kernel_l
            yl = value[None, :, :] - self.kernel_l
            l_corr = torch.prod(xl * yl, dim=2)
            l_corr = l_corr - torch.diag(torch.diag(l_corr))
            Kl = K * l_corr
            Kpl = Kp * l_corr[:, :, None] + Kl[:, :, None] / xl
            Kppl = (Kpp + torch.sum(Kp / yl, dim=2)
                    + torch.sum(- Kp / xl, dim=2)
                    + torch.sum(1 / (xl * yl), dim=2) * K) * l_corr
            Kbarl = torch.sum(Kl)
            return Kl, Kpl, Kppl, Kbarl
        else:
            return K, Kp, Kpp, Kbar

    def log_prob(self, value):
        """
        The normalized kernelized Stein discrepancy acts as a generalized
        log likelihood.
        """
        # Stein score.
        sscore = self.stein_score(value)

        # Kernel terms.
        K, Kp, Kpp, Kbar = self.kernel_terms(value)

        # Kernelized Stein discrepancy.
        ksd = (torch.einsum('ia,ja,ij->', sscore, sscore, K) +
               2 * torch.einsum('ija,ja->', Kp, sscore) +
               torch.sum(Kpp))
        # Normalized KSD.
        nksd = ksd / Kbar

        # Negative loss.
        nloss = - (value.shape[0]/self.nksd_T) * nksd

        return nloss


class DataSelector:
    def __init__(self, datapoints, ngenes, PY_mix_d=2, PY_conc=-0.25,
                 loorf_init=0.99, learning_rate=0.01,
                 milestones=[], learning_gamma=1.):

        # Set up SVC volume correction.
        self.PY_mix_d = torch.tensor(PY_mix_d)
        self.PY_conc = torch.tensor(PY_conc)
        self.PY_alpha = torch.tensor(0.5)
        assert PY_conc > -self.PY_alpha, (
                'Pitman-Yor model constraint violated.')

        # Compute Pitman-Yor mixture model effective dimension prefactor.
        cB = (self.PY_mix_d / self.PY_alpha) * torch.exp(
                torch.lgamma(self.PY_conc + 1.) -
                torch.lgamma(self.PY_conc + self.PY_alpha))
        self.cfactor = (0.5 * cB * np.sqrt(datapoints)
                        * (np.log(2 * np.pi) - np.log(datapoints)))

        # Set up stochastic optimization of SVC.
        self.select_phi = (loorf_init * torch.ones(ngenes)
                           ).requires_grad_(True)
        self.loorf_optimizer = Adam([self.select_phi], lr=learning_rate)
        self.loorf_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.loorf_optimizer, milestones, gamma=learning_gamma)

    def correction(self, select):
        """SVC volume correction."""
        return (1 - select).sum(-1) * self.cfactor

    def _logit(self):
        """Logit of stochastic selection weights."""
        return 1/(1 + torch.exp(-self.select_phi))

    def sample_select(self):
        """Sample selection variable."""
        probs = self._logit()
        bern = torch.distributions.bernoulli.Bernoulli(probs)
        return bern.sample().to(torch.bool)

    def step(self, selects, nelbos):
        """Update stochastic selection biases with LOORF estimator."""
        selects = selects.to(torch.double)
        # Compute total SVC.
        fb = self.correction(selects) - nelbos
        # LOORF gradient estimator.
        n = len(fb)
        baseline = fb.mean()
        grad_est = (1/(n-1)) * torch.sum((fb - baseline)[:, None] * (
                                         selects - self._logit()[None, :]),
                                         dim=0)
        # Update.
        self.select_phi.grad = -grad_est
        self.loorf_optimizer.step()
        self.loorf_scheduler.step()

        return baseline


class SparseGlass(nn.Module):
    """
    Stein glass with sparsity promoting prior on interactions.
    """

    def __init__(self, ngenes,
                 prior_gene_scale_mn=0., prior_gene_scale_sd=1.,
                 prior_gene_scale_lbound=1.,
                 prior_gene_mean_mn=1., prior_gene_mean_sd=0.,
                 prior_glass_h_sd=1., prior_glass_J_scale=0.1,
                 prior_on_mean_mn=0., prior_on_mean_sd=0.,
                 prior_on_scale_mn=0., prior_on_scale_sd=0.,
                 prior_off_scale_mn=0., prior_off_scale_sd=0.,
                 onoff_shapes=False,
                 kernel_c=1., kernel_beta=-0.5, kernel_l=-0.5, nksd_T=1.,
                 cuda=False, pin_memory=False):
        super().__init__()
        self.ngenes = ngenes
        self.ninteracts = int(self.ngenes * (self.ngenes - 1) / 2)
        self.prior_gene_scale_mn = prior_gene_scale_mn
        self.prior_gene_scale_sd = prior_gene_scale_sd
        self.prior_gene_scale_lbound = prior_gene_scale_lbound
        self.prior_gene_mean_mn = prior_gene_mean_mn
        self.prior_gene_mean_sd = prior_gene_mean_sd
        self.prior_glass_h_sd = prior_glass_h_sd
        self.prior_glass_J_scale = prior_glass_J_scale
        self.prior_on_mean_mn = prior_on_mean_mn
        self.prior_on_mean_sd = prior_on_mean_sd
        self.prior_on_scale_mn = prior_on_scale_mn
        self.prior_on_scale_sd = prior_on_scale_sd
        self.prior_off_scale_mn = prior_off_scale_mn
        self.prior_off_scale_sd = prior_off_scale_sd
        self.onoff_shapes = onoff_shapes
        self.kernel_c = kernel_c
        self.kernel_beta = kernel_beta
        self.kernel_l = kernel_l
        self.nksd_T = nksd_T
        self.cuda = cuda
        self.pin_memory = pin_memory
        self.jorganizer = Jorganizer(ngenes)

    def model(self, data, select, local_scale):
        # Effective spin parameters.
        gene_scale = pyro.sample(
            "gene_scale",
            dist.Normal(torch.tensor(self.prior_gene_scale_mn),
                        torch.tensor(self.prior_gene_scale_sd)))
        gene_scale = softplus(gene_scale) + self.prior_gene_scale_lbound
        gene_mean = pyro.sample(
            "gene_mean",
            dist.Normal(torch.tensor(self.prior_gene_mean_mn),
                        torch.tensor(self.prior_gene_mean_sd)))
        gene_mean = softplus(gene_mean)
        # First order energies.
        glass_h = pyro.sample(
            "glass_h",
            dist.Normal(torch.zeros((self.ngenes, 2)),
                        torch.tensor(self.prior_glass_h_sd)).to_event(2)
        )
        # Second order energies.
        glass_J = pyro.sample(
            "glass_J",
            dist.Laplace(torch.zeros((self.ninteracts, 2, 2)),
                         torch.tensor(self.prior_glass_J_scale)).to_event(3)
        )
        glass_J = self.jorganizer.forward(glass_J)

        # Mode shapes.
        if self.onoff_shapes:
            on_mean = pyro.sample(
                "on_mean",
                dist.Normal(torch.tensor(self.prior_on_mean_mn)
                            * torch.ones((self.ngenes,)),
                            torch.tensor(self.prior_on_mean_sd)).to_event(1)
            )
            on_mean = softplus(on_mean) + gene_mean
            on_mean = on_mean[select]
            on_scale = pyro.sample(
                "on_scale",
                dist.Normal(torch.tensor(self.prior_on_scale_mn)
                            * torch.ones((self.ngenes,)),
                            torch.tensor(self.prior_on_scale_sd)).to_event(1)
            )
            on_scale = softplus(on_scale)
            on_scale = on_scale[select]
            off_scale = pyro.sample(
                "off_scale",
                dist.Normal(torch.tensor(self.prior_off_scale_mn),
                            torch.tensor(self.prior_off_scale_sd))
            )
            off_scale = softplus(off_scale)
        else:
            on_mean, on_scale = torch.tensor(1.), torch.tensor(1.)
            off_scale = torch.tensor(1.)

        # Take selected subset.
        data = data[:, select]
        glass_h = glass_h[select]
        glass_J = glass_J[select][:, select]

        # Compute NKSD term.
        with pyro.plate("batch", data.shape[0]):
            with poutine.scale(scale=local_scale):
                # Observations.
                pyro.sample(
                    "obs_seq",
                    SteinGlass(
                        gene_scale, gene_mean, glass_h, glass_J,
                        on_mean, on_scale, off_scale,
                        onoff_shapes=self.onoff_shapes,
                        kernel_c=self.kernel_c, kernel_beta=self.kernel_beta,
                        kernel_l=self.kernel_l, nksd_T=self.nksd_T
                    ),
                    obs=data,
                )

    def guide(self, data, select, local_scale):
        gene_scale_mn = pyro.param(
            "gene_scale_mn", torch.tensor(0.)
        )
        gene_scale_sd = pyro.param(
            "gene_scale_sd", torch.tensor(0.)
        )
        pyro.sample("gene_scale",
                    dist.Normal(gene_scale_mn,
                                softplus(gene_scale_sd)))
        gene_mean_mn = pyro.param(
            "gene_mean_mn", torch.tensor(0.)
        )
        gene_mean_sd = pyro.param(
            "gene_mean_sd", torch.tensor(0.)
        )
        pyro.sample("gene_mean",
                    dist.Normal(gene_mean_mn,
                                softplus(gene_mean_sd)))
        glass_h_mn = pyro.param(
            "glass_h_mn", torch.zeros((self.ngenes, 2))
        )
        glass_h_sd = pyro.param(
            "glass_h_sd", torch.zeros((self.ngenes, 2))
        )
        pyro.sample("glass_h",
                    dist.Normal(glass_h_mn,
                                softplus(glass_h_sd)).to_event(2))
        glass_J_mn = pyro.param(
            "glass_J_mn", torch.zeros((self.ninteracts, 2, 2))
        )
        glass_J_sd = pyro.param(
            "glass_J_sd", torch.zeros((self.ninteracts, 2, 2))
        )
        pyro.sample("glass_J",
                    dist.Laplace(glass_J_mn,
                                 softplus(glass_J_sd)).to_event(3))
        if self.onoff_shapes:
            on_mean_mn = pyro.param(
                "on_mean_mn", torch.zeros((self.ngenes,))
            )
            on_mean_sd = pyro.param(
                "on_mean_sd", torch.zeros((self.ngenes,))
            )
            pyro.sample("on_mean",
                        dist.Normal(on_mean_mn,
                                    softplus(on_mean_sd)).to_event(1))
            on_scale_mn = pyro.param(
                "on_scale_mn", torch.zeros((self.ngenes,))
            )
            on_scale_sd = pyro.param(
                "on_scale_sd", torch.zeros((self.ngenes,))
            )
            pyro.sample("on_scale",
                        dist.Normal(on_scale_mn,
                                    softplus(on_scale_sd)).to_event(1))
            off_scale_mn = pyro.param(
                "off_scale_mn", torch.tensor(0.)
            )
            off_scale_sd = pyro.param(
                "off_scale_sd", torch.tensor(0.)
            )
            pyro.sample("off_scale",
                        dist.Normal(off_scale_mn,
                                    softplus(off_scale_sd)))

    def fit_svi(self, dataset, epochs=2, batch_size=100, scheduler=None,
                jit=False, learning_rate=0.01,
                milestones=[], learning_gamma=1., PY_mix_d=2, PY_conc=-0.25,
                loorf_samples=10, loorf_init=1.0, select_all=False,
                early_stop=True, smooth_wind=2000):
        """Fit via stochastic variational inference."""
        # GPU.
        if self.cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # Initialize guide.
        self.guide(None, None, None)
        dataload = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            generator=torch.Generator(device=device),
        )
        N = len(dataset)
        # Optimizer for model variational approximation.
        scheduler = MultiStepLR(
            {
                "optimizer": Adam,
                "optim_args": {"lr": learning_rate},
                "milestones": milestones,
                "gamma": learning_gamma,
            }
        )
        if not select_all:
            # Optimizer for data selection.
            dataselector = DataSelector(
                          N, self.ngenes, PY_mix_d=PY_mix_d, PY_conc=PY_conc,
                          loorf_init=loorf_init, learning_rate=learning_rate,
                          milestones=milestones, learning_gamma=learning_gamma)
        else:
            select = torch.ones(self.ngenes).to(torch.bool)
            select_prob = torch.ones(self.ngenes)
            select_gap = torch.tensor(0.5)

        # Setup stochastic variational inference.
        if jit:
            elbo = JitTrace_ELBO(ignore_jit_warnings=True)
        else:
            elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, scheduler, loss=elbo)

        # Run inference.
        svcs = []
        select_gaps = []
        t0 = datetime.now()
        stop = False
        for epoch in range(epochs):
            for data in dataload:
                data = data[0]
                if self.cuda:
                    data = data.cuda()

                # Take SVI step.
                if not select_all:
                    select = dataselector.sample_select()
                nelbo = svi.step(data, select, torch.tensor(N / data.shape[0]))
                scheduler.step()

                # Draw LOORF samples.
                if not select_all:
                    selects = torch.zeros((loorf_samples, self.ngenes),
                                          dtype=torch.bool)
                    nelbos = torch.zeros(loorf_samples)
                    for s in range(loorf_samples):
                        selects[s] = dataselector.sample_select()
                        nelbos[s] = svi.evaluate_loss(
                                data, selects[s],
                                torch.tensor(N / data.shape[0]))
                    # Update stochastic selection.
                    svc = dataselector.step(selects, nelbos)
                    select_prob = dataselector._logit().detach()
                    select_gap = torch.abs(select_prob - 0.5).min()
                else:
                    svc = torch.tensor(-nelbo)

                # Record.
                svcs.append(svc.cpu())
                select_gaps.append(select_gap.cpu())
                if early_stop and len(svcs) > 2 * smooth_wind:
                    mu_m_var_0 = (np.mean(svcs[-smooth_wind:])
                                  - np.std(svcs[-smooth_wind:]))
                    mu_m_var_1 = (np.mean(svcs[-2*smooth_wind:-smooth_wind])
                                  - np.std(svcs[-2*smooth_wind:-smooth_wind]))
                    if mu_m_var_0 < mu_m_var_1:
                        stop = True
                        break
            print(epoch, svc, select_gap, torch.sum(select_prob > 0.5),
                  " ", datetime.now() - t0)
            if stop:
                print('Stopped early based on mean - std criterion.')
                break
        return np.array(svcs), np.array(select_gaps), select_prob.cpu().numpy()


def simulate_data(small, device):
    """Generate example dataset."""
    if small:
        N = 10
    else:
        N = 10000
    # Sample discrete variable that determines whether expression is on or off.
    z = torch.distributions.bernoulli.Bernoulli(
                torch.tensor([0.2, 0.5, 0.8])).sample((N,))
    # Induce negative correlation between dimensions 0 and 3, and 1 and 4
    z = torch.cat([z, 1. - z[:, :1], 1. - z[:, 1:2]], dim=-1)
    # Draw observations.
    x = torch.randn((N, 5)) * 0.2 + (1 + torch.randn((N, 5)) * 0.2) * z
    # Introduce additional contamination in dimension 3.
    x[:, 2] = torch.randn(N) * 0.001 + 0.5
    # Gene names.
    gene_names = ['gene_0_in', 'gene_1_in', 'gene_2_out', 'gene_3_in',
                  'gene_4_in']
    return TensorDataset(x), gene_names


def load_data(config, device):
    """Load (preprocessed) data from file."""
    with open(config['data']['file'], 'rb') as dr:
        X = pickle.load(dr)
        gene_names = pickle.load(dr)

    return TensorDataset(torch.tensor(X, device=device)), gene_names


def main(config):
    # Load dataset.
    cpu_data = config['general']['cpu_data'] == 'True'
    cuda = config['general']['cuda'] == 'True'
    if cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_dtype(torch.float64)
    if cpu_data or not cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    pin_memory = config['general']['pin_memory'] == 'True'

    # Training seed.
    pyro.set_rng_seed(int(config['general']['rng_seed']))

    # Load data.
    simulate = config['general']['simulate'] == 'True'
    if simulate:
        small = config['general']['small'] == 'True'
        dataset, gene_names = simulate_data(small, device)
    else:
        dataset, gene_names = load_data(config, device)
    ngenes = len(gene_names)

    # Construct model
    if config['nksd']['kernel_l'] == 'None':
        kernel_l = None
    else:
        kernel_l = float(config['nksd']['kernel_l'])
    onoff_shapes = config['model']['onoff_shapes'] == 'True'
    model = SparseGlass(
            ngenes,
            prior_gene_scale_mn=float(config['model']['prior_gene_scale_mn']),
            prior_gene_scale_sd=float(config['model']['prior_gene_scale_sd']),
            prior_gene_scale_lbound=float(config['model'][
                                            'prior_gene_scale_lbound']),
            prior_gene_mean_mn=float(config['model']['prior_gene_mean_mn']),
            prior_gene_mean_sd=float(config['model']['prior_gene_mean_sd']),
            prior_glass_h_sd=float(config['model']['prior_glass_h_sd']),
            prior_glass_J_scale=float(config['model']['prior_glass_J_scale']),
            prior_on_mean_mn=float(config['model']['prior_on_mean_mn']),
            prior_on_mean_sd=float(config['model']['prior_on_mean_sd']),
            prior_on_scale_mn=float(config['model']['prior_on_scale_mn']),
            prior_on_scale_sd=float(config['model']['prior_on_scale_sd']),
            prior_off_scale_mn=float(config['model']['prior_off_scale_mn']),
            prior_off_scale_sd=float(config['model']['prior_off_scale_sd']),
            onoff_shapes=onoff_shapes,
            kernel_c=float(config['nksd']['kernel_c']),
            kernel_beta=float(config['nksd']['kernel_beta']),
            kernel_l=kernel_l,
            nksd_T=float(config['nksd']['nksd_T']),
            cuda=cuda, pin_memory=pin_memory)

    # Infer with SVI and LOORF.
    PY_mix_d = float(config['svc']['PY_mix_d'])
    PY_conc = float(config['svc']['PY_conc'])
    learning_rate = float(config['train']['learning_rate'])
    milestones = json.loads(config['train']['milestones'])
    learning_gamma = float(config['train']['learning_gamma'])
    n_epochs = int(config['train']['n_epochs'])
    loorf_samples = int(config['train']['loorf_samples'])
    loorf_init = float(config['train']['loorf_init'])
    select_all = config['train']['select_all'] == 'True'
    jit = config['train']['jit'] == 'True'
    batch_size = min([len(dataset), int(config['train']['batch_size'])])
    early_stop = config['train']['early_stop'] == 'True'
    smooth_wind = int(config['train']['smooth_wind'])
    svcs, select_gaps, select_prob = model.fit_svi(
                           dataset, epochs=n_epochs, batch_size=batch_size,
                           jit=jit, learning_rate=learning_rate,
                           milestones=milestones,
                           learning_gamma=learning_gamma,
                           PY_mix_d=PY_mix_d, PY_conc=PY_conc,
                           loorf_samples=loorf_samples, loorf_init=loorf_init,
                           select_all=select_all, early_stop=early_stop,
                           smooth_wind=smooth_wind)

    # Plot and save.
    if config['general']['save'] == 'True':
        out_folder = config['general']['out_folder']
        if config['general']['make_subfolder'] == 'True':
            time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_folder = os.path.join(out_folder, time_stamp)
            os.mkdir(out_folder)
        # Training curve.
        plt.figure(figsize=(8, 6))
        plt.plot(svcs)
        plt.xlabel('iteration', fontsize=18)
        plt.ylabel(r'Estimated SVC', fontsize=18)
        plt.savefig(os.path.join(out_folder, 'SVC.pdf'))
        plt.figure(figsize=(8, 6))

        svc_mns = np.array([np.mean(svcs[np.maximum(0, i - smooth_wind):i])
                            for i in range(1, len(svcs))])
        svc_sds = np.array([np.std(svcs[np.maximum(0, i - smooth_wind):i])
                            for i in range(1, len(svcs))])
        plt.plot(svc_mns, 'b-', label=r'$\mu$')
        plt.plot(svc_mns - svc_sds, 'b:', label=r'$\mu - \sigma$')
        plt.legend(fontsize=16)
        plt.xlabel('iteration', fontsize=18)
        plt.ylabel(r'SVC', fontsize=18)
        plt.savefig(os.path.join(out_folder, 'SVC_mn_sd.pdf'))

        # Data selection results.
        plt.figure(figsize=(8, 6))
        plt.plot(select_prob, 'o')
        plt.xticks(np.arange(len(gene_names)), gene_names,
                   rotation=-90, fontsize=16)
        plt.ylabel(r'selection probability $\phi$', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, 'selection_prob.pdf'))
        # Plot posterior over h.
        h = pyro.param("glass_h_mn").detach().cpu().numpy()
        h_gap = h[:, 0] - h[:, 1]
        plt.figure(figsize=(8, 6))
        plt.plot(h_gap, 'o')
        plt.ylabel(r'$H_1 - H_2$', fontsize=18)
        plt.xticks(np.arange(len(gene_names)), gene_names,
                   rotation=-90, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, 'glass_h_deltaE.pdf'))

        # Plot posterior over J.
        Jraw = pyro.param("glass_J_mn").detach()
        J = model.jorganizer.forward(Jraw).cpu().numpy()
        interact = (J[:, :, 1, 0] + J[:, :, 0, 1]
                    - J[:, :, 0, 0] - J[:, :, 1, 1])
        plt.figure(figsize=(8, 8))
        clim = np.max(np.abs(interact))
        plt.imshow(-interact, cmap='seismic', vmin=-clim, vmax=clim)
        plt.colorbar()
        plt.title(r'$\Delta E$', fontsize=18)
        plt.xticks(np.arange(len(gene_names)), gene_names,
                   rotation=-90, fontsize=16)
        plt.yticks(np.arange(len(gene_names)), gene_names, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, 'glass_J_deltaE.pdf'))

        # Save config.
        config['results']['gene_mean_mn'] = str(pyro.param(
                "gene_mean_mn").detach().cpu().numpy())
        config['results']['gene_mean_sd'] = str(pyro.param(
                "gene_mean_sd").detach().cpu().numpy())
        config['results']['gene_scale_mn'] = str(pyro.param(
                "gene_scale_mn").detach().cpu().numpy())
        config['results']['gene_scale_sd'] = str(pyro.param(
                "gene_scale_sd").detach().cpu().numpy())
        config['results']['glass_h_mn'] = os.path.join(out_folder,
                                                       'glass_h_mn.npy')
        np.save(config['results']['glass_h_mn'],
                pyro.param("glass_h_mn").detach().cpu().numpy())
        config['results']['glass_h_sd'] = os.path.join(out_folder,
                                                       'glass_h_sd.npy')
        np.save(config['results']['glass_h_sd'],
                pyro.param("glass_h_sd").detach().cpu().numpy())
        config['results']['glass_J_mn'] = os.path.join(out_folder,
                                                       'glass_J_mn.npy')
        np.save(config['results']['glass_J_mn'], J)
        config['results']['glass_J_sd'] = os.path.join(out_folder,
                                                       'glass_J_sd.npy')
        Jsdraw = pyro.param("glass_J_sd").detach()
        Jsd = model.jorganizer.forward(Jsdraw).cpu().numpy()
        np.save(config['results']['glass_J_sd'], Jsd)
        if onoff_shapes:
            config['results']['on_mean_mn'] = os.path.join(out_folder,
                                                           'on_mean_mn.npy')
            np.save(config['results']['on_mean_mn'],
                    pyro.param("on_mean_mn").detach().cpu().numpy())
            config['results']['on_mean_sd'] = os.path.join(out_folder,
                                                           'on_mean_sd.npy')
            np.save(config['results']['on_mean_sd'],
                    pyro.param("on_mean_sd").detach().cpu().numpy())
            config['results']['on_scale_mn'] = os.path.join(out_folder,
                                                            'on_scale_mn.npy')
            np.save(config['results']['on_scale_mn'],
                    pyro.param("on_scale_mn").detach().cpu().numpy())
            config['results']['on_scale_sd'] = os.path.join(out_folder,
                                                            'on_scale_sd.npy')
            np.save(config['results']['on_scale_sd'],
                    pyro.param("on_scale_sd").detach().cpu().numpy())
            config['results']['off_scale_mn'] = str(pyro.param(
                    "off_scale_mn").detach().cpu().numpy())
            config['results']['off_scale_sd'] = str(pyro.param(
                    "off_scale_sd").detach().cpu().numpy())
        config['results']['select_prob'] = os.path.join(out_folder,
                                                        'select_prob.npy')
        np.save(config['results']['select_prob'], select_prob)
        config['results']['svcs'] = os.path.join(out_folder, 'svcs.npy')
        np.save(config['results']['svcs'], svcs)
        with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
            config.write(cw)

        if simulate:
            with open(os.path.join(out_folder, 'simulated_data.pickle'), 'wb'
                      ) as rw:
                pickle.dump(dataset.tensors[0].numpy(), rw)
                pickle.dump(gene_names, rw)


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="SVC data selection on a glass model of scRNA data.")
    parser.add_argument('configPath')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.configPath)

    main(config)
