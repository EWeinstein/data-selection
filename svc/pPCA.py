"""Consistency experiments for PCA with KSD loss."""

import autograd.numpy as np
from autograd import grad, hessian
from autograd.numpy.linalg import slogdet

from pymanopt import Problem
from pymanopt.solvers import TrustRegions
from pymanopt.manifolds import Stiefel, Product, Euclidean

import argparse
import configparser
import pickle
from numpy.linalg import svd, qr, solve
from scipy.linalg import expm
from scipy.stats import multivariate_normal
from scipy.special import gammaln
from matplotlib import pyplot as plt
from datetime import datetime
import pprint
import os


def K_matrices(X, c=1, beta=-0.5):
    """Kernel matrices."""
    xmy = X[:, None, :] - X[None, :, :]
    fl2 = np.power(xmy, 2)
    K = np.prod(np.power(c*c + fl2, beta), axis=2)
    K = K - np.diag(np.diag(K))
    XKX = np.matmul(X.T, np.matmul(K, X))
    Kbar = np.sum(K)

    Kp = np.sum(2 * beta * xmy * np.power(c*c + fl2, -1) * K[:, :, None],
                axis=0)
    XKp = np.matmul(X.T, Kp)

    Kpp = np.sum(np.sum(
           - 2 * beta * np.power(c*c + fl2, -1)
           - 4 * beta * (beta - 1) * fl2 * np.power(c*c + fl2, -2),
           axis=2) * K)

    return XKX, XKp, Kpp, Kbar


def onedrop_K_matrices(X, c=1, beta=-0.5):
    """Kernel matrices for datasets with one dimension dropped."""
    # Final dimension gives K matrices for variable sigma, each with
    # corresponding dimension dropped.
    N, d = X.shape
    sig = 1. - np.eye(d)
    # n x n x d
    xmy = X[:, None, :] - X[None, :, :]
    fl2 = np.power(xmy, 2)
    # n x n x d
    Ki = np.power(c*c + fl2, beta)
    # n x n x d (-1replicas)
    K = np.prod(Ki, axis=2, keepdims=True)/Ki
    # Set diagonal K values to 0.
    diag_screen = 1. - np.eye(X.shape[0])
    K = K * diag_screen[:, :, None]
    # construct Vsigma.
    # d-1 x d (-1replicas)
    sigs = (np.arange(d-1)[:, None] +
            (np.arange(d-1)[:, None] >= np.arange(d)[None, :]))
    # d x (d-1) x d (-1replicas)
    Vsigs = sigs[None, :, :] == np.arange(d)[:, None, None]
    # n x (d-1) x d (-1replicas)
    XV = np.einsum('ij,jlk->ilk', X, Vsigs)
    # (d-1) x (d-1) x d (-1replicas)
    XKX = np.einsum('ijk,ibk->jbk', XV, np.einsum('iak,abk->ibk', K, XV))
    # d (-1replicas)
    Kbar = np.einsum('ijk->k', K)

    # n x n x d x d (-1 replicas)
    Kp = np.einsum('ijk,ija->jka', 2 * beta * xmy * np.power(c*c + fl2, -1),
                   K) * sig[None, :, :]
    # n x n x d (-1 replicas)
    XKp = np.einsum('ijk,ick->jck', XV, np.einsum('ibk,bck->ick', Kp, Vsigs))

    # n x n x d x d (-1 replicas)
    gxy = (- 2 * beta * np.power(c*c + fl2, -1)
           - 4 * beta * (beta - 1) * fl2 * np.power(c*c + fl2, -2))
    Kpp = np.einsum('ka,ka->a', np.einsum('ijk,ija->ka', gxy, K), sig)
    return XKX, XKp, Kpp, Kbar, Vsigs


def get_H(U, L, v):

    return np.matmul(U, np.diag(np.sqrt(L - v)))


def get_Sigma(H, v):

    return np.matmul(H, H.T) + v*np.eye(np.shape(H)[0])


def PCA_MLE(X, k):
    """MLE of the PCA model."""
    # Statistics.
    N, d = np.shape(X)
    S = np.matmul(X.T, X)/X.shape[0]

    # SVD.
    Ufull, Lamfull, R = svd(S)

    # Construct PPCA estimate.
    U = Ufull[:, :k]
    L = Lamfull[:k]
    v = np.mean(Lamfull[k:])

    return U, L, v, Lamfull


def PCA_volumeln(N, Lamfull, v, k, alpha):
    """Prior volume term for PCA model."""
    lamhat = v * np.ones(len(Lamfull))
    lamhat[:k] = Lamfull[:k]
    d = len(Lamfull)
    AZln = np.sum([np.sum([np.log(N) + np.log((1/lamhat[jj] - 1/lamhat[ii]) *
                                              (Lamfull[ii] - Lamfull[jj]))
                           for jj in range(ii+1, d)])
                   for ii in range(k)])
    ALln = k * np.log((N - 1 + alpha)/2)
    Avln = np.log(((N + 1 + alpha) * (d - k) - 2)/2)

    return AZln + ALln + Avln


def PCA_MLE_loglike(N, d, k, Lp, vp):
    """Likelihood term for PCA model."""
    return (-N * d * 0.5 * np.log(2 * np.pi)
            - N * 0.5 * np.sum(Lp)
            - N * (d - k) * 0.5 * vp
            - N * d * 0.5)


def PCA_marginal(X, k, alpha):
    """Marginal likelihood of PCA model."""
    N, d = X.shape

    U, L, v, Lamfull = PCA_MLE(X, k)
    Lp = np.log(L)
    vp = np.log(v)
    vol_term = -0.5 * PCA_volumeln(X.shape[0], Lamfull, v, k, alpha)
    pr_term = (U_prior(d, k) + L_prior(Lp, alpha)
               + v_prior(vp, d, k, alpha))
    ml_term = PCA_MLE_loglike(N, d, k, Lp, vp)
    return vol_term + pr_term + ml_term


def PCA_latent_z_posterior(X, U, L, v):
    """Mean posterior over latent variable."""
    H = np.matmul(U, np.diag(np.sqrt(L - v)))
    F = np.matmul(H.T, H) + v * np.eye(H.shape[1])
    zi = solve(F, np.matmul(H.T, X.T)).T

    return zi


def build_Z(uparam, d, k):
    """Building the Z matrix in Eqn. 51 of Minka (2000)."""
    m = int(d*k - k*(k+1)/2)
    indx = np.zeros((d, d)) - 1
    c = -1
    for i in range(k):
        for j in range(i+1, d):
            c += 1
            indx[i, j] = c
    incl = indx[:, :, None] == np.arange(m)[None, None, :]
    upz = np.sum(incl[:, :, :] * uparam[None, None, :], axis=2)
    z = upz - upz.T
    return z


def build_Ud(U):
    """Building the Ud matrix in Eqn. 51 of Minka (2000)."""
    d, k = np.shape(U)
    Fd = np.zeros((d, d))
    Fd[:, :k] = U
    Ud, _ = qr(Fd)
    return Ud


def U_prior(d, k):
    """Prior log likelihood of U, from Minka (2000)."""
    a0 = (d - np.arange(k))/2
    return -k * np.log(2) + np.sum(gammaln(a0) - a0 * np.log(np.pi))


def L_prior(Lp, alpha):
    """Prior log likelihood of L (including Jacobian), from Minka (2000)."""
    # Prior (including jacobian)
    a0 = alpha/2
    a1 = (alpha/2) * np.exp(-Lp)
    return np.sum(- gammaln(a0) + a0 * np.log(a1) - a1)


def v_prior(vp, d, k, alpha):
    """Prior log likelihood of v, from Minka (2000)."""
    a0 = (alpha + 2)*(d - k)/2 - 1
    a1 = alpha * (d - k) * np.exp(-vp) / 2

    return - gammaln(a0) + a0 * np.log(a1) - a1


def Lp_transform(Bp, tp, vmin):
    """Numerically stable transform to get constrained v and L parameters."""
    vp = np.logaddexp(tp, np.log(vmin))
    Lp = np.logaddexp(Bp, vp)

    return Lp, vp


def Bt_LnJacobian(Bp, tp, vmin):
    """Jacobians for transform of Bp and tp."""
    vpmin = np.log(vmin)
    vp = np.logaddexp(tp, vpmin)
    Bdetln = np.sum(Bp - np.logaddexp(Bp, vp))
    tdetln = tp - np.logaddexp(tp, vpmin)
    return Bdetln + tdetln


def construct_cost(X, k, c=1, beta=-0.5, vmin=1e-5, T=1.):
    """Construct the NKSD loss function."""
    # Basics.
    N, d = np.shape(X)
    m = int(d*k - k*(k+1)/2)

    # Compute K matrices.
    XKX, XKp, Kpp, Kbar = K_matrices(X, c=c, beta=beta)

    # Build cost function.
    def cost(params, T=T):
        U = params[0]
        Bp = params[1]
        tp = params[2]
        Lp, vp = Lp_transform(Bp, tp, vmin)

        # Diagonal parameters.
        vinv = np.exp(-vp)
        Lmv = np.diag(np.exp(-Lp) - vinv)

        # KSD construction.
        prefac = N / (T * Kbar)
        term0 = np.trace(
            np.matmul(U.T, np.matmul(XKX, np.matmul(U, Lmv*Lmv))))
        term1 = np.trace(
            np.matmul(U.T, np.matmul(2 * vinv * XKX - 2 * XKp,
                                     np.matmul(U, Lmv))))
        term2 = vinv * np.trace(vinv * XKX - 2 * XKp)
        term3 = Kpp

        return prefac * (term0 + term1 + term2 + term3)

    def laplace_cost(params, U0, T=T):

        Ud = build_Ud(U0)
        uparam = params[:m]
        Z = build_Z(uparam, d, k)
        U = np.matmul(np.matmul(Ud, (
                np.eye(d) + Z + 0.5 * np.matmul(Z, Z) +
                (1/6) * np.matmul(np.matmul(Z, Z), Z))), np.eye(d, k))

        # Base cost.
        cost_0 = cost([U, params[m:(m+k)], params[m+k]], T=T)

        return cost_0

    return cost, laplace_cost


def construct_od_cost(X, U0, d, k, c=1, beta=-0.5, vmin=1e-5, T=1.):
    """Construct the NKSD loss function with individual dimensions dropped."""
    m = int(d*k - k*(k+1)/2)
    N = X.shape[0]

    # Construct new U parameterization.
    Ud = build_Ud(U0)

    # Construct dropped out K matrices
    odXKX, odXKp, odKpp, odKbar, Vsigs = onedrop_K_matrices(
                                                X, c=c, beta=beta)

    # Construct dropped out loss function, with new U parameterization.
    def od_cost(params, od, T=T, perturbed=False):
        uparam = params[:m]
        Z = build_Z(uparam, d, k)
        if not perturbed:
            U = np.matmul(np.matmul(Ud, (
                    np.eye(d) + Z + 0.5 * np.matmul(Z, Z) +
                    (1/6) * np.matmul(np.matmul(Z, Z), Z))), np.eye(d, k))
        else:
            U = np.matmul(Ud, np.matmul(expm(Z), np.eye(d, k)))

        # Diagonal parameters.
        Bp = params[m:(m+k)]
        tp = params[m+k]
        Lp, vp = Lp_transform(Bp, tp, vmin)
        vinv = np.exp(-vp)
        Lmv = np.diag(np.exp(-Lp) - vinv)

        VU = np.einsum('ij,ia->ja', Vsigs[:, :, od], U)

        prefac = N / (T * odKbar[od])
        term0 = np.einsum('bd,db->',
                          np.einsum('ab,ad->bd', VU,
                                    np.einsum('ac,cd->ad', odXKX[:, :, od],
                                              VU)), Lmv*Lmv)
        term1 = np.einsum('bd,db->',
                          np.einsum('ab,ad->bd', VU,
                                    np.einsum('ac,cd->ad',
                                              2 * vinv * odXKX[:, :, od]
                                              - 2 * odXKp[:, :, od],
                                              VU)), Lmv)
        term2 = np.trace(vinv * (vinv * odXKX[:, :, od] - 2 * odXKp[:, :, od]))
        term3 = odKpp[od]
        cost = prefac * (term0 + term1 + term2 + term3)
        return cost

    return od_cost


def laplace_approx(cost0, hess0, params, d, k, alpha, T=1, vmin=1e-5):
    """Compute the Laplace approximation."""
    # Loss.
    lf0 = -cost0

    # Volume.
    det_hess = slogdet(hess0)[1]
    vol = (d/2) * np.log(2 * np.pi) - 0.5 * det_hess

    # Prior (w/jacobian).
    m = int(d*k - k*(k+1)/2)
    Bp = params[m:(m+k)]
    tp = params[m+k]
    Lp, vp = Lp_transform(Bp, tp, vmin)
    pr = (U_prior(d, k) + L_prior(Lp, alpha) + v_prior(vp, d, k, alpha)
          + Bt_LnJacobian(Bp, tp, vmin))

    # Total approximate marginal.
    return lf0 + vol + pr, {'f0': -lf0, 'volume': vol, 'prior': pr,
                            'det': det_hess}


def min_NKSD(X, k, T=1, c=1, beta=-0.5, vmin=1e-5, maxiter=100):
    """Optimize the NKSD using pymanopt."""
    # Basics.
    N, d = np.shape(X)
    m = int(d*k - k*(k+1)/2)

    # Construct optimization cost.
    print('-------- Construct objective function ... --------')
    t0 = datetime.now()
    cost, laplace_cost = construct_cost(X, k, c=c, beta=beta, vmin=vmin, T=T)
    print('... done', datetime.now() - t0)

    # Initialize from MLE
    print('-------- Get MLE for initialization ... --------')
    t0 = datetime.now()
    U_mle, L_mle, v_mle, Lamfull = PCA_MLE(X, k)
    print('... done', datetime.now() - t0)

    # Solve.
    print('-------- Solve pymanopt problem ... --------')
    t0 = datetime.now()
    # Build pymanopt problem.
    solver = TrustRegions(logverbosity=1, maxiter=maxiter)
    manifold = Product([Stiefel(d, k), Euclidean(k), Euclidean(1)])
    problem = Problem(manifold=manifold, cost=cost, verbosity=0)
    # Solve.
    wopt, optlog = solver.solve(
            problem, x=[U_mle, np.log(L_mle), np.log(v_mle)])
    # Print
    pp = pprint.PrettyPrinter()
    pp.pprint(optlog)
    print('MLE:', U_mle, L_mle - v_mle, v_mle)
    # Vectorized and U transformed optimal parameters.
    U0 = wopt[0]
    params0 = np.zeros(m + k + 1)
    params0[m:(m+k)] = wopt[1]
    params0[m+k] = wopt[2]
    cost0 = optlog['final_values']['f(x)']
    print('... done', datetime.now() - t0)

    # Assemble estimators.
    print('-------- Assemble estimators ... --------')
    t0 = datetime.now()
    H_mle_est = get_H(U_mle, L_mle, v_mle)
    Sigma_mle_est = get_Sigma(H_mle_est, v_mle)

    Lp_opt, vp_opt = Lp_transform(wopt[1], wopt[2], vmin)
    H_ksd_est = get_H(wopt[0], np.exp(Lp_opt), np.exp(vp_opt))
    Sigma_ksd_est = get_Sigma(H_ksd_est, np.exp(vp_opt))
    print('... done', datetime.now() - t0)

    return (U0, params0, cost0, cost, laplace_cost,
            Sigma_ksd_est, Sigma_mle_est, optlog)


def IJ_approx(X, k, alpha, U0, params0, cost0, cost, laplace_cost,
              T=1, c=1, beta=-0.5, vmin=1e-5):
    """
    Estimate the SVC for different foreground dimensions
    using the fast linear approximation.
    """
    # Construct laplace cost for IJ.
    print('-------- Compute Hessian... --------')
    t0 = datetime.now()
    hess0 = hessian(lambda par: laplace_cost(par, U0, T=T))(params0)
    print('... done', datetime.now() - t0)

    # Initial marginal.
    print('-------- Compute full dataset marginal nksd ... --------')
    t0 = datetime.now()
    N, d = np.shape(X)
    marginal_est, lapl_extras = laplace_approx(cost0, hess0, params0, d, k,
                                               alpha, T=T, vmin=vmin)
    print('... done', datetime.now() - t0)

    print('-------- Compute hold-one-out cost ... --------')
    t0 = datetime.now()
    od_cost = construct_od_cost(X, U0, d, k, c=c, beta=beta, vmin=vmin)
    print('... done', datetime.now() - t0)

    print('-------- Get IJ gradients ... --------')
    t0 = datetime.now()
    theta_IJs = np.zeros((d, len(params0)))
    cost_inits, cost_IJs = np.zeros(d), np.zeros(d)
    for j in range(d):
        cost_inits[j] = od_cost(params0, j, T=T)
        grad_theta_1 = grad(lambda par: od_cost(par, j, T=T))(params0)
        theta_IJs[j, :] = params0 - solve(hess0, grad_theta_1)
        cost_IJs[j] = od_cost(theta_IJs[j, :], j, T=T, perturbed=True)
    delta_mnksd = -cost0 + cost_IJs
    print('... done', datetime.now() - t0)

    print('-------- Done! ... --------')
    extras = {'theta_IJs': theta_IJs, 'cost_inits': cost_inits,
              'cost_IJs': cost_IJs, 'cost0': cost0}
    extras.update(lapl_extras)

    return delta_mnksd, marginal_est, extras


def marg_NKSD_IJ(X, k, alpha, T=1, c=1, beta=-0.5, vmin=1e-5, maxiter=100):
    """
    Optimize the NKSD on the full dataset, then approximate the SVC for
    datasets with individual columns removed.
    """
    (U0, params0, cost0, cost, laplace_cost,
     Sigma_ksd_est, Sigma_mle_est, optlog) = min_NKSD(
        X, k, T=T, c=c, beta=beta, vmin=vmin, maxiter=maxiter)

    delta_mnksd, marginal_est, extras = IJ_approx(
            X, k, alpha, U0, params0, cost0, cost, laplace_cost,
            T=T, c=c, beta=beta, vmin=vmin)
    extras.update({'optlog': optlog})

    return delta_mnksd, marginal_est, Sigma_ksd_est, Sigma_mle_est, T, extras


def simulate_data(N, k, d, corruptd=0, corruptfrac=0.5):
    """Simulate an example dataset."""
    # Draw parameter.
    H_true = np.random.randn(d, k)
    v_true = np.random.rand()
    # Compute correlation matrix, set up distribution.
    Sigma_true = np.matmul(H_true, H_true.T) + v_true**2 * np.eye(d)
    # Sample.
    X = multivariate_normal.rvs(np.zeros(d), cov=Sigma_true, size=N)

    # Corrupt some of the dimensions
    if corruptd > 0 and corruptfrac > 0:
        X[:int(N*corruptfrac), -corruptd:] = np.random.randn(corruptd)[None, :]
    gene_names = np.array(['in', 'out'])[
                      np.concatenate((np.zeros(d-corruptd, dtype=np.int64),
                                      np.ones(corruptd, dtype=np.int64)))]
    return X, gene_names, H_true, v_true, Sigma_true


def load_data(file):
    """Load (preprocessed) data from file."""
    with open(file, 'rb') as dr:
        X = pickle.load(dr)
        gene_names = pickle.load(dr)

    return X, gene_names


def main(config):

    np.random.seed(int(config['general']['rng_seed']))
    simulate = config['general']['simulate'] == 'True'
    if simulate:
        N = 500
        k = 2
        d = 12
        corruptd = 2
        corruptfrac = 0.5
        X, gene_names, H_true, v_true, Sigma_true = simulate_data(
                    N, k, d, corruptd=corruptd, corruptfrac=corruptfrac)
    else:
        X, gene_names = load_data(config['data']['file'])
        N, d = X.shape

    # Load model parameters.
    k = int(config['model']['k'])
    alpha = float(config['model']['alpha'])

    # Load NKSD parameters.
    c = float(config['nksd']['kernel_c'])
    beta = float(config['nksd']['kernel_beta'])
    T = float(config['nksd']['nksd_T'])

    # Estimate.
    IJ_grad, marginal_est, Sigma_ksd_est, Sigma_mle_est, T, extras = (
                marg_NKSD_IJ(X, k, alpha, T=T, c=c, beta=beta))

    if config['general']['save'] == 'True':
        out_folder = config['general']['out_folder']
        if config['general']['make_subfolder'] == 'True':
            time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_folder = os.path.join(out_folder, time_stamp)
            os.mkdir(out_folder)

        # Results.
        if simulate:
            with open(os.path.join(out_folder, 'simulated_data.pickle'), 'wb'
                      ) as rw:
                pickle.dump(X, rw)
                pickle.dump(gene_names, rw)
        with open(os.path.join(out_folder, 'results.pickle'), 'wb') as rw:
            pickle.dump(IJ_grad, rw)
            pickle.dump(marginal_est, rw)
            pickle.dump(Sigma_ksd_est, rw)
            pickle.dump(Sigma_mle_est, rw)
            pickle.dump(T, rw)
            pickle.dump(extras, rw)
        with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
            config.write(cw)

        # Plot.
        plt.figure(figsize=(8, 6))
        plt.plot(np.reshape(Sigma_mle_est, [-1]), 'o')
        plt.plot(np.reshape(Sigma_ksd_est, [-1]), 'o')
        if simulate:
            plt.plot(np.reshape(Sigma_true, [-1]), 'o')
        plt.legend(['true', 'mle', 'ksd'])
        plt.ylabel('covariance', fontsize=18)
        plt.xlabel('matrix element', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, 'PCA_Sigma_estimation.pdf'))

        plt.figure(figsize=(6, 6))
        plt.plot(-IJ_grad, 'o')
        plt.xticks(np.arange(len(IJ_grad)), gene_names,
                   rotation=-90, fontsize=14)
        plt.yticks(fontsize=16)
        plt.ylabel(r'$\log \mathcal{K}_j - \log \mathcal{K}_0$', fontsize=18)
        plt.xlabel('dimension', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, 'PCA_IJ.pdf'))

        plt.figure(figsize=(6, 6))
        naive_shift = -extras['cost0'] + extras['cost_inits']
        plt.plot(-naive_shift, 'o')
        plt.xticks(np.arange(len(IJ_grad)), gene_names,
                   rotation=-90, fontsize=14)
        plt.yticks(fontsize=16)
        plt.ylabel(r'$\log \mathcal{E}_j - \log \mathcal{E}_0$', fontsize=18)
        plt.xlabel('dimension', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, 'PCA_init_cost.pdf'))


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="SVC data selection on a pPCA model.")
    parser.add_argument('configPath')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.configPath)

    main(config)
