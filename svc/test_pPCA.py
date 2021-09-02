from svc.pPCA import np
from svc import pPCA as pds
from copy import deepcopy
from autograd import jacobian


def test_compare_K_matrices():
    X = np.random.randn(20, 4)
    c = 2.
    beta = -0.3

    XKX, XKp, Kpp, Kbar = (
        pds.K_matrices(X[:, np.array([True, False, True, True])],
                       c=c, beta=beta))

    odXKX, odXKp, odKpp, odKbar, _ = pds.onedrop_K_matrices(X, c=c, beta=beta)

    assert np.allclose(odXKX[:, :, 1], XKX)
    assert np.allclose(odXKp[:, :, 1], XKp)
    assert np.allclose(odKpp[1], Kpp)
    assert np.allclose(odKbar[1], Kbar)


def test_compare_costs_basic():

    N = 20
    d = 4
    k = 2
    c = 2.
    beta = -0.3
    vmin = 0.1
    T = 0.4
    X = np.random.randn(N, d)
    m = int(d*k - k*(k+1)/2)

    U_mle, L_mle, v_mle, _ = pds.PCA_MLE(X, k)
    params = [U_mle, np.log(L_mle), np.log(v_mle)]

    cost, lp_cost = pds.construct_cost(X, k, c=c, beta=beta, vmin=vmin)
    cost0 = cost(params, T=T)

    params0 = np.zeros(m + k + 1)
    params0[m:(m+k)] = params[1]
    params0[m+k] = params[2]

    cost1 = lp_cost(params0, U_mle, T=T)
    print(cost0, cost1)
    assert np.allclose(cost0, cost1)


def test_compare_costs_subset():

    N = 20
    d = 4
    k = 2
    c = 2.
    beta = -0.3
    vmin = 0.1
    T = 0.4
    X = np.random.randn(N, d)
    m = int(d*k - k*(k+1)/2)

    U_mle, L_mle, v_mle, _ = pds.PCA_MLE(X, k)
    U_cut = np.concatenate((U_mle[:1, :], U_mle[2:, :]), axis=0)
    params = [U_cut, np.log(L_mle), np.log(v_mle)]

    cost, _ = pds.construct_cost(X[:, np.array([True, False, True, True])], k,
                                 c=c, beta=beta, vmin=vmin)
    cost0 = cost(params, T=T)

    params0 = np.zeros(m + k + 1)
    params0[m:(m+k)] = params[1]
    params0[m+k] = params[2]

    od_cost = pds.construct_od_cost(X, U_mle, d, k, c=c, beta=beta, vmin=vmin)
    cost1 = od_cost(params0, 1, T=T)
    print(cost0, cost1)

    assert np.allclose(cost0, cost1)


def test_PCA_volumeln():

    N = 22
    Lamfull = np.array([3.2, 1.3, 0.7, 0.3, 0.1])
    d = 5
    v = 0.2
    k = 3
    alpha = 0.1

    chk_volumeln = pds.PCA_volumeln(N, Lamfull, v, k, alpha)

    Lamhat = deepcopy(Lamfull)
    Lamhat[k:] = v
    tst_volumeln = 1
    for i in range(k):
        for j in range(i+1, d):
            tst_volumeln = tst_volumeln*(
                    (1/Lamhat[j] - 1/Lamhat[i])*(Lamfull[i] - Lamfull[j])*N)
    print(np.log(tst_volumeln))
    tst_volumeln = tst_volumeln*(
            np.power((N - 1 + alpha)/2, k)*(((N+1+alpha)*(d-k) - 2)/2))
    tst_volumeln = np.log(tst_volumeln)

    assert np.allclose(chk_volumeln, tst_volumeln)


def test_PCA_MLE():

    X = np.array([[1.2, 0.3, -0.1],
                  [0.1, -0.7, 0.4],
                  [0.8, -0.4, 0.2],
                  [-1.0, -2., -1.0]])
    k = 2
    chk_U, chk_L, chk_v, chk_Lamfull = pds.PCA_MLE(X, k)

    S = np.matmul(X.T, X)/X.shape[0]
    w, v = np.linalg.eig(S)
    srt = np.array([elem[0] for elem in
                    sorted(enumerate(w), key=lambda x: -x[1])], dtype=np.int64)
    w, v = w[srt], v[:, srt]

    tst_U = v[:, :k]
    tst_L = w[:k]
    tst_v = np.mean(w[k:])

    for j in range(k):
        assert (np.allclose(tst_U[:, j], chk_U[:, j]) or
                np.allclose(-tst_U[:, j], chk_U[:, j]))
    assert np.allclose(tst_L, chk_L)
    assert np.allclose(tst_v, chk_v)
    assert np.allclose(w, chk_Lamfull)


def test_PCA_latent_z_posterior():

    X = np.array([[1.2, 0.3, -0.1],
                  [0.1, -0.7, 0.4],
                  [0.8, -0.4, 0.2],
                  [-1.0, -2., -1.0]])
    k = 2
    U, L, v, _ = pds.PCA_MLE(X, k)
    chk_zi = pds.PCA_latent_z_posterior(X, U, L, v)

    W = np.matmul(U, np.diag(np.sqrt(L - v)))
    F = np.matmul(W.T, W) + np.diag(v*np.ones(2))
    tst_zi = np.matmul(np.linalg.inv(F), np.matmul(W.T, X.T))

    assert np.allclose(chk_zi.T, tst_zi)


def test_min_NKSD():

    if True:
        N = 20
        d = 4
        k = 2
        c = 2.
        beta = -0.3
        vmin = 0.1
        T = 0.4
        X = np.random.randn(N, d)

        (U0, params0, cost0, cost, laplace_cost,
         Sigma_ksd_est, Sigma_mle_est, optlog) = pds.min_NKSD(
                X, k, T=T, c=c, beta=beta, vmin=vmin, maxiter=100)

        chk_cost = laplace_cost(params0, U0)
        assert np.allclose(chk_cost, cost0)

        hess0 = pds.hessian(lambda par: laplace_cost(par, U0, T=T))(params0)
        det = pds.slogdet(hess0)

        assert det[0] > 0


def test_IJ_approx():

    N = 20
    d = 4
    k = 2
    c = 2.
    beta = -0.3
    vmin = 0.1
    T = 0.4
    X = np.random.randn(N, d)

    (U0, params0, cost0, cost, laplace_cost,
     Sigma_ksd_est, Sigma_mle_est, optlog) = pds.min_NKSD(
            X, k, T=T, c=c, beta=beta, vmin=vmin, maxiter=100)

    hess0 = pds.hessian(lambda par: laplace_cost(par, U0, T=T))(params0)
    od_cost = pds.construct_od_cost(X, U0, d, k, c=c, beta=beta, vmin=vmin)
    grad_theta_1 = jacobian(lambda par: od_cost(par, 1, T=T))(params0)
    w = 1.0
    theta_IJ = params0 - np.matmul(np.linalg.inv(hess0), grad_theta_1) * w
    cost_init = od_cost(params0, 1, T=T)
    v1 = cost_init + w*(od_cost(theta_IJ, 1, T=T) - cost_init)
    v0 = cost_init + w*(od_cost(params0, 1, T=T) - cost_init)

    assert v1 < v0

    delta_mnksd, marginal_est, extras = pds.IJ_approx(
            X, k, 0.1, U0, params0, cost0, cost, laplace_cost,
            T=T, c=c, beta=beta, vmin=vmin)

    assert np.allclose(extras['theta_IJs'][1, :], theta_IJ)
    print(extras['cost_inits'], extras['cost_IJs'])
    assert np.all(extras['cost_inits'] > extras['cost_IJs'])
