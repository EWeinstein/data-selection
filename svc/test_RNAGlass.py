import pytest
import torch

from svc import RNAGlass as model


def test_Jorganizer():
    Jraw = torch.zeros((3, 2, 2))
    Jraw[0] = torch.tensor([[1., 2.],
                            [3., 4.]])
    Jraw[1] = torch.tensor([[5., 6.],
                            [7., 8.]])
    Jraw[2] = torch.tensor([[9., 10.],
                            [11., 12.]])
    tst_J = torch.zeros((3, 3, 2, 2))
    tst_J[0, 1] = Jraw[0]
    tst_J[1, 0] = Jraw[0].transpose(0, 1)
    tst_J[0, 2] = Jraw[1]
    tst_J[2, 0] = Jraw[1].transpose(0, 1)
    tst_J[1, 2] = Jraw[2]
    tst_J[2, 1] = Jraw[2].transpose(0, 1)

    jorganizer = model.Jorganizer(3)
    chk_J = jorganizer.forward(Jraw)

    assert torch.allclose(chk_J, tst_J)


@pytest.mark.parametrize('onoff_shapes', [False, True])
def test_stein_score(onoff_shapes):
    N = 5
    D = 3

    Jraw = torch.randn((D, 2, 2))
    jorganizer = model.Jorganizer(D)
    glass_J = jorganizer.forward(Jraw)
    glass_h = torch.randn((D, 2))
    gene_scale = 1.5
    gene_mean = 0.1
    x = torch.randn((N, 3))
    if onoff_shapes:
        on_mean = 2 + torch.randn(D)
        on_scale = 0.5 * torch.ones(D)
        on_scale[0] += 1.
        off_scale = torch.tensor(0.5)
    else:
        on_mean, on_scale, off_scale = None, None, None

    steinglass = model.SteinGlass(gene_scale, gene_mean, glass_h, glass_J,
                                  on_mean, on_scale, off_scale,
                                  onoff_shapes=onoff_shapes)
    chk_score = steinglass.stein_score(x)

    xrg = torch.tensor(x, requires_grad=True)
    tst_score = torch.ones_like(xrg)
    invlogit = 1/(1 + torch.exp(-gene_scale * (xrg - gene_mean)))
    sigma = torch.cat([(1-invlogit.unsqueeze(-1)), invlogit.unsqueeze(-1)],
                      -1)
    energy = (torch.einsum('ijk,jk->', sigma, glass_h) +
              0.5*torch.einsum('ijk,jakb,iab', sigma, glass_J, sigma))
    if onoff_shapes:
        energy += torch.sum(invlogit * (-0.5*((xrg - on_mean[None, :])**2)
                                        / (on_scale**2)[None, :]))
        energy += torch.sum((1-invlogit) * (-0.5*(xrg**2)
                                            / (off_scale**2)))
    energy.backward()
    tst_score = xrg.grad

    assert torch.allclose(chk_score, tst_score)


@pytest.mark.parametrize('kernel_l', [None, -0.5])
def test_kernel_terms(kernel_l):
    N = 5
    D = 3

    Jraw = torch.randn((D, 2, 2))
    jorganizer = model.Jorganizer(D)
    glass_J = jorganizer.forward(Jraw)
    glass_h = torch.randn((D, 2))
    gene_scale = 1.5
    gene_mean = 0.1
    x = torch.randn((N, 3))

    steinglass = model.SteinGlass(gene_scale, gene_mean, glass_h, glass_J,
                                  None, None, None, onoff_shapes=False,
                                  kernel_beta=-0.5*3, kernel_l=kernel_l)
    chk_K, chk_Kp, chk_Kpp, chk_Kbar = steinglass.kernel_terms(x)

    assert torch.allclose(torch.diag(chk_K), torch.zeros(N))
    assert torch.allclose(torch.diag(chk_Kp[:, :, 1]), torch.zeros(N))
    assert torch.allclose(torch.diag(chk_Kpp), torch.zeros(N))

    tst_K = torch.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                tst_K[i, j] = torch.prod((1 + (x[i, :] - x[j, :])**2)**-0.5,
                                         dim=-1)
                if kernel_l is not None:
                    tst_K[i, j] = tst_K[i, j] * torch.prod(
                            (x[i, :] - kernel_l) * (x[j, :] - kernel_l))
    assert torch.allclose(tst_K, chk_K)
    assert torch.allclose(torch.sum(tst_K), chk_Kbar)

    xrg = torch.tensor(x, requires_grad=True)
    if kernel_l is None:
        K02 = torch.prod((1 + (xrg[0, :] - x[2, :])**2)**-0.5, dim=-1)
    else:
        K02 = torch.prod((1 + (xrg[0, :] - x[2, :])**2)**-0.5
                         * (xrg[0, :] - kernel_l) * (x[2, :] - kernel_l),
                         dim=-1)
    K02.backward()
    assert torch.allclose(chk_Kp[0, 2], xrg.grad[0])

    tst_Kpp02 = 0
    for d in range(D):
        xrg = torch.tensor(x, requires_grad=True)
        if kernel_l is None:
            Kp02 = (2 * (-0.5) * (x[0, :] - xrg[2, :]) *
                    (1/(1 + (x[0, :] - xrg[2, :])**2)) *
                    torch.prod((1 + (x[0, :] - xrg[2, :])**2)**-0.5, dim=-1)
                    )[d]
        else:
            Kp02 = (2 * (-0.5) * (x[0, :] - xrg[2, :]) *
                    (1/(1 + (x[0, :] - xrg[2, :])**2)) *
                    torch.prod((1 + (x[0, :] - xrg[2, :])**2)**-0.5
                               * (x[0, :] - kernel_l)
                               * (xrg[2, :] - kernel_l), dim=-1) +
                    torch.prod((1 + (x[0, :] - xrg[2, :])**2)**-0.5
                               * (x[0, :] - kernel_l)
                               * (xrg[2, :] - kernel_l), dim=-1) / (
                               x[0, :] - kernel_l))[d]
        Kp02.backward()
        tst_Kpp02 += xrg.grad[2, d]
    assert torch.allclose(chk_Kpp[0, 2], tst_Kpp02)


@pytest.mark.parametrize('kernel_l', [None, -0.5])
def test_log_prob(kernel_l):
    N = 5
    D = 3

    Jraw = torch.randn((D, 2, 2))
    jorganizer = model.Jorganizer(D)
    glass_J = jorganizer.forward(Jraw)
    glass_h = torch.randn((D, 2))
    gene_scale = 1.5
    gene_mean = 0.1
    x = torch.randn((N, 3))

    steinglass = model.SteinGlass(gene_scale, gene_mean, glass_h, glass_J,
                                  None, None, None, onoff_shapes=False,
                                  kernel_l=kernel_l)
    sscore = steinglass.stein_score(x)
    K, Kp, Kpp, Kbar = steinglass.kernel_terms(x)

    chk_nloss = steinglass.log_prob(x)

    terms = torch.zeros(4)
    for j in range(N):
        for jp in range(N):
            if j != jp:
                terms[0] += torch.dot(sscore[j, :], sscore[jp, :]) * K[j, jp]
                terms[1] += torch.dot(sscore[j, :], Kp[jp, j, :])
                terms[2] += torch.dot(Kp[j, jp, :], sscore[jp, :])
                terms[3] += Kpp[j, jp]
    ksd = torch.sum(terms)
    nksd = ksd / Kbar
    tst_nloss = -N * nksd

    assert torch.allclose(chk_nloss, tst_nloss)


def test_dataselector_optim():

    torch.set_default_dtype(torch.float64)

    dataselector = model.DataSelector(50, 2, PY_mix_d=0., loorf_init=0.6)
    epochs = 1000
    loorf_samples = 10
    target = torch.tensor([1., 0.], dtype=torch.double)
    for epoch in range(epochs):
        selects = torch.zeros((loorf_samples, 2),
                              dtype=torch.bool)
        nelbos = torch.zeros(loorf_samples)
        for s in range(loorf_samples):
            selects[s] = dataselector.sample_select()
            nelbos[s] = torch.abs(selects[s].to(torch.double) -
                                  target).sum()
        svc = dataselector.step(selects, nelbos)
    print(svc, dataselector._logit())
    assert torch.allclose(dataselector._logit(), target, atol=0.1)


@pytest.mark.parametrize('select_all', ['False', 'True'])
@pytest.mark.parametrize('onoff_shapes', ['False', 'True'])
def test_sparseglass(select_all, onoff_shapes):
    # Smoke test.
    config = {'general': {'cpu_data': 'False',
                          'cuda': 'False',
                          'pin_memory': 'False',
                          'test': 'True',
                          'small': 'True',
                          'rng_seed': '1',
                          'save': 'False'},
              'model': {'prior_gene_scale_mn': '0',
                        'prior_gene_scale_sd': '1',
                        'prior_gene_scale_lbound': '1',
                        'prior_gene_mean_mn': '0',
                        'prior_gene_mean_sd': '1',
                        'prior_on_mean_mn': '0',
                        'prior_on_mean_sd': '10',
                        'prior_on_scale_mn': '0',
                        'prior_on_scale_sd': '10',
                        'prior_off_scale_mn': '0',
                        'prior_off_scale_sd': '1',
                        'onoff_shapes': onoff_shapes,
                        'prior_glass_h_sd': '1',
                        'prior_glass_J_scale': '0.1'},
              'svc': {'PY_mix_d': '2',
                      'PY_conc': '-0.25'},
              'nksd': {'kernel_c': '1',
                       'kernel_beta': '-0.5',
                       'kernel_l': '-10',
                       'nksd_T': '1'},
              'train': {'batch_size': '5',
                        'learning_rate': '0.1',
                        'milestones': '[]',
                        'learning_gamma': '1.',
                        'n_epochs': '2',
                        'jit': 'True',
                        'loorf_samples': '4',
                        'loorf_init': '0.8',
                        'select_all': select_all,
                        'early_stop': 'True',
                        'smooth_wind': '2'}}
    model.main(config)
