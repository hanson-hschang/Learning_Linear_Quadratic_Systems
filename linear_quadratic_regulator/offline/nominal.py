"""nominal.py


"""


import numpy as np
import scipy.linalg

import utils

def nominal(Astar, Bstar, Q, R, K0, sigma_w, sigma_u,
            num_iters, horizon_length, rng=None):


    assert utils.spectral_radius(Astar + Bstar.dot(K0)) < 1

    if rng is None:
        rng = np.random

    n, d = Bstar.shape

    rls = utils.RecursiveLeastSquaresEstimator(n, d, 1e-4)

    policies = [None] * num_iters

    for idx in range(num_iters):
        etas = sigma_u * rng.normal(size=(horizon_length, d))
        ws = sigma_w * rng.normal(size=(horizon_length, n))

        xcur = np.zeros((n,))
        for t in range(horizon_length):
            ucur = K0.dot(xcur) + etas[t]
            xnext = Astar.dot(xcur) + Bstar.dot(ucur) + ws[t]
            rls.update(xcur, ucur, xnext)
            xcur = xnext

        Ah, Bh, _ = rls.get_estimate()
        _, Kcur = utils.dlqr(Ah, Bh, Q, R)

        if utils.spectral_radius(Astar + Bstar.dot(Kcur)) < 1:
            policies[idx] = Kcur

    return policies
