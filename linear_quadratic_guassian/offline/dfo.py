"""dfo.py


"""


import numpy as np
import scipy.linalg

import utils


def dfo(Astar, Bstar, Q, R, K0, sigma_w, sigma_eta,
        num_dfo_iters, horizon_length, step_size, norm_project,
        rng=None):

    assert utils.spectral_radius(Astar + Bstar.dot(K0)) < 1

    if rng is None:
        rng = np.random

    Kcur = np.array(K0)

    n, d = Bstar.shape

    policies = [None] * num_dfo_iters

    for idx in range(num_dfo_iters):
        delta = rng.normal(size=(d, n))
        ws = sigma_w * rng.normal(size=(horizon_length, n))

        def roll_forward(K):

            xs = np.zeros((horizon_length+1, n))
            costs = np.zeros((horizon_length,))

            xcur = np.zeros((n,))
            for t in range(horizon_length):
                ucur = K.dot(xcur)
                xnext = Astar.dot(xcur) + Bstar.dot(ucur) + ws[t]
                costs[t] = utils.quad_form(Q, xcur) + utils.quad_form(R, ucur)
                xs[t+1] = xnext
                xcur = xnext

            return (1/horizon_length) * np.sum(costs)


        ghat = (roll_forward(Kcur + sigma_eta * delta) - roll_forward(Kcur - sigma_eta * delta))/(2*sigma_eta) * delta
        #print("ghat")
        #print(ghat)

        Kcur -= step_size * ghat
        if np.linalg.norm(Kcur, ord='fro') > norm_project:
            Kcur = Kcur * norm_project / np.linalg.norm(Kcur, ord='fro')

        #print("Knext", Kcur)
        #print("loop next", Astar + Bstar.dot(Kcur))

        if utils.spectral_radius(Astar + Bstar.dot(Kcur)) >= 1:
            return policies
        else:
            policies[idx] = np.array(Kcur)

    return policies
