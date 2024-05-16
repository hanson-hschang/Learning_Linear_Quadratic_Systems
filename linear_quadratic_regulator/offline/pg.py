"""pg.py


"""

import numpy as np

import utils


def policy_gradients(Astar, Bstar, Q, R, K0, sigma_w, sigma_eta,
                     num_pg_iters, horizon_length, step_size, norm_project,
                     baseline, rng=None):

    assert utils.spectral_radius(Astar + Bstar.dot(K0)) < 1
    assert baseline in ('simple', 'value_function')

    if rng is None:
        rng = np.random

    Kcur = np.array(K0)

    n, d = Bstar.shape

    policies = [None] * num_pg_iters
    prev_cost = 0

    for idx in range(num_pg_iters):
        etas = sigma_eta * rng.normal(size=(horizon_length, d))
        ws = sigma_w * rng.normal(size=(horizon_length, n))

        xs = np.zeros((horizon_length+1, n))
        costs = np.zeros((horizon_length,))

        xcur = np.zeros((n,))
        for t in range(horizon_length):
            ucur = Kcur.dot(xcur) + etas[t]
            xnext = Astar.dot(xcur) + Bstar.dot(ucur) + ws[t]
            costs[t] = utils.quad_form(Q, xcur) + utils.quad_form(R, ucur)
            xs[t+1] = xnext
            xcur = xnext

        if baseline == 'simple':
            baseline_fn = lambda xt: prev_cost
        else:
            Vt = utils.solve_discrete_lyapunov(Astar + Bstar.dot(Kcur), Q + Kcur.T.dot(R).dot(Kcur))
            baseline_fn = lambda xt: utils.quad_form(Vt, xt)
        prev_cost = np.sum(costs)/horizon_length

        ghat = np.zeros_like(Kcur)
        for t in range(horizon_length):
            ghat += np.outer(etas[t], xs[t])*(np.sum(costs[t:]) - baseline_fn(xs[t]))
        ghat /= (horizon_length * (sigma_eta ** 2))

        #print(costs)
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
