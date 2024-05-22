"""uniform_search.py


"""


import numpy as np

import utils


def uniform_search(Astar, Bstar, Q, R, sigma_w,
                   num_iters, horizon_length, ball_width,
                   rng=None):

    if rng is None:
        rng = np.random

    n, d = Bstar.shape

    def uniform_sample(d, n, width):
        """Uniformly sample a d x n matrix from the set { K : ||K||_F <= width }

        """

        ret = rng.normal(size=(d, n))
        ret *= np.power(rng.uniform(), 1/(n*d)) / np.linalg.norm(ret, ord="fro")
        return ret

    best_so_far = None
    best_cost_so_far = None

    policies = []

    for idx in range(num_iters):

        ws = sigma_w * rng.normal(size=(horizon_length, n))
        Kcur = uniform_sample(d, n, ball_width)

        xcur = np.zeros((n,))
        cost = 0
        for t in range(horizon_length):
            ucur = Kcur.dot(xcur)
            xnext = Astar.dot(xcur) + Bstar.dot(ucur) + ws[t]
            cost += utils.quad_form(Q, xcur) + utils.quad_form(R, ucur)
            xcur = xnext
        cost /= horizon_length

        if (best_so_far is None) or (cost < best_cost_so_far):
            best_so_far = Kcur
            best_cost_so_far = cost

        policies.append(best_so_far)

    return policies


