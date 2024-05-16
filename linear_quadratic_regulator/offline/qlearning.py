"""qlearning.py


"""

import numpy as np
import scipy.linalg

import utils


def psd_project(M, mu, L):
    #assert mu < L
    #_assert_symmetric_matrix(M)

    evals, evecs = np.linalg.eigh(M)
    evals[evals < mu] = mu
    evals[evals > L] = L
    return evecs.dot(np.diag(evals)).dot(evecs.T)


def sarsa(Astar, Bstar, Q, R, S0, sigma_w, sigma_eta,
          num_outer_iters, num_inner_iters, step_size, mu, L, rng=None):

    if rng is None:
        rng = np.random

    n, d = Bstar.shape

    Scur = psd_project(np.array(S0), mu, L)

    # debugging for now
    Pstar, _ = utils.dlqr(Astar, Bstar, Q, R)
    lambda_star = np.trace(Pstar) * (sigma_w ** 2)

    policies = []

    xcur = np.zeros((n,))

    ctr = 1

    for _ in range(num_outer_iters):

        wts = sigma_w * rng.normal(size=(num_inner_iters, n))
        etas = sigma_eta * rng.normal(size=(num_inner_iters, d))

        for t in range(num_inner_iters):
            Kcur = -scipy.linalg.solve(Scur[n:, n:], Scur[:n, n:].T, sym_pos=True)

            ut = Kcur.dot(xcur) + etas[t]
            zt = np.hstack((xcur, ut))
            ct = utils.quad_form(Q, xcur) + utils.quad_form(R, ut)
            xnext = Astar.dot(xcur) + Bstar.dot(ut) + wts[t]
            utp1 = Kcur.dot(xnext)

            g = (ct - lambda_star + utils.quad_form(Scur, np.hstack((xnext, utp1))) - utils.quad_form(Scur, zt)) * np.outer(zt, zt)
            xcur = xnext
            Scur = psd_project(Scur - (step_size/ctr) * g, mu, L)

        ctr += 1

        policies.append(Kcur)

    return policies


def qlearning(Astar, Bstar, Q, R, S0, Kplay, sigma_w, sigma_eta,
              num_outer_iters, num_inner_iters, step_size, mu, L, rng=None):

    assert utils.spectral_radius(Astar + Bstar.dot(Kplay)) < 1

    if rng is None:
        rng = np.random

    n, d = Bstar.shape

    Scur = psd_project(np.array(S0), mu, L)

    # debugging for now
    Pstar, _ = utils.dlqr(Astar, Bstar, Q, R)
    lambda_star = np.trace(Pstar) * (sigma_w ** 2)

    policies = []

    xcur = np.zeros((n,))

    ctr = 1

    for _ in range(num_outer_iters):

        wts = sigma_w * rng.normal(size=(num_inner_iters, n))
        etas = sigma_eta * rng.normal(size=(num_inner_iters, d))

        for t in range(num_inner_iters):
            ut = Kplay.dot(xcur) + etas[t]
            zt = np.hstack((xcur, ut))
            ct = utils.quad_form(Q, xcur) + utils.quad_form(R, ut)
            #I_K = np.vstack((np.eye(n), Kstar))
            #F = (sigma_w ** 2) * I_K.dot(I_K.T)
            xnext = Astar.dot(xcur) + Bstar.dot(ut) + wts[t]
            Sopt = Scur[:n, :n] - np.dot(Scur[:n, n:], scipy.linalg.solve(Scur[n:, n:], Scur[:n, n:].T, sym_pos=True))
            g = (ct - lambda_star + utils.quad_form(Sopt, xnext) - utils.quad_form(Scur, zt)) * np.outer(zt, zt)
            xcur = xnext
            Scur = psd_project(Scur + step_size * g, mu, L)

            ctr += 1

        #print("Scur", Scur)

        Kcur = -scipy.linalg.solve(Scur[n:, n:], Scur[:n, n:].T, sym_pos=True)
        policies.append(Kcur)

    return policies
