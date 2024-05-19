"""policy_iteration.py

"""

import numpy as np
import scipy.linalg

import utils

from numba import jit


class InstabilityException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)

class RankDegeneracyException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)


def _assert_matrix(M):
    assert len(M.shape) == 2


def _assert_square_matrix(M):
    _assert_matrix(M)
    assert M.shape[0] == M.shape[1]


def _assert_symmetric_matrix(M):
    _assert_square_matrix(M)
    assert np.allclose(M, M.T)


@jit(nopython=True)
def extract_upper_triangle(M):
    # assume M is square
    n, _ = M.shape
    d = n * (n-1) // 2
    ret = np.zeros((d,), dtype=M.dtype)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            ret[count] = M[i, j]
            count += 1
    return ret


@jit(nopython=True)
def svec(M):
    #assert len(M.shape) == 2
    #assert M.shape[0] == M.shape[1]
    diag = np.diag(M)
    #off_diag = np.sqrt(2)*M[np.triu_indices(M.shape[0], k=1)]
    off_diag = np.sqrt(2)*extract_upper_triangle(M)
    return np.hstack((diag, off_diag))

def svec_simple(M):
    assert len(M.shape) == 2
    assert M.shape[0] == M.shape[1]
    diag = np.diag(M)
    off_diag = np.sqrt(2)*M[np.triu_indices(M.shape[0], k=1)]
    return np.hstack((diag, off_diag))


def smat(v):
    n = int((np.sqrt(1+8*v.shape[0])-1)/2)
    assert n*(n+1) == 2*v.shape[0]
    V = np.zeros((n, n))
    off_diag = v[n:]/np.sqrt(2.0)
    V[np.triu_indices(V.shape[0], k=1)] = off_diag
    V += V.T
    V[np.diag_indices(V.shape[0])] = v[:n]
    return V


@jit(nopython=True)
def phi(x, u):
    z = np.hstack((x, u))
    return svec(np.outer(z, z))


def phi_simple(x, u):
    z = np.hstack((x, u))
    return svec_simple(np.outer(z, z))


def psd_project(M, mu, L):
    assert mu < L
    _assert_symmetric_matrix(M)

    evals, evecs = np.linalg.eigh(M)
    evals[evals < mu] = mu
    evals[evals > L] = L
    return evecs.dot(np.diag(evals)).dot(evecs.T)


@jit(nopython=True)
def _fill_Psis(Phis, Psis, next_states, next_inputs, f):
    T, _ = Phis.shape
    for t in range(T):
        Psis[t] = Phis[t] - phi(next_states[t], next_inputs[t]) + f


def lstd_q_simple(Phis, next_states, costs, Keval, sigma_w, mu, L):
 
    T, n = next_states.shape

    I_K = np.vstack((np.eye(n), Keval))
    f = (sigma_w ** 2) * svec(I_K.dot(I_K.T))

    Psis = np.zeros_like(Phis)
    next_inputs = next_states.dot(Keval.T)

    def psi(x, Kcur):
        z = np.hstack((x, Kcur.dot(x)))
        return svec_simple(np.outer(z, z))

    for t in range(T):
        phi_t = Phis[t]
        psi_tp1 = psi(next_states[t], Keval)
        Psis[t] = phi_t - psi_tp1 + f

    Amat = Phis.T.dot(Psis)
    bmat = Phis.T.dot(costs)

    svals = scipy.linalg.svdvals(Amat)
    if min(svals) <= 1e-8:
        raise RankDegeneracyException(
            "Amat is degenerate: s_min(Amat)={}".format(min(svals)))
    qhat = np.linalg.lstsq(Amat, bmat)[0]
    Qhat = psd_project(smat(qhat), mu, L)
    return Qhat


def lstd_q(Phis, next_states, costs, Keval, sigma_w, mu, L):

    _, n = next_states.shape

    I_K = np.vstack((np.eye(n), Keval))
    f = (sigma_w ** 2) * svec(I_K.dot(I_K.T))
    #print(f)

    Psis = np.zeros_like(Phis)
    next_inputs = next_states.dot(Keval.T)

    _fill_Psis(Phis, Psis, next_states, next_inputs, f)
    #print(Psis)

    Amat = Phis.T.dot(Psis)
    bmat = Phis.T.dot(costs)

    svals = scipy.linalg.svdvals(Amat)
    if min(svals) <= 1e-8:
        raise RankDegeneracyException(
            "Amat is degenerate: s_min(Amat)={}".format(min(svals)))
    qhat = np.linalg.lstsq(Amat, bmat)[0]
    Qhat = psd_project(smat(qhat), mu, L)
    return Qhat


def policy_iteration_full(Astar, Bstar, Q, R, K0, Kplay, sigma_w, sigma_eta,
                          num_resets, horizon_length, num_PI_iters, mu, L, rng=None):

    assert utils.spectral_radius(Astar + Bstar.dot(K0)) < 1

    if rng is None:
        rng = np.random

    Kcur = np.array(K0)

    n, d = Bstar.shape
    lifted_dim = (n + d)*(n + d + 1) // 2

    policies = [None] * num_resets

    Phis = np.zeros((num_resets * horizon_length, lifted_dim))
    next_states = np.zeros((num_resets * horizon_length, n))
    costs = np.zeros((num_resets * horizon_length,))

    for reset_idx in range(num_resets):

        etas = sigma_eta * rng.normal(size=(horizon_length, d))
        ws = sigma_w * rng.normal(size=(horizon_length, n))

        xcur = np.zeros((n,))
        for t in range(horizon_length):
            ucur = Kplay.dot(xcur) + etas[t]
            Phis[reset_idx*horizon_length + t] = phi(xcur, ucur)
            xnext = Astar.dot(xcur) + Bstar.dot(ucur) + ws[t]
            costs[reset_idx*horizon_length + t] = utils.quad_form(Q, xcur) + utils.quad_form(R, ucur)
            next_states[reset_idx*horizon_length + t] = xnext
            xcur = xnext

        Kt = np.array(K0)
        unstable = False
        for i in range(num_PI_iters):
            Qt = lstd_q(Phis[:horizon_length*(reset_idx + 1)],
                        next_states[:horizon_length*(reset_idx + 1)],
                        costs[:horizon_length*(reset_idx + 1)],
                        Kt,
                        sigma_w,
                        mu, 
                        L)
            Ktp1 = -scipy.linalg.solve(Qt[n:, n:], Qt[:n, n:].T, sym_pos=True)

            if utils.spectral_radius(Astar + Bstar.dot(Ktp1)) >= 1:
                unstable = True
                break

            Kt = Ktp1

        if not unstable:
            policies[reset_idx] = Kt

    return policies 


def policy_iteration(Astar, Bstar, Q, R, K0, sigma_w, sigma_eta,
                     num_pi_iters, horizon_length, mu, L, rng=None):

    assert utils.spectral_radius(Astar + Bstar.dot(K0)) < 1

    if rng is None:
        rng = np.random

    Kcur = np.array(K0)

    n, d = Bstar.shape

    policies = [None] * num_pi_iters

    def phi(x, u):
        z = np.hstack((x, u))
        return svec(np.outer(z, z))

    for idx in range(num_pi_iters):

        # learn the Q function

        I_K = np.vstack((np.eye(n), Kcur))
        #print(I_K)

        def psi(x, Kcur):
            z = np.hstack((x, Kcur.dot(x)))
            return svec(np.outer(z, z))

        f = (sigma_w ** 2) * svec(I_K.dot(I_K.T))

        etas = sigma_eta * rng.normal(size=(horizon_length, d))
        ws = sigma_w * rng.normal(size=(horizon_length, n))

        xs = np.zeros((horizon_length+1, n))
        us = np.zeros((horizon_length, d))
        costs = np.zeros((horizon_length,))

        xcur = np.zeros((n,))
        for t in range(horizon_length):
            ucur = Kcur.dot(xcur) + etas[t]
            us[t] = ucur
            xnext = Astar.dot(xcur) + Bstar.dot(ucur) + ws[t]
            costs[t] = utils.quad_form(Q, xcur) + utils.quad_form(R, ucur)
            xs[t+1] = xnext
            xcur = xnext

        lifted_dim = (n + d)*(n + d + 1) // 2
        Amat = np.zeros((lifted_dim, lifted_dim))
        bmat = np.zeros((lifted_dim,))
        for t in range(horizon_length):
            phi_t = phi(xs[t], us[t])
            psi_tp1 = psi(xs[t+1], Kcur)
            Amat += np.outer(phi_t, phi_t - psi_tp1 + f)
            bmat += phi_t * costs[t]
        svals = scipy.linalg.svdvals(Amat)
        if min(svals) <= 1e-8:
            raise RankDegeneracyException(
                "Amat is degenerate: s_min(Amat)={}".format(min(svals)))
        qhat = np.linalg.lstsq(Amat, bmat)[0]

        # compute true Q to compare soln
        Vcur = utils.solve_discrete_lyapunov(Astar + Bstar.dot(Kcur), Q + Kcur.T.dot(R).dot(Kcur))
        A_B = np.hstack((Astar, Bstar))
        Qcur = scipy.linalg.block_diag(Q, R) + A_B.T.dot(Vcur).dot(A_B)

        #Qhat = psd_project(smat(qhat), mu, 2 * utils.lambda_max(Qcur))
        Qhat = psd_project(smat(qhat), mu, L)

        #print("Qcur")
        #print(Qcur)
        #print("Qhat")
        #print(Qhat)
        print("||Qhat - Q||_F", np.linalg.norm(Qhat - Qcur, ord='fro'))

        Kcur = -scipy.linalg.solve(Qhat[n:, n:], Qhat[:n, n:].T, sym_pos=True)

        if utils.spectral_radius(Astar + Bstar.dot(Kcur)) >= 1:
            return policies
        else:
            policies[idx] = np.array(Kcur)

    return policies


def test():

    for i in [2, 3, 4, 5, 6]:
        M = np.random.normal(size=(i, i))
        M += M.T
        assert np.allclose(svec(M), svec_simple(M)) 


if __name__ == '__main__':
    test()
