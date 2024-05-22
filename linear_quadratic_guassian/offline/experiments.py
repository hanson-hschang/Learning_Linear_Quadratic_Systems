import numpy as np
import scipy.linalg

import dfo
import nominal
import pg
import policy_iteration
import uniform_search
import qlearning

import utils
import systems
import time

import itertools as it
import multiprocessing as mp


def nominal_wrapper(args):
    Astar, Bstar, Q, R, K0, sigma_w, sigma_eta, num_iters, horizon_length, seed = args

    rng = np.random.RandomState(seed)

    policies = nominal.nominal(Astar, Bstar, Q, R, K0, sigma_w, sigma_eta,
                               num_iters, horizon_length, rng)
    costs = np.array([np.inf if policy is None else
                      utils.LQR_cost(Astar, Bstar, policy, Q, R, sigma_w)
                      for policy in policies])

    return costs


def policy_gradients_wrapper(args):
    Astar, Bstar, Q, R, K0, sigma_w, sigma_eta, num_pg_iters, horizon_length, step_size, proj_norm, baseline, seed = args

    rng = np.random.RandomState(seed)

    policies = pg.policy_gradients(Astar, Bstar, Q, R, K0, sigma_w, sigma_eta,
                                   num_pg_iters, horizon_length, step_size, proj_norm, baseline, rng)
    costs = np.array([np.inf if policy is None else
                      utils.LQR_cost(Astar, Bstar, policy, Q, R, sigma_w)
                      for policy in policies])

    return costs


def dfo_wrapper(args):
    Astar, Bstar, Q, R, K0, sigma_w, sigma_eta, num_dfo_iters, horizon_length, step_size, norm_project, seed = args

    rng = np.random.RandomState(seed)

    policies = dfo.dfo(Astar, Bstar, Q, R, K0, sigma_w, sigma_eta, num_dfo_iters, horizon_length, step_size, norm_project, rng)

    costs = np.array([np.inf if policy is None else
                      utils.LQR_cost(Astar, Bstar, policy, Q, R, sigma_w)
                      for policy in policies])

    return costs


def policy_iteration_wrapper(args):
    Astar, Bstar, Q, R, K0, sigma_w, sigma_eta, num_pi_iters, horizon_length, mu, L, seed = args

    rng = np.random.RandomState(seed)

    policies = policy_iteration.policy_iteration(Astar, Bstar, Q, R, K0, sigma_w, sigma_eta, num_pi_iters, horizon_length, mu, L, rng)

    costs = np.array([np.inf if policy is None else
                      utils.LQR_cost(Astar, Bstar, policy, Q, R, sigma_w)
                      for policy in policies])

    return costs


def policy_iteration_full_wrapper(args):
    Astar, Bstar, Q, R, K0, Kplay, sigma_w, sigma_eta, num_resets, horizon_length, num_PI_iters, mu, L, seed = args

    rng = np.random.RandomState(seed)

    policies = policy_iteration.policy_iteration_full(Astar, Bstar, Q, R, K0, Kplay, sigma_w, sigma_eta, num_resets, horizon_length, num_PI_iters, mu, L, rng)

    costs = np.array([np.inf if policy is None else
                      utils.LQR_cost(Astar, Bstar, policy, Q, R, sigma_w)
                      for policy in policies])

    return costs


def uniform_search_wrapper(args):
    Astar, Bstar, Q, R, sigma_w, num_iters, horizon_length, ball_width, seed = args

    rng = np.random.RandomState(seed)

    policies = uniform_search.uniform_search(Astar, Bstar, Q, R, sigma_w, num_iters, horizon_length, ball_width, rng)

    costs = np.array([np.inf if policy is None else
                      utils.LQR_cost(Astar, Bstar, policy, Q, R, sigma_w)
                      for policy in policies])

    return costs


def sarsa_wrapper(args):
    Astar, Bstar, Q, R, S0, sigma_w, sigma_eta, num_outer_iters, num_inner_iters, step_size, mu, L, seed = args

    rng = np.random.RandomState(seed)

    policies = qlearning.sarsa(Astar, Bstar, Q, R, S0, sigma_w, sigma_eta, num_outer_iters, num_inner_iters, step_size, mu, L, rng)

    costs = np.array([np.inf if policy is None else
                      utils.LQR_cost(Astar, Bstar, policy, Q, R, sigma_w)
                      for policy in policies])

    return costs


def nominal_experiment(fname):

    n_trials = 100

    Astar, Bstar, Q, R = systems.example1()

    sigma_w = 1
    sigma_eta = 1

    num_iters = 10000
    horizon_length = 100

    n, d = Bstar.shape

    with mp.Pool(mp.cpu_count()) as p:
        args = (Astar, Bstar, Q, R, np.zeros((d, n)), sigma_w, sigma_eta,
                num_iters, horizon_length)
        results = np.array(list(p.map(nominal_wrapper, [args + (np.random.randint(0xFFFFFFFF),) for _ in range(n_trials)])))

    np.savez(fname, results)


def pg_experiment(fname, sigma_eta, step_size, baseline):

    n_trials = 100

    Astar, Bstar, Q, R = systems.example1()
    _, Kstar = utils.dlqr(Astar, Bstar, Q, R)

    sigma_w = 1
    #sigma_eta = 1
    #step_size = 1e-6

    num_pg_iters = 10000
    horizon_length = 100

    n, d = Bstar.shape

    with mp.Pool(mp.cpu_count()) as p:
        args = (Astar, Bstar, Q, R, np.zeros((d, n)), sigma_w, sigma_eta,
                num_pg_iters, horizon_length, step_size, 5 * np.linalg.norm(Kstar, ord='fro'), baseline)
        results = np.array(list(p.map(policy_gradients_wrapper, [args + (np.random.randint(0xFFFFFFFF),) for _ in range(n_trials)])))

    np.savez(fname, results)


def dfo_experiment(fname, sigma_eta, step_size):

    n_trials = 100

    Astar, Bstar, Q, R = systems.example1()
    _, Kstar = utils.dlqr(Astar, Bstar, Q, R)

    sigma_w = 1
    #sigma_eta = 1e-3
    #step_size = 1e-6

    num_dfo_iters = 5000
    horizon_length = 100

    n, d = Bstar.shape

    with mp.Pool(mp.cpu_count()) as p:
        args = (Astar, Bstar, Q, R, np.zeros((d, n)), sigma_w, sigma_eta,
                num_dfo_iters, horizon_length, step_size, 5 * np.linalg.norm(Kstar, ord='fro'))
        results = np.array(list(p.map(dfo_wrapper, [args + (np.random.randint(0xFFFFFFFF),) for _ in range(n_trials)])))

    np.savez(fname, results)


def policy_iteration_experiment(fname, num_pi_iters, horizon_length, clamp_upper):

    n_trials = 100
    Astar, Bstar, Q, R = systems.example1()
    Vstar, Kstar = utils.dlqr(Astar, Bstar, Q, R)
    Astar_Bstar = np.hstack((Astar, Bstar))
    Qstar = scipy.linalg.block_diag(Q, R) + Astar_Bstar.T.dot(Vstar).dot(Astar_Bstar)

    sigma_w = 1
    sigma_eta = 1

    #num_pi_iters = 5
    #horizon_length = 100000

    n, d = Bstar.shape

    with mp.Pool(mp.cpu_count()) as p:
        args = (Astar, Bstar, Q, R, np.zeros((d, n)), sigma_w, sigma_eta,
                num_pi_iters, horizon_length, min(utils.lambda_min(Q), utils.lambda_min(R)),
                5 * utils.lambda_max(Qstar) if clamp_upper else 1e10)
        results = np.array(list(p.map(policy_iteration_wrapper, [args + (np.random.randint(0xFFFFFFFF),) for _ in range(n_trials)])))

    np.savez(fname, results)


def policy_iteration_full_experiment(fname, n_pi_iters):

    n_trials = 100
    n_resets = 1000
    horizon_length = 100
    #n_pi_iters = 10
    Astar, Bstar, Q, R = systems.example1()
    Vstar, Kstar = utils.dlqr(Astar, Bstar, Q, R)
    Astar_Bstar = np.hstack((Astar, Bstar))
    Qstar = scipy.linalg.block_diag(Q, R) + Astar_Bstar.T.dot(Vstar).dot(Astar_Bstar)

    sigma_w = 1
    sigma_eta = 1

    n, d = Bstar.shape

    with mp.Pool(mp.cpu_count()) as p:
        args = (Astar, Bstar, Q, R, np.zeros((d, n)), np.zeros((d, n)),
                sigma_w, sigma_eta,
                n_resets, horizon_length, n_pi_iters,
                min(utils.lambda_min(Q), utils.lambda_min(R)), 1e10)
        results = np.array(list(p.map(policy_iteration_full_wrapper, [args + (np.random.randint(0xFFFFFFFF),) for _ in range(n_trials)])))

    np.savez(fname, results)


def sarsa_experiment(fname):

    n_trials = 100
    num_outer_iters = 1000
    horizon_length = 100
    step_size = 1e-5
    Astar, Bstar, Q, R = systems.example1()
    Vstar, Kstar = utils.dlqr(Astar, Bstar, Q, R)
    Astar_Bstar = np.hstack((Astar, Bstar))
    Qstar = scipy.linalg.block_diag(Q, R) + Astar_Bstar.T.dot(Vstar).dot(Astar_Bstar)
    Q0 = scipy.linalg.block_diag(Q, R) + Astar_Bstar.T.dot(utils.solve_discrete_lyapunov(Astar, Q)).dot(Astar_Bstar)

    sigma_w = 1
    sigma_eta = 1

    n, d = Bstar.shape

    with mp.Pool(mp.cpu_count()) as p:
        args = (Astar, Bstar, Q, R, Q0,
                sigma_w, sigma_eta,
                num_outer_iters, horizon_length, step_size,
                min(utils.lambda_min(Q), utils.lambda_min(R)),
                5 * utils.lambda_max(Qstar))
        results = np.array(list(p.map(sarsa_wrapper, [args + (np.random.randint(0xFFFFFFFF),) for _ in range(n_trials)])))

    np.savez(fname, results)


def uniform_search_experiment(fname):

    n_trials = 100
    Astar, Bstar, Q, R = systems.example1()
    _, Kstar = utils.dlqr(Astar, Bstar, Q, R)

    sigma_w = 1

    num_iters = 10000
    horizon_length = 100

    n, d = Bstar.shape

    with mp.Pool(mp.cpu_count()) as p:
        args = (Astar, Bstar, Q, R, sigma_w,
                num_iters, horizon_length, 5 * np.linalg.norm(Kstar, ord='fro'))
        results = np.array(list(p.map(uniform_search_wrapper, [args + (np.random.randint(0xFFFFFFFF),) for _ in range(n_trials)])))

    np.savez(fname, results)



def main():

    start_time = time.time()
    nominal_experiment('data/nominal_experiment.npz')
    print("finished nominal experiment in {} seconds".format(time.time() - start_time))

    step_sizes = [1e-3, 1e-4, 1e-5, 1e-6]
    sigma_etas = [1, 1e-1, 1e-2, 1e-3]
    baselines = ['simple', 'value_function']

    for sigma_eta, step_size, baseline in it.product(sigma_etas, step_sizes, baselines):
        start_time = time.time()
        fname = 'data/pg_experiment_sigma_eta_{}_step_size_{}_baseline_{}.npz'.format(sigma_eta, step_size, baseline)
        pg_experiment(fname, sigma_eta, step_size, baseline)
        print("finished pg experiment in {} seconds".format(time.time() - start_time))

    for sigma_eta, step_size in it.product(sigma_etas, step_sizes):
        start_time = time.time()
        fname = 'data/dfo_experiment_sigma_eta_{}_step_size_{}.npz'.format(sigma_eta, step_size)
        dfo_experiment(fname, sigma_eta, step_size)
        print("finished dfo experiment in {} seconds".format(time.time() - start_time))

    n_pi_iters_values = [5, 10, 15]
    for n_pi_iters in n_pi_iters_values:
        start_time = time.time()
        fname = 'data/policy_iteration_full_experiment_n_pi_iters_{}.npz'.format(n_pi_iters)
        policy_iteration_full_experiment(fname, n_pi_iters)
        print("finished policy iteration full experiment in {} seconds".format(time.time() - start_time))

     policy_iteration_configs = [
         (1, 1000000),
         (2, 500000),
         (3, 333333),
         (4, 250000),
         (5, 200000),
         (6, 166666),
     ]

     for num_pi_iters, horizon_length in policy_iteration_configs:
         start_time = time.time()
         fname = 'data/policy_iteration_experiment_num_pi_iters_{}_horizon_length_{}_no_clamp_upper.npz'.format(num_pi_iters, horizon_length)
         policy_iteration_experiment(fname, num_pi_iters, horizon_length, False)
         print("finished policy iteration experiment in {} seconds".format(time.time() - start_time))

    start_time = time.time()
    sarsa_experiment('data/sarsa_experiment.npz')
    print("finished sarsa experiment in {} seconds".format(time.time() - start_time))

    start_time = time.time()
    uniform_search_experiment('data/uniform_search_experiment.npz')
    print("finished uniform search experiment in {} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    main()
