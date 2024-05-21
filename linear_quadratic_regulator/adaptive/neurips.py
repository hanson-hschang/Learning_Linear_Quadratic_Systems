import os
import control
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pickle
import itertools as it
sns.set_style('ticks', {'font.family':'serif', 'font.serif':'Times New Roman'})

matplotlib.rcParams['text.usetex'] = True
sns.set(style="ticks")
plt.rc('font', family='serif')
plt.rc('font', serif='Times')

bigfont = 18
medfont = 14
smallfont = 18

import time
import logging
from multiprocessing import Pool, cpu_count
import traceback

import sys
sys.path.append('python/')

import utils
import examples
from optimal import OptimalStrategy
from nominal import NominalStrategy
from ofu import OFUStrategy
from sls import SLS_FIRStrategy, SLS_CommonLyapunovStrategy, sls_common_lyapunov, SLSInfeasibleException
from ts import TSStrategy
from lspi import LSPIStrategy
from mflq import MFLQStrategy

# logging.basicConfig(level=logging.CRITICAL)
logging.basicConfig(
    level=logging.INFO, 
    filename="neurips.log", 
    filemode="w",
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

bigfont = 18
medfont = 14
smallfont = 18

# PARAMETERS
rng = np.random
horizon = 100000
trials_per_method = 100

def set_up_example(A_star, B_star, qr_ratio, prime_horizon, prime_excitation, sigma_excitation=0.1):
    n,p = B_star.shape
    # design a stabilizing controller
    _, K_init = utils.dlqr(A_star, B_star, 1e-3*np.eye(n), np.eye(p))
    assert utils.spectral_radius(A_star + B_star.dot(K_init)) < 1
    Q = qr_ratio * np.eye(n)
    R = np.eye(p)
    sigma_w = 1
    return A_star, B_star, K_init, Q, R, prime_horizon, prime_excitation, sigma_excitation, sigma_w

def laplacian_dynamics(qr_ratio=1e1, prime_horizon=100, prime_excitation=1):
    A_star, B_star = examples.unstable_laplacian_dynamics()
    return set_up_example(A_star, B_star, qr_ratio, prime_horizon, prime_excitation)

def unstable_dynamics(qr_ratio=1e1, prime_horizon=250, prime_excitation=2):
    A_star, B_star = examples.transient_dynamics(diag_coeff=2, upperdiag=4)
    return set_up_example(A_star, B_star, qr_ratio, prime_horizon, prime_excitation, sigma_excitation=0.1)

def Kaiqing_example():
    A_star = np.array([
        [1, 0, -5],
        [-1, 1, 0],
        [0, 0, 1]
    ])
    B_star = np.array([
        [1, -10, 0],
        [0, 3, 1],
        [-1, 0, 2]
    ])
    K_init = -np.array([
        [-0.08, 0.35, 0.62],
        [-0.21, 0.19, 0.32],
        [-0.06, 0.10, 0.41]
    ])
    Q = np.array([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
    ])
    R = np.array([
        [4, -1, 0],
        [-1, 4, -2],
        [0, -2, 3]
    ])
    prime_horizon = 5
    prime_excitation = 1
    sigma_excitation = 0.01
    sigma_w = 1
    return A_star, B_star, K_init, Q, R, prime_horizon, prime_excitation, sigma_excitation, sigma_w

example = Kaiqing_example()  # laplacian_dynamics() # unstable_dynamics()
A_star, B_star, K_init, Q, R, prime_horizon, prime_excitation, sigma_excitation, sigma_w = example

# print(np.linalg.eig(A_star))
# quit()
print("\n A")
print(A_star)
print("\n B")
print(B_star)
print("\n Q")
print(Q)
print("\n R")
print(R)
print("\n")
print("prime_horizon", prime_horizon)
print("prime_excitation", prime_excitation)
print("sigma_excitation", sigma_excitation)

# K_init =  * np.linalg.inv(R).dot(B_star.T)
print("\n K_init")
print(K_init)

K_inf, _, _ = control.dlqr(A_star, B_star, Q, R)
K_inf = -K_inf
print("\n K_inf")
print(K_inf)

# J_inf = utils.LQR_cost(A_star, B_star, K_inf, Q, R, sigma_w)


def optimal_ctor():
    return OptimalStrategy(Q=Q, R=R, A_star=A_star, B_star=B_star, sigma_w=sigma_w)

def nominal_ctor():
    return NominalStrategy(Q=Q,
                          R=R,
                          A_star=A_star,
                          B_star=B_star,
                          sigma_w=sigma_w,
                          sigma_explore=sigma_excitation,
                          reg=1e-5,
                          epoch_multiplier=10, rls_lam=None)

def ofu_ctor():
    return OFUStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  reg=1e-5,
                  actual_error_multiplier=1, rls_lam=None)

def ts_ctor():
    return TSStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  reg=1e-5,
                  tau=500,
                  actual_error_multiplier=1, rls_lam=None)

def sls_fir_ctor():
    return SLS_FIRStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  sigma_explore=sigma_excitation,
                  reg=1e-5,
                  epoch_multiplier=10,
                  truncation_length=12,
                  actual_error_multiplier=1, rls_lam=None)

def sls_cl_ctor():
    return SLS_CommonLyapunovStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  sigma_explore=sigma_excitation,
                  reg=1e-5,
                  epoch_multiplier=10,
                  actual_error_multiplier=1, rls_lam=None)

def lspi_ctor():
    return LSPIStrategy(Q=Q,
                        R=R,
                        A_star=A_star,
                        B_star=B_star,
                        sigma_w=sigma_w,
                        sigma_explore=sigma_excitation,
                        epoch_multiplier=10,
                        num_PI_iters=3,
                        K_init=K_init)
def mflq_ctor():
    return MFLQStrategy(Q=Q,
                        R=R,
                        A_star=A_star,
                        B_star=B_star,
                        sigma_w=sigma_w,
                        sigma_explore=0.01, #sigma_excitation,
                        epoch_length=100,# 0.5 * horizon ** (3.0/ 4),
                        exploration_period=500, #horizon ** (1.0 / 4),
                        K_init=K_init)

prime_seed = 45727
def run_one_trial(new_env_ctor, seed, prime_fixed=False):
    rng = np.random.RandomState(seed)
    if prime_fixed: # reducing variance
        rng_prime = np.random.RandomState(prime_seed) 
    else:
        rng_prime = rng
    env = new_env_ctor()
    env.reset(rng_prime)
    env.prime(prime_horizon, K_init, prime_excitation, rng_prime)
    regret = [env.regret()]
    Kt_iterations = [K_init]
    time_iterations = [time.time()]
    for iter_step in range(horizon):
        regret.append(env.step(rng))
        Kt_iterations.append(env._Kt.copy())
        time_iterations.append(time.time())

    regret = np.array(regret)
    Kt_iterations = np.array(Kt_iterations)
    time_iterations = np.array(time_iterations)
    env.complete_epoch(rng)
    err, cost = env.get_statistics(iteration_based=True)
    return regret, err, cost, Kt_iterations, time_iterations

ctor = {
    'optimal': optimal_ctor,
    'nominal': nominal_ctor,
    'ofu': ofu_ctor,
    'ts': ts_ctor,
    'sls_fir': sls_fir_ctor,
    'sls_cl': sls_cl_ctor,
    'lspi': lspi_ctor,
    'mflq': mflq_ctor,
    'LSPI': lspi_ctor,
    'MFLQ': mflq_ctor,
}

# strategies = ['optimal', 'nominal', 'LSPI', 'MFLQ']
# strategies = ['LSPI']
# list_of_results = []
# start_time = time.time()


def calculate_K_error(Kt_iterations, K_inf):
    return np.array([np.linalg.norm(Kt - K_inf, ord='fro')/np.linalg.norm(K_inf, ord='fro') for Kt in Kt_iterations])

method = 'LSPI'
prime_horizon = 7500

for prime_horizon in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]:
    regrets = []
    errors = []
    costs = []
    seeds = []
    Kts = []
    times = []
    bad_invocations = 0
    start_time = time.time()
    for i in range(trials_per_method):
        logging.info(method + " with trial No. " + str(i+1))
        seed = np.random.randint(0xFFFFFFFF)
        try:
            reg, err, cost, Kt_iterations, time_iterations = run_one_trial(ctor[method], seed, prime_fixed=True)
            regrets.append(reg)
            errors.append(err)
            costs.append(cost)
            Kts.append(Kt_iterations)
            times.append(time_iterations)
            seeds.append(seed)
        except Exception as e:
            traceback.print_exc()
            bad_invocations += 1
    # result = (np.array(regrets), np.array(errors), np.array(costs), np.array(Kts), np.array(times), np.array(seeds), bad_invocations)

    print(
        "finished execution of {} prime horizon in {} seconds with {} many time trials".format(
            prime_horizon, time.time() - start_time, trials_per_method
        )
    )

    neuirps_data_folder_directory = "neurips_data/"
    if not os.path.isdir(neuirps_data_folder_directory):
        os.mkdir(neuirps_data_folder_directory)

    neuirps_data_folder_directory += "prime_horizon_" + str(prime_horizon) + "/"
    if not os.path.isdir(neuirps_data_folder_directory):
        os.mkdir(neuirps_data_folder_directory)


    for ind, (cost_iterations, Kt_iterations, time_iterations) in enumerate(zip(costs, Kts, times)):

        # print(ind)
        # print(cost_iterations.shape)
        # print(Kt_iterations.shape)
        # print(time_iterations.shape)

        K_error_iterations = calculate_K_error(Kt_iterations, K_inf)
        cost_iterations = np.insert(cost_iterations, 0, np.inf)
        time_iterations = time_iterations - time_iterations[0]

        min_K_error = K_error_iterations[0]
        min_cost = cost_iterations[0]
        for i in range(1, len(time_iterations)):
            if K_error_iterations[i] < min_K_error:
                min_K_error = K_error_iterations[i]
            else:
                K_error_iterations[i] = min_K_error
            if cost_iterations[i] < min_cost:
                min_cost = cost_iterations[i]
            else:
                cost_iterations[i] = min_cost

        # gain_K_ax.plot(
        #     K_error_iterations[1:],
        #     time_iterations[1:]
        # )
        # cost_ax.plot(
        #     cost_iterations[1:],
        #     time_iterations[1:]
        # )

        logging.info("Saving benchmark file with ind = " + str(ind))
        # benchmark algo list filenames
        data_file_name = neuirps_data_folder_directory + "Benjamin_ind_" + str(ind) + ".pk"
        
        data_file = open(data_file_name, "wb")

        pickle.dump(
            dict(
                benchmark_time=time_iterations,
                benchmark_normalized_K_error=K_error_iterations,
                benchmark_normalized_cost=cost_iterations
            ), 
            data_file
        )

        data_file.close()


# plt.rcParams['text.usetex'] = True
# fontsize = 18
# gain_K_fig = plt.figure()
# gain_K_ax = gain_K_fig.subplots(1, 1)
# cost_fig = plt.figure()
# cost_ax = cost_fig.subplots(1,1)


# gain_K_ax.set_xscale('log')
# gain_K_ax.set_xlabel("normalized error", fontsize=fontsize)
# gain_K_ax.set_ylabel("time [sec]", fontsize=fontsize)
# cost_ax.set_xscale('log')
# cost_ax.set_xlabel(r'$(J-J^*)/J^*$', fontsize=fontsize)
# cost_ax.set_ylabel("time [sec]", fontsize=fontsize)


# plt.show()