
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
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

logging.basicConfig(level=logging.CRITICAL)
# logging.basicConfig(
#     level=logging.DEBUG, 
#     filename="regret_comparison.log", 
#     filemode="w",
#     format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
# )

# PARAMETERS
rng = np.random
horizon = 10000
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

example = laplacian_dynamics() # unstable_dynamics()
A_star, B_star, K_init, Q, R, prime_horizon, prime_excitation, sigma_excitation, sigma_w = example

prime_horizon = 2000
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
    regret = np.array([env.step(rng) for _ in range(horizon)])
    env.complete_epoch(rng)
    err, cost = env.get_statistics(iteration_based=True)
    return regret, err, cost

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

strategies = ['optimal', 'nominal', 'LSPI', 'MFLQ']
list_of_results = []
start_time = time.time()
for method in strategies:
    regrets = []
    errors = []
    costs = []
    seeds = []
    bad_invocations = 0
    for i in range(trials_per_method):
        logging.info(method + " with trial No. " + str(i+1))
        seed = np.random.randint(0xFFFFFFFF)
        try:
            reg, err, cost = run_one_trial(ctor[method], seed, prime_fixed=True)
            regrets.append(reg)
            errors.append(err)
            costs.append(cost)
            seeds.append(seed)
        except Exception as e:
            traceback.print_exc()
            bad_invocations += 1
    list_of_results.append((np.array(regrets), np.array(errors), np.array(costs), np.array(seeds), bad_invocations))

print("finished execution in {} seconds".format(time.time() - start_time))

def get_errorbars(regrets, q=10, percent_bad=0):
    median = np.percentile(regrets, q=50, axis=0)
    low10 = np.percentile(regrets, q=q, axis=0)
    high90 = np.percentile(regrets, q=100-q, axis=0)
    return median, low10, high90

def plot_list_medquantile(datalist, legendlist=None, xlabel=None, ylabel=None, semilogy=False, 
                          loc='upper left', alpha=0.25, figsize=(8,6)):
    rgblist = sns.color_palette(n_colors=len(datalist))
    plt.figure(figsize=figsize)
    for idx, data in enumerate(datalist):
        median, lower, higher = data
        if semilogy:
            plt.semilogy(range(len(median)), median, color=rgblist[idx], label=legendlist[idx], markersize=8)
        else:
            plt.plot(range(len(median)), median, color=rgblist[idx], label=legendlist[idx], markersize=8)
        plt.fill_between(np.array(np.arange(len(median))), median.astype(np.float64), 
                        higher.astype(np.float64), color=rgblist[idx], alpha=alpha)
    if legendlist is not None:
        plt.legend(loc=loc, fontsize=medfont, markerscale=1, framealpha=1, frameon=False)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=bigfont)
        plt.xticks(fontsize=smallfont)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=bigfont)
        plt.yticks(fontsize=smallfont)
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

regretlist = []
costs_list = []

strat_rearranged = strategies
res_rearranged = list_of_results
for name, result in zip(strat_rearranged, res_rearranged):
    regrets, errors, costs, _, bad_invocations = result
    print(name, "bad_invocations", bad_invocations)
    percent_bad = bad_invocations / trials_per_method * 100
    regretlist.append(get_errorbars(regrets, q=10, percent_bad=percent_bad))
    costs_list.append(get_errorbars(costs, q=10, percent_bad=percent_bad))
sns.set_palette("muted")
plot_list_medquantile(regretlist, legendlist=strat_rearranged, xlabel="iteration", ylabel="regret", loc="upper left")
plt.savefig('plots/regret_comparison_neurips19.pdf', bbox_inches='tight')
plot_list_medquantile(costs_list[1:], legendlist=strat_rearranged[1:], xlabel="iteration", 
                      ylabel="cost suboptimality", semilogy=True, loc="lower left")
plt.savefig('plots/cost_comparison_neurips19.pdf', bbox_inches='tight')