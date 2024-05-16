import numpy as np
import pickle 
import os
# from rsa import sign
import torch
from matplotlib import pyplot as plt 
from scipy import linalg
from zo_lq_model import ZO_LQ_model
import sys


'''
parameters in Kaiqing's work:
horizon length H
A, B, D, Q, Ru, Rw
K0, L0

'''
horizon_length = 5

temp_model_params = {
    
    # # xt - m dimensional
    # self.x_dim = model_params["x_dim"]
    # # ut - d dimensional
    # self.control_dim1 = model_params["control_dim1"]
    # # wt - n dimensional
    # self.control_dim2 = model_params["control_dim2"]
    # self.dK = self.x_dim * self.control_dim1 * self.H
    # self.dL = self.x_dim * self.control_dim2 * self.H
    # self.r1 = model_params["r1"]
    # self.r2 = model_params["r2"]
    # self.epsilon1 = model_params["epsilon1"]
    # self.M1 = model_params["M1"]
    # self.M2 = model_params["M2"]
    "H": horizon_length,
    
    "A": [torch.tensor(np.array([[1, 0, -5], [-1, 1, 0], [0, 0, 1]]), dtype=torch.float64) for _ in range(horizon_length)], 

    "B": [torch.tensor(np.array([[1, -10, 0], [0, 3, 1], [-1, 0, 2]]), dtype=torch.float64) for _ in range(horizon_length)],

    "D": [torch.tensor(np.array([[0.5, 0, 0], [0, 0.2, 0], [0, 0, 0.2]]), dtype=torch.float64) for _ in range(horizon_length)],
    
    "Q": [torch.tensor(np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]), dtype=torch.float64) for _ in range(horizon_length+1)],

    "Ru": [torch.tensor(np.array([[4, -1, 0], [-1, 4, -2], [0, -2, 3]]), dtype=torch.float64) for _ in range(horizon_length)],

    "Rw": [torch.tensor(5 * np.eye(3), dtype=torch.float64) for _ in range(horizon_length)],

    "x_dim": 3,

    "control_dim1": 3,

    "control_dim2": 3
}


'''
This is the main function of algorithms: nested natural gradient algorithm in model-free case. 
Both new nested natural gradient algorithm and the old nested natural gradient algorithm in Kaiqing's work
In order to show the convergence of the outer-loop algorithm, we probably can choose the exact solution/exact gradient
update for the inner-loop oracle
K & L are policies of two controllers of the systems
'''

def Simulation(model_params, algo_params):
    
    model = ZO_LQ_model(model_params, algo_params)
    
    inner_loop_model_free = int(algo_params["inner_loop_model_free"])
    outer_loop_model_free = int(algo_params["outer_loop_model_free"])
    exact_inner_loop = int(algo_params["exact_inner_loop"])
    benchmark_algo = int(algo_params["benchmark_algo"])
    K = [None for _ in range(model.H)]
    L = [None for _ in range(model.H)]
    for h in range(model.H):
        K[h] = torch.clone(algo_params["K0"][h])
        L[h] = torch.clone(algo_params["L0"][h])
    
    tau1 = algo_params["tau1"]
    tau2 = algo_params["tau2"]
    # to compare with Kaiqing's work, we also choose the cost and the \lambda_{\min}(R^w - D^TPD)
    memory_length = algo_params["memory_length"]
    initial_values_index = algo_params["initial_values_index"]
    cost_list = [np.nan for _ in range(memory_length)]
    cost_diff_list = [np.nan for _ in range(memory_length)]
    lambda_list = [np.nan for _ in range(memory_length)]
    # we also record the convergence measure - average of the gradient, only for the outer-loop iterations
    avg_ng_list = [np.nan for _ in range(memory_length)]

    signature  = str(memory_length) + "_" + str(tau1) + "_" + str(tau2) + "_" + str(model.M1) + "_" + str(model.M2) \
                    + "_" + str(model.epsilon1) + "_" + str(model.epsilon2)

    initial_val_dir = "initial_values_set" + str(initial_values_index) + "/" 
    
    if benchmark_algo: 
        initial_val_dir += "benchmark_algo/"
    
    if not os.path.isdir(initial_val_dir):
        os.mkdir(initial_val_dir)

    # benchmark is irrelevant to the initial values
    if not os.path.isdir("benchmarks/"):
        os.mkdir("benchmarks/")

    benchmarkK_filename = "benchmarks/K.pk"
    benchmarkK_file = open(benchmarkK_filename, "wb+")
    pickle.dump(model.nash_K, benchmarkK_file)
    benchmarkL_filename = "benchmarks/L.pk"
    benchmarkL_file = open(benchmarkL_filename, "wb+")
    pickle.dump(model.nash_L, benchmarkL_file)
    benchmark_cost_filename = "benchmarks/cost.pk"
    benchmark_cost_file = open(benchmark_cost_filename, "wb+")
    pickle.dump(model.nash_cost, benchmark_cost_file)
    benchmark_lambda_filename = "benchmarks/lambda.pk"
    benchmark_lambda_file = open(benchmark_lambda_filename, "wb+")
    pickle.dump(model.nash_lambda, benchmark_lambda_file)

    cost_list[0] = model.compute_cost(K, L)
    cost_diff_list[0] = cost_list[0] - model.nash_cost
    lambda_list[0] = model.compute_lambda(K, L)
    # avg_ng_list record natural gradients at (K_t, L(K_t)) - only update at outer-loop iterations
    avg_ng_list[0] = model.compute_avg_ng(K, 1, 0)
    # iteration flag
    iter_num = 0
    outer_loop_iter_num = 0

    # for t_out in range(T_outer):
    # print("memory length: ", memory_length)
    while avg_ng_list[outer_loop_iter_num] > model.epsilon2 and iter_num < memory_length-1:
        # for t_in in range(T_inner):
        
        # initialize L with zeros everytime
        L = [None for _ in range(model.H)]
        for h in range(model.H):
            L[h] = torch.clone(algo_params["L0"][h])
        
        temp_optL = model.compute_optL(K)
        temp_cost = model.compute_cost(K, L)
        temp_optimal_cost = model.compute_cost(K, temp_optL)

        cost_diff = temp_optimal_cost - temp_cost
       
        while cost_diff > model.epsilon1 and iter_num < memory_length-2:
            
            if not exact_inner_loop:

                if inner_loop_model_free:
                    L_ngrad = model.est_L_ngrad(K, L)
                else: 
                    L_ngrad = model.compute_L_ngrad(K, L)
                
                for h in range(model.H):
                    L[h] = L[h] + tau1 * L_ngrad[h] 
            else:
                L = model.compute_optL(K)

            ''' 
            record convergence measure for inner-loop problem
            '''
            iter_num += 1
            temp_cost = model.compute_cost(K, L)
            cost_diff = temp_optimal_cost - temp_cost
            cost_list[iter_num] = temp_cost
            print("inner loop iteration:", iter_num, ": ", temp_cost)
            print("cost diff: ", cost_diff)
            lambda_list[iter_num] = model.compute_lambda(K, L)

        # now L is the estimated inner-loop solution given K
        if outer_loop_model_free:
            if benchmark_algo:
                K_ngrad = model.est_K_ngrad_benchmark(K, L)
            else:
                K_ngrad = model.est_K_ngrad(K, L)    
        
        else:
            K_ngrad = model.compute_K_ngrad(K, L)

        for h in range(model.H):
            K[h] = K[h] - tau2 * K_ngrad[h]

        iter_num += 1
        outer_loop_iter_num += 1
        ''' 
        record convergence measure
        '''
        cost_list[iter_num] = model.compute_cost(K, L)
        cost_diff_list[outer_loop_iter_num] = cost_list[iter_num] - model.nash_cost
        lambda_list[iter_num] = model.compute_lambda(K, L)
        avg_ng_list[outer_loop_iter_num] = model.compute_avg_ng(K, outer_loop_iter_num+1, avg_ng_list[outer_loop_iter_num-1])
        print("outer loop cost: ", iter_num, ": ", cost_list[iter_num])
        print("outer loop lambda: ", iter_num, ": ", lambda_list[iter_num])
        print("avg ng grad: ", avg_ng_list[outer_loop_iter_num])

    dir = ""

    if exact_inner_loop:
        if not os.path.isdir(initial_val_dir + "exact_inner_sol/"):
            os.mkdir(initial_val_dir + "exact_inner_sol/")
        if outer_loop_model_free:
            if not os.path.isdir(initial_val_dir + "exact_inner_sol/estimated_outer_grad/"):
                os.mkdir(initial_val_dir + "exact_inner_sol/estimated_outer_grad/")
            dir = initial_val_dir + "exact_inner_sol/estimated_outer_grad/"
        else: 
            if not os.path.isdir(initial_val_dir + "exact_inner_sol/exact_outer_grad/"):
                os.mkdir(initial_val_dir + "exact_inner_sol/exact_outer_grad/")
            dir = initial_val_dir + "exact_inner_sol/exact_outer_grad/"
            
    elif inner_loop_model_free:
        if not os.path.isdir(initial_val_dir + "estimated_inner_grad/"):
            os.mkdir(initial_val_dir + "estimated_inner_grad/")
        if outer_loop_model_free:
            if not os.path.isdir(initial_val_dir + "estimated_inner_grad/estimated_outer_grad/"):
                os.mkdir(initial_val_dir + "estimated_inner_grad/estimated_outer_grad/")
            dir = initial_val_dir + "estimated_inner_grad/estimated_outer_grad/"
        else: 
            if not os.path.isdir(initial_val_dir + "estimated_inner_grad/exact_outer_grad/"):
                os.mkdir(initial_val_dir + "estimated_inner_grad/exact_outer_grad/")
            dir = initial_val_dir + "estimated_inner_grad/exact_outer_grad/"

    else: 
        if not os.path.isdir(initial_val_dir + "exact_inner_grad/"):
            os.mkdir(initial_val_dir + "exact_inner_grad/")
        if outer_loop_model_free: 
            if not os.path.isdir(initial_val_dir + "exact_inner_grad/estimated_outer_grad/"):
                os.mkdir(initial_val_dir + "exact_inner_grad/estimated_outer_grad/")
            dir = initial_val_dir + "exact_inner_grad/estimated_outer_grad/"
        else:
            if not os.path.isdir(initial_val_dir + "exact_inner_grad/exact_outer_grad/"):
                os.mkdir(initial_val_dir + "exact_inner_grad/exact_outer_grad/")
            dir = initial_val_dir + "exact_inner_grad/exact_outer_grad/"
        

    cost_list_filename = dir + "/cost_list/" + signature + "_" + str(0) + ".pk"
    ind = 0
    while os.path.isfile(cost_list_filename):
        ind += 1
        cost_list_filename = dir + "/cost_list/" + signature + "_" + str(ind) + ".pk"
    
    lambda_list_filename = dir + "/lambda_list/" + signature + "_" + str(ind) + ".pk"
    avg_ng_list_filename = dir + "/avg_ng_list/" + signature + "_" + str(ind) + ".pk"
    cost_diff_list_filename = dir + "/cost_diff_list/" + signature + "_" + str(ind) + ".pk"

    if not os.path.isdir(dir + "/cost_list/"):
        os.mkdir(dir + "/cost_list/")
    if not os.path.isdir(dir + "/cost_diff_list/"):
        os.mkdir(dir + "/cost_diff_list/")
    if not os.path.isdir(dir + "/lambda_list/"):
        os.mkdir(dir + "/lambda_list/")
    if not os.path.isdir(dir + "/avg_ng_list/"):
        os.mkdir(dir + "/avg_ng_list/")


    cost_file = open(cost_list_filename, "wb+")
    cost_diff_file = open(cost_diff_list_filename, "wb+")
    lambda_file = open(lambda_list_filename, "wb+")
    avg_ng_file = open(avg_ng_list_filename, "wb+")

    pickle.dump(cost_list, cost_file)
    pickle.dump(cost_diff_list, cost_diff_file)
    pickle.dump(lambda_list, lambda_file)
    pickle.dump(avg_ng_list, avg_ng_file)
    print("debug cost list: ", cost_list)



# Input algorithm parameters
arg_num = len(sys.argv)

# sets of initial values for K, L
initial_K_values = [torch.tensor(np.array([[-0.08, 0.35, 0.62], [-0.21, 0.19, 0.32], [-0.06, 0.10, 0.41]]), dtype=torch.float64),
                    ]
initial_L_values = [torch.tensor(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), dtype=torch.float64),
                    ]

temp_algo_params = {

    # M1 is the sample size for solving inner-loop problem
    # M2 is the sample size for outer-loop iterations
    # "M1": int(sys.argv[1]), # 10^6
    # "M2": int(sys.argv[2]), # 5 x 10^5
    "M1": 10**6,
    # "M2": 250000,
    # for comparison plots
    "M2": 500000,
    # r1 is the perturbation radius for solving inner-loop problem
    # r2 is the perturbation radius for outer-loop iterations
    # "r1": float(sys.argv[3]), # 1
    # "r2": float(sys.argv[4]), # 0.08
    "r1": 1,
    "r2": 0.08,
    # tau1 is the step size for inner-loop problem
    # tau2 is the step size for outer-loop problem 
    # "tau1": float(sys.argv[5]), # 0.1
    # "tau2": float(sys.argv[6]), # 4.67 x 10^(-5)
    "tau1": 0.1,
    # "tau2": 2 * 10**(-3),
    # for comparison plots
    "tau2": 4.67 * 10**(-4),
    # inner-loop model free - use estimated natural gradients for inner-loop problem or not
    # outer-loop model free - use estimated natural gradients for outer-loop problem or not
    # exact inner loop - use exact solution for inner-loop problem or not
    # "inner_loop_model_free": int(sys.argv[7]),
    # "outer_loop_model_free": int(sys.argv[8]),
    # "exact_inner_loop": int(sys.argv[9]),
    "inner_loop_model_free": 0,
    "outer_loop_model_free": 1,
    "exact_inner_loop": 1,
    # epsilon 1 is the accuracy requirement for inner-loop problem
    # epsilon 2 is the accuracy requirement for outer-loop
    # "epsilon1": float(sys.argv[10]), # 10^{-4}
    # "epsilon2": float(sys.argv[11]), # 0.8
    "epsilon1": 10**(-4),
    "epsilon2": 0.8,
    # variance of the noises
    # "variance": float(sys.argv[12]), # 0.05
    "variance": 0.05,
    # hard-coded initial values
    # input initial values index
    # "initial_values_index": int(sys.argv[13]),
    "initial_values_index": 0,
    # "K0": [initial_K_values[int(sys.argv[13])] for _ in range(horizon_length)],
    # "L0": [initial_L_values[int(sys.argv[13])] for _ in range(horizon_length)],
    "K0": [initial_K_values[0] for _ in range(horizon_length)],
    "L0": [initial_L_values[0] for _ in range(horizon_length)],
    # maximum inner-loop iterations x outer-loop iterations
    # "memory_length": int(sys.argv[14]),
    "memory_length": 2000,
    # 10 is for debugging
    # "memory_length": 8,
    # "benchmark_algo": int(sys.argv[15]),
    "benchmark_algo": 0
}

temp_algo_params["benchmark_algo"] = 0
res = Simulation(temp_model_params, temp_algo_params) 
temp_algo_params["benchmark_algo"] = 1
benchmark_res = Simulation(temp_model_params, temp_algo_params)  

memory_length = temp_algo_params["memory_length"]
tau1 = temp_algo_params["tau1"]
tau2 = temp_algo_params["tau2"]
M1 = temp_algo_params["M1"]
M2 = temp_algo_params["M2"]
epsilon1 = temp_algo_params["epsilon1"]
epsilon2 = temp_algo_params["epsilon2"]

signature  = str(memory_length) + "_" + str(tau1) + "_" + str(tau2) + "_" + str(M1) + "_" + str(M2) \
                    + "_" + str(epsilon1) + "_" + str(epsilon2)


initial_val_dir = "initial_values_set" + str(temp_algo_params["initial_values_index"]) + "/" 
benchmark_initial_dir = "initial_values_set" + str(temp_algo_params["initial_values_index"]) + "/benchmark_algo/" 


if temp_algo_params["exact_inner_loop"]:
    # dir = initial_val_dir + "exact_inner_sol/"
    if temp_algo_params["outer_loop_model_free"]:
        dir = initial_val_dir + "exact_inner_sol/estimated_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "exact_inner_sol/estimated_outer_grad/"

    else: 
        dir = initial_val_dir + "exact_inner_sol/exact_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "exact_inner_sol/exact_outer_grad/"

elif temp_algo_params["inner_loop_model_free"]:
    # dir = initial_val_dir + "estimated_inner_grad/"
    if temp_algo_params["outer_loop_model_free"]:
        dir = initial_val_dir + "estimated_inner_grad/estimated_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "estimated_inner_grad/estimated_outer_grad/"
    else: 
        dir = initial_val_dir + "estimated_inner_grad/exact_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "estimated_inner_grad/exact_outer_grad/"

else: 
    # dir = initial_val_dir + "exact_inner_grad/"
    if temp_algo_params["outer_loop_model_free"]:
        dir = initial_val_dir + "exact_inner_grad/estimated_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "exact_inner_grad/estimated_outer_grad/"
    else: 
        dir = initial_val_dir + "exact_inner_grad/exact_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "exact_inner_grad/estimated_outer_grad/"

ind = 0

cost_list_filename = dir + "cost_list/" + signature + "_" + str(ind) + ".pk"
cost_diff_filename = dir + "cost_diff_list/" + signature + "_" + str(ind) + ".pk"
lambda_list_filename = dir + "lambda_list/" + signature + "_" + str(ind) + ".pk"
avg_ng_list_filename = dir + "avg_ng_list/" + signature + "_" + str(ind) + ".pk"

# benchmark algo list filenames
benchmark_cost_list_filename = benchmark_dir + "cost_list/" + signature + "_" + str(ind) + ".pk"
benchmark_cost_diff_filename = benchmark_dir + "cost_diff_list/" + signature + "_" + str(ind) + ".pk"
benchmark_lambda_list_filename = benchmark_dir + "lambda_list/" + signature + "_" + str(ind) + ".pk"
benchmark_avg_ng_list_filename = benchmark_dir + "avg_ng_list/" + signature + "_" + str(ind) + ".pk"

benchmark_cost_filename = "benchmarks/" + "cost.pk"
benchmark_lambda_filename = "benchmarks/" + "lambda.pk"

cost_list_file = open(cost_list_filename, "rb")
cost_diff_file = open(cost_diff_filename, "rb")
lambda_list_file = open(lambda_list_filename, "rb")
avg_ng_list_file = open(avg_ng_list_filename, "rb")

benchmark_cost_list_file = open(benchmark_cost_list_filename, "rb")
benchmark_cost_diff_file = open(benchmark_cost_diff_filename, "rb")
benchmark_lambda_list_file = open(benchmark_lambda_list_filename, "rb")
benchmark_avg_ng_list_file = open(benchmark_avg_ng_list_filename, "rb")

benchmark_cost_file = open(benchmark_cost_filename, "rb")
benchmark_lambda_file = open(benchmark_lambda_filename, "rb")

temp_cost_list = pickle.load(cost_list_file)
temp_cost_diff_list = pickle.load(cost_diff_file)
temp_lambda_list = pickle.load(lambda_list_file)
temp_avg_ng_list = pickle.load(avg_ng_list_file)

benchmark_cost_list = pickle.load(benchmark_cost_list_file)
benchmark_cost_diff_list = pickle.load(benchmark_cost_diff_file)
benchmark_lambda_list = pickle.load(benchmark_lambda_list_file)
benchmark_avg_ng_list = pickle.load(benchmark_avg_ng_list_file)

nash_cost = pickle.load(benchmark_cost_file)
nash_lambda = pickle.load(benchmark_lambda_file)

compare_plot = 1

# Already plotted the linear convergence part
plt.rcParams['text.usetex'] = True
# plot linear convergence
plt.figure()
if compare_plot:
    plt.plot(temp_cost_diff_list, label="Our algo")
    plt.plot(benchmark_cost_diff_list, linestyle='--', label="Benchmark algo")
    plt.yscale('log')
    plt.xlabel('Outer-loop iterations $t$', fontsize=18)
    plt.ylabel(r'$\mathcal{G}(\mathbf{K}_t,\mathbf{L}(\mathbf{K}_t))-\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$', fontsize=18)

    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.legend()
    plt.savefig(dir + signature + "/compare_cost_diff.pdf")
    
else: 
    plt.plot(temp_cost_diff_list, label='Our algo')
    plt.yscale('log')
    plt.xlabel('Outer-loop iterations $t$', fontsize=18)
    plt.ylabel(r'$\mathcal{G}(\mathbf{K}_t,\mathbf{L}(\mathbf{K}_t))-\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$', fontsize=18)

    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    
    plt.savefig(dir + signature + "/cost_diff.pdf")





# plot cost plot
plt.figure()
if compare_plot:

    plt.plot([x - nash_cost for x in temp_cost_list], label="Our algo")
    plt.plot([x - nash_cost for x in benchmark_cost_list], linestyle='--', label="Benchmark algo")
    # plt.axhline(y=nash_cost, linestyle='--', label=r'$\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$')
    plt.yscale('log')
    plt.xlabel(r'Total iterations of $(\mathbf{K}_t,\mathbf{L}_k)$', fontsize=18)
    plt.ylabel(r'$\mathcal{G}(\mathbf{K}_t,\mathbf{L}_k)-\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$', fontsize=18)
    
    plt.legend()
    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.savefig(dir + signature + "/compare_cost.pdf")
else:

    plt.plot(temp_cost_list, label='Our algorithm')
    # plt.axhline(y=nash_cost, linestyle='--', label=r'$\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$')
    plt.xlabel(r'Total iterations of $(\mathbf{K}_t,\mathbf{L}_k)$', fontsize=18)
    plt.ylabel(r'$\mathcal{G}(\mathbf{K}_t,\mathbf{L}_k)-\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$', fontsize=18)

    plt.legend()
    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.savefig(dir + signature + "/cost.pdf")
    



# plot lambda plot
plt.figure()
if compare_plot:

    plt.plot([nash_lambda - x for x in temp_lambda_list], label="Our algo")
    plt.plot([nash_lambda - x for x in benchmark_lambda_list], linestyle='--', label="Benchmark algo")
    # plt.axhline(y=nash_lambda, linestyle='--', label=r'$\lambda_{\min}(\mathbf{H}_{\mathbf{K}^*,\mathbf{L}^*})$')
    plt.xlabel(r'Total iterations of $(\mathbf{K}_t,\mathbf{L}_k)$', fontsize=18)
    plt.ylabel(r'$\lambda_{\min}(\mathbf{H}_{\mathbf{K}^*,\mathbf{L}^*})-\lambda_{\min}(\mathbf{H}_{\mathbf{K}_t,\mathbf{L}_k })$', fontsize=18)

    plt.legend()
    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.savefig(dir + signature + "/compare_lambda.pdf")

else: 

    plt.plot(nash_lambda - temp_lambda_list, label='Our algorithm')
    # plt.axhline(y=nash_lambda, linestyle='--', label=r'$\lambda_{\min}(\mathbf{H}_{\mathbf{K}^*,\mathbf{L}^*})$')

    plt.xlabel(r'Total iterations of $(\mathbf{K}_t,\mathbf{L}_k)$')
    plt.ylabel(r'$\lambda_{\min}(\mathbf{H}_{\mathbf{K}^*,\mathbf{L}^*})-\lambda_{\min}(\mathbf{H}_{\mathbf{K}_t,\mathbf{L}_k })$', fontsize=18)

    plt.legend()
    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.savefig(dir + signature + "/lambda.pdf")



# plot average gradient norm plot
plt.figure()
plt.yscale('log')
if compare_plot:

    plt.plot(temp_avg_ng_list, label="Our algo")
    plt.plot(benchmark_avg_ng_list, linestyle='--', label="Benchmark algo")
    plt.legend()
    plt.xlabel(r'Outer-loop iterations $t$', fontsize=18)
    plt.ylabel('Average natural gradient norm', fontsize=18)
    
    plt.savefig(dir + signature + "/compare_avg_ng_norm.pdf")
else: 

    plt.plot(temp_avg_ng_list)
    plt.xlabel(r'Outer-loop iterations $t$', fontsize=18)
    plt.ylabel('Average natural gradient norm', fontsize=18)
    # plt.legend(ncol=1, loc='upper right')

    # plt.savefig(dir + signature + "/avg_ng_norm.pdf", bbox="tight")
    plt.savefig(dir + signature + "/avg_ng_norm.pdf")