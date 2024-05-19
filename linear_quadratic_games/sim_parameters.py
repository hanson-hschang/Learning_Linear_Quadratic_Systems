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
    # "memory_length": 2000,
    # 10 is for debugging
    "memory_length": 10000,
    # "benchmark_algo": int(sys.argv[15]),
    "benchmark_algo": 0
}
