''' 
This is the simulatioin model for discrete-time finite-horizon zero-sum LQ games.
'''

import numpy as np 
import torch 
from scipy import linalg 



class ZO_LQ_model:

    def __init__(self, model_params, algo_params):
        
        # torch.cuda.set_device(0)

        '''
        Initialize parameters with 'params' input
        'H' is the 'N' in the paper, denoting horizon length
        M2 is the sample size for outer-loop 
        M1 is the sample size for inner-loop problem
        '''
        
        self.H = model_params["H"]
        # it is a list of At
        self.A = model_params["A"]
        # xt - m dimensional
        self.x_dim = model_params["x_dim"]
        # ut - d dimensional
        self.control_dim1 = model_params["control_dim1"]
        # wt - n dimensional
        self.control_dim2 = model_params["control_dim2"]
        self.dK = self.x_dim * self.control_dim1 * self.H
        self.dL = self.x_dim * self.control_dim2 * self.H

        self.compact_A = torch.cat((torch.cat((torch.zeros((self.x_dim, self.x_dim * self.H)), torch.zeros(self.x_dim, self.x_dim)), dim=1), 
                                    torch.cat((torch.block_diag(*self.A), torch.zeros((self.x_dim * self.H, self.x_dim))), dim=1)), dim=0)
        # np.block([[torch.zeros((self.x_dim, self.x_dim * self.H)), torch.zeros(self.x_dim, self.x_dim)], 
                                #    [torch.block_diag(*self.A), torch.zeros((self.control_dim1 * self.H, self.x_dim))]])
        self.B = model_params["B"]
        self.compact_B = torch.cat((torch.zeros(self.x_dim, self.control_dim1 * self.H), torch.block_diag(*self.B)), dim=0)
        # np.block([[torch.zeros(self.x_dim * self.control_dim1 * self.H)], [linalg.block_diag(*self.B)]])
        
        self.D = model_params["D"]
        self.compact_D = torch.cat((torch.zeros(self.x_dim, self.control_dim2 * self.H), torch.block_diag(*self.D)), dim=0)
        # np.block([[torch.zeros(self.x_dim * self.control_dim2 * self.H)], [linalg.block_diag(*self.D)]])
        
        # print("compact A: ", self.compact_A, ", compact B:", self.compact_B, ", compact D:", self.compact_D)

        self.Q = model_params["Q"]
        self.compact_Q = torch.block_diag(*self.Q)

        self.Ru = model_params["Ru"]
        self.compact_Ru = torch.block_diag(*self.Ru)
        self.Rw = model_params["Rw"]
        self.compact_Rw = torch.block_diag(*self.Rw)
        
        self.r1 = algo_params["r1"]
        self.r2 = algo_params["r2"]
        self.epsilon1 = algo_params["epsilon1"]
        self.epsilon2 = algo_params["epsilon2"]
        self.M1 = algo_params["M1"]
        self.M2 = algo_params["M2"]
        # variance of the noises
        self.variance = algo_params["variance"]
        self.sigma0 = [self.variance * torch.eye(self.x_dim) for _ in range(self.H+1)]
        self.compact_sigma0 = torch.block_diag(*self.sigma0)
        

        ''' 
        compute Nash equilibrium - use as the benchmark
        '''
        self.nash_K = [None for _ in range(self.H)]
        self.nash_L = [None for _ in range(self.H)]
        self.nash_P = [None for _ in range(self.H+1)]
        self.nash_P[self.H] = self.Q[self.H]
        for h in range(self.H-1, -1, -1):
            lambda_h = torch.eye(self.x_dim) + (self.B[h] @ torch.inverse(self.Ru[h]) @ torch.transpose(self.B[h], 0, 1) - 
                                                self.D[h] @ torch.inverse(self.Rw[h]) @ torch.transpose(self.D[h], 0, 1)) @ self.nash_P[h+1]
            self.nash_K[h] = torch.inverse(self.Ru[h]) @ torch.transpose(self.B[h], 0, 1) @ self.nash_P[h+1] @ torch.inverse(lambda_h) @ self.A[h]
            self.nash_L[h] = -torch.inverse(self.Rw[h]) @ torch.transpose(self.D[h], 0, 1) @ self.nash_P[h+1] @ torch.inverse(lambda_h) @ self.A[h]
            self.nash_P[h] = self.Q[h] + torch.transpose(self.A[h], 0, 1) @ self.nash_P[h+1] @ torch.inverse(lambda_h) @ self.A[h]
        
        nash_compact_P = torch.block_diag(*self.nash_P)
        self.nash_cost = torch.trace(nash_compact_P @ self.compact_sigma0.double())

        nash_H = self.compact_Rw - torch.transpose(self.compact_D, 0, 1) @ nash_compact_P @ self.compact_D
        nash_H = nash_H.numpy()
        nash_eigenvalues = np.linalg.eigvals(nash_H)
        self.nash_lambda = np.min(nash_eigenvalues)
        # print("nash cost: ", self.nash_cost) - 3.2330
        # print("nash lambda: ", self.nash_lambda) - 4.2860
    '''
    compute exact natural gradient, 2E for L and 2F for K
    '''

    def compute_K_ngrad(self, temp_K, temp_L):
        
        # we need to make the list temp_K, temp_L to a compact form
        # diag_Ks = tuple([temp_K[h] for h in range(self.H)])
        # diag_Ls = tuple([temp_L[h] for h in range(self.H)])
        # compact_temp_K = np.block([linalg.block_diag(*diag_Ks), np.zeros((self.control_dim1 * self.H, self.x_dim))])
        # compact_temp_L = np.block([linalg.block_diag(*diag_Ls), np.zeros((self.control_dim2 * self.H, self.x_dim))])
        # P = linalg.solve_discrete_lyapunov(torch.transpose(self.compact_A - self.compact_B @ compact_temp_K - self.compact_D @ compact_temp_L, 0, 1), 
        #                                    self.compact_Q + torch.transpose(compact_temp_K, 0, 1) @ self.compact_Ru @ compact_temp_K - torch.transpose(compact_temp_L, 0, 1) @ self.compact_Rw @ compact_temp_L)
        # F = (self.compact_Ru + torch.transpose(self.compact_B, 0, 1) @ P @ self.compact_B) @ compact_temp_K - torch.transpose(self.compact_B, 0, 1) @ P @ (self.compact_A - self.compact_D @ compact_temp_L)
        # # extract blocks of F and build a list, 2 * F
        # ng_blocks = []

        # Or we can start directly from blocks
        P_blocks = self.compute_P(temp_K, temp_L)
        ng_blocks_F = [None for _ in range(self.H)]
        for h in range(self.H):
            ng_blocks_F[h] = 2 * ((self.Ru[h] + torch.transpose(self.B[h], 0, 1) @ P_blocks[h+1] @ self.B[h]) @ temp_K[h] 
                                - torch.transpose(self.B[h], 0, 1) @ P_blocks[h+1] @ (self.A[h] - self.D[h] @ temp_L[h]))
        return ng_blocks_F

    def compute_L_ngrad(self, temp_K, temp_L):
        

        # diag_Ks = tuple([temp_K[h] for h in range(self.H)])
        # diag_Ls = tuple([temp_L[h] for h in range(self.H)])
        # compact_temp_K = np.block([linalg.block_diag(*diag_Ks), np.zeros((self.control_dim1 * self.H, self.x_dim))])
        # compact_temp_L = np.block([linalg.block_diag(*diag_Ls), np.zeros((self.control_dim2 * self.H, self.x_dim))])
        # P = linalg.solve_discrete_lyapunov(torch.transpose(self.compact_A - self.compact_B @ compact_temp_K - self.compact_D @ compact_temp_L, 0, 1),
        #                                     self.compact_Q + torch.transpose(compact_temp_K, 0, 1) @ self.compact_Ru @ compact_temp_K - torch.transpose(compact_temp_L, 0, 1) @ self.compact_Rw @ compact_temp_L)
        # E = (-self.compact_Rw +  torch.transpose(self.compact_D, 0, 1) @ P @ self.compact_D) @ compact_temp_L - torch.transpose(self.compact_D, 0, 1) @ P @ (self.compact_A - self.compact_B @ compact_temp_K)

        # # extract blocks of E and build a list, 2 * E
        # ng_blocks = []

        ''' 
        Start from iterative computations
        '''
        P_blocks = self.compute_P(temp_K, temp_L)
        ng_blocks_E = [None for _ in range(self.H)]
        for h in range(self.H):
            ng_blocks_E[h] = 2 * ((-self.Rw[h] + torch.transpose(self.D[h], 0, 1) @ P_blocks[h+1] @ self.D[h]) @ temp_L[h] 
                                - torch.transpose(self.D[h], 0, 1) @ P_blocks[h+1] @ (self.A[h] - self.B[h] @ temp_K[h]))
        return ng_blocks_E
    
    '''
    Natural gradient estimation in Kaiqing's work
    '''
    def est_K_ngrad_benchmark(self, temp_K, temp_L):
        # generate perturbations
        # for debug:
        # self.M2 = 5
        perturbed_temp_K = torch.zeros(self.M2, self.H, self.control_dim1, self.x_dim, dtype=torch.float64)
        # perturbed_temp_L = torch.zeros(self.M2, self.H, self.control_dim2, self.x_dim, dtype=torch.float64)
        perturbed_P = torch.zeros(self.M2, self.H+1, self.x_dim, self.x_dim, dtype=torch.float64)
        perturbations = torch.normal(0, 1, (self.M2, self.dK, 1), dtype = torch.float64)
        perturbations = torch.nn.functional.normalize(perturbations, p=2, dim=1) * self.r2
        
        covariance = [None for _ in range(self.H+1)]
        # reshape the perturbations to block-wise
        perturbations = torch.reshape(perturbations, (self.M2, self.H, self.control_dim1, self.x_dim))
        # perturbations2 = perturbations.clone()
        # generate two realizations of the noies: initial state and process noises, the noises are bounded within [-sqrt{3}, sqrt{3}] and with variance 1
        xi1 = [(torch.rand(self.M2, self.x_dim, 1, dtype=torch.float64) - 0.5) * 2 * np.sqrt(3 * self.variance) for _ in range(self.H+1)]
        xi2 = [(torch.rand(self.M2, self.x_dim, 1, dtype=torch.float64) - 0.5) * 2 * np.sqrt(3 * self.variance) for _ in range(self.H+1)]
        
        
        x2 = xi2[0]
        covariance[0] = x2 @ torch.transpose(x2, 1, 2)
        # torch.kron need torch >= 2.0.0  
        # perturbed_P[:, self.H, :, :] = torch.kron(torch.ones(self.M2, 1, 1), self.Q[self.H]) 
        perturbed_P[:, self.H, :, :] = torch.zeros(self.M2, self.x_dim, self.x_dim) + self.Q[self.H]
        # print("perturbed P self.H: ", perturbed_P[:, self.H, :, :])
        # add last cost first
        perturbed_cost = torch.diagonal(perturbed_P[:, self.H, :, :] @ xi1[self.H] @ torch.transpose(xi1[self.H], 1, 2), offset=0, dim1=1, dim2=2).sum(-1) 
        for h in range(self.H-1, -1, -1):
            
            # the summation computation is correct
            perturbed_temp_K[:, h, :, :] = temp_K[h] + perturbations[:, h, :, :] 
            # Riccati equation 
            perturbed_P[:, h, :, :] = self.Q[h] + torch.transpose(perturbed_temp_K[:, h, :, :], 1, 2) @ self.Ru[h] @ perturbed_temp_K[:, h, :, :] + torch.transpose(self.A[h] - self.B[h] @ perturbed_temp_K[:, h, :, :], 1, 2) @ \
                                            (perturbed_P[:, h+1, :, :] + perturbed_P[:, h+1, :, :] @ self.D[h] @ torch.inverse(self.Rw[h] - torch.transpose(self.D[h], 0, 1) @ perturbed_P[:, h+1, :, :] @ self.D[h]) @ torch.transpose(self.D[h], 0, 1) @ perturbed_P[:, h+1, :, :]) @ \
                                            (self.A[h] - self.B[h] @ perturbed_temp_K[:, h, :, :])
            
            # covariance matrix estimation
            perturbed_cost += torch.diagonal(perturbed_P[:, h, :, :] @ xi1[h] @ torch.transpose(xi1[h], 1, 2), offset=0, dim1=1, dim2=2).sum(-1) 
            
        for h in range(self.H):
            x2 = (self.A[h] - self.B[h] @ temp_K[h] - self.D[h] @ temp_L[h]) @ x2 + xi2[h+1]
            covariance[h+1] = x2 @ torch.transpose(x2, 1, 2)  

        # unsqueeze before repeat
        perturbed_cost = perturbed_cost.unsqueeze(1)
        perturbed_cost = perturbed_cost.repeat(1, self.H * self.control_dim1 * self.x_dim)
        perturbed_cost = torch.reshape(perturbed_cost, (self.M2, self.H, self.control_dim1, self.x_dim))
        # print("before weighted perturbations: ", perturbations)
        perturbations = perturbations * perturbed_cost
        # print("after weighted perturbation: ", perturbations)
        # print("perturbed cost: ", perturbed_cost)
        
        # # too slow
        # flag_cost = [0. for _ in range(5)]
        # for s in range(5):
        #     for h in range(self.H):
        #         perturbed_temp_K[s, h, :, :] = temp_K[h] + perturbations2[s, h, :, :]
        #         # perturbed_temp_L[s, h, :, :] = self.max_oracle(perturbed_temp_K[s, h, :, :])
        #         # here we use optimal_L as a substitution
        #     # In Kaiqing's work, they solve an inner-loop problem at each perturbation
        #     print("progress: sample ", s)
        #     temp_opt_L = self.compute_optL(perturbed_temp_K[s, :, :, :])
        #     temp_flag_P = self.compute_P(perturbed_temp_K[s, :, :, :], temp_opt_L)
        #     for h in range(self.H+1):
        #         flag_cost[s] += torch.trace(temp_flag_P[h] @ xi1[h][s] @ torch.transpose(xi1[h][s], 0, 1))
        #     # for h in range(self.H):
        #     #     perturbed_temp_L[s, h, :, :] = temp_opt_L[h]
        # print("iterativly computed costs: ", perturbed_cost[:5, :, :])
        # print("hahahahahha")
        # print("formular computed cost: ", flag_cost)

        
        
        # too slow
        # perturbed_cost = torch.transpose(x1, 1, 2) @ self.Q[0] @ x1 + torch.transpose(x1, 1, 2) @ torch.transpose(perturbed_temp_K[:, 0, :, :], 1, 2) @ self.Ru[0] @ perturbed_temp_K[:, 0, :, :] @ x1 - \
                                # torch.transpose(x1, 1, 2) @ torch.transpose(perturbed_temp_L[0], 0, 1) @ self.Rw[0] @ perturbed_temp_L[0] @ x1 
        # for h in range(self.H-1):

            # x1 = (self.A[h] - self.B[h] @ perturbed_temp_K[h] - self.D[h] @ perturbed_temp_L[h]) @ x1 + xi1[h+1]
            # perturbed_cost += torch.transpose(x1, 1, 2) @ self.Q[h+1] @ x1 + torch.transpose(x1, 1, 2) @ torch.transpose(perturbed_temp_K[:, h+1, :, :], 1, 2) @ self.Ru[h+1] @ perturbed_temp_K[:, h+1, :, :] @ x1 - \
                                # torch.transpose(x1, 1, 2) @ torch.transpose(perturbed_temp_L[h+1], 0, 1) @ self.Rw[h+1] @ perturbed_temp_L[h+1] @ x1 
            
        # last cost
        # perturbed_cost += torch.transpose(x1, 1, 2) @ self.Q[self.H] @ x1 
        # perturbed_cost = torch.squeeze(perturbed_cost)
        # for m in range(self.M2):
            # perturbations[m] = perturbations[m] * perturbed_cost[m]
        # print("after shape: ", perturbations.shape)
        # compute summation and times the perturbation vector
        # 1 / M2 \sum_{m=0}^{M2-1} d2 / r2 @ cost_M @ V_m
        #  sigma = 1 / M2 \sum_{m=0}^{M2-1} 
        # compute product with the inverse of the covariance matrix, covariance at H is not needed
        est_cov = [torch.sum(covariance[h], axis=0) / self.M2 for h in range(self.H)] 
        est_ng_K = [self.dK / (self.M2 * self.r2) * torch.sum(perturbations[:, h, :, :], axis=0) @ torch.inverse(est_cov[h]) for h in range(self.H)]
        # print("est ng K: ", est_ng_K)
        # return type is a list of blocks
        return est_ng_K


    ''' 
    Estimate the gradients
    '''
    def est_K_ngrad(self, temp_K, temp_L):
        
        # generate perturbations
        perturbed_temp_K = torch.zeros(self.M2, self.H, self.control_dim1, self.x_dim, dtype=torch.float64)
        perturbed_P = torch.zeros(self.M2, self.H+1, self.x_dim, self.x_dim, dtype=torch.float64)

        perturbations = torch.normal(0, 1, (self.M2, self.dK, 1), dtype = torch.float64)
        perturbations = torch.nn.functional.normalize(perturbations, p=2, dim=1) * self.r2
        covariance = [None for _ in range(self.H + 1)]
        # reshape the perturbations to block-wise
        perturbations = torch.reshape(perturbations, (self.M2, self.H, self.control_dim1, self.x_dim))
        # generate two realizations of the noies: initial state and process noises, the noises are bounded within [-sqrt{3}, sqrt{3}] and with variance 1
        xi1 = [(torch.rand(self.M2, self.x_dim, 1, dtype=torch.float64) - 0.5) * 2 * np.sqrt(3 * self.variance) for _ in range(self.H+1)]
        xi2 = [(torch.rand(self.M2, self.x_dim, 1, dtype=torch.float64) - 0.5) * 2 * np.sqrt(3 * self.variance) for _ in range(self.H+1)]

        x2 = xi2[0]
        covariance[0] = x2 @ torch.transpose(x2, 1, 2)
        # use the pre-initialized P matrices
        perturbed_P[:, self.H, :, :] = torch.zeros(self.M2, self.x_dim, self.x_dim) + self.Q[self.H]
        perturbed_cost = torch.diagonal(perturbed_P[:, self.H, :, :] @ xi1[self.H] @ torch.transpose(xi1[self.H], 1, 2), offset=0, dim1=1, dim2=2).sum(-1)
        # Lyapunov equation
        for h in range(self.H-1, -1, -1):
            perturbed_temp_K[:, h, :, :] = temp_K[h] + perturbations[:, h, :, :]
            perturbed_P[:, h, :, :] = self.Q[h] + torch.transpose(perturbed_temp_K[:, h, :, :], 1, 2) @ self.Ru[h] @ perturbed_temp_K[:, h, :, :] - \
                                        torch.transpose(temp_L[h], 0, 1) @ self.Rw[h] @ temp_L[h] + torch.transpose(self.A[h] - self.B[h] @ perturbed_temp_K[:, h, :, :] - self.D[h] @ temp_L[h], 1, 2) @ \
                                        perturbed_P[:, h+1, :, :] @ (self.A[h] - self.B[h] @ perturbed_temp_K[:, h, :, :] - self.D[h] @ temp_L[h])
            
            perturbed_cost += torch.diagonal(perturbed_P[:, h, :, :] @ xi1[h] @ torch.transpose(xi1[h], 1, 2), offset=0, dim1=1, dim2=2).sum(-1)

        
        for h in range(self.H):

            x2 = (self.A[h] - self.B[h] @ temp_K[h] - self.D[h] @ temp_L[h]) @ x2 + xi2[h+1]
            covariance[h+1] = x2 @ torch.transpose(x2, 1, 2)  
        
        # last cost
        perturbed_cost = perturbed_cost.unsqueeze(1)
        perturbed_cost = perturbed_cost.repeat(1, self.H * self.control_dim1 * self.x_dim)
        perturbed_cost = torch.reshape(perturbed_cost, (self.M2, self.H, self.control_dim1, self.x_dim))
        perturbations = perturbations * perturbed_cost
        
        # compute summation and times the perturbation vector
        # 1 / M2 \sum_{m=0}^{M2-1} d2 / r2 @ cost_M @ V_m
        #  sigma = 1 / M2 \sum_{m=0}^{M2-1} 
        # compute product with the inverse of the covariance matrix, covariance at H is not needed
        est_cov = [torch.sum(covariance[h], axis=0) / self.M2 for h in range(self.H)] 
        est_ng_K = [self.dK / (self.M2 * self.r2) * torch.sum(perturbations[:, h, :, :], axis=0) @ torch.inverse(est_cov[h]) for h in range(self.H)]
        # print("est ng K: ", est_ng_K)
        # return type is a list of blocks
        return est_ng_K

    def est_L_ngrad(self, temp_K, temp_L):

        # generate perturbations
       # generate perturbations
        perturbed_temp_L = torch.zeros(self.M1, self.H, self.control_dim2, self.x_dim, dtype=torch.float64)
        perturbed_P = torch.zeros(self.M1, self.H+1, self.x_dim, self.x_dim, dtype=torch.float64)

        perturbations = torch.normal(0, 1, (self.M1, self.dL, 1), dtype = torch.float64)
        perturbations = torch.nn.functional.normalize(perturbations, p=2, dim=1) * self.r2
        covariance = [None for _ in range(self.H + 1)]
        # reshape the perturbations to block-wise
        perturbations = torch.reshape(perturbations, (self.M1, self.H, self.control_dim2, self.x_dim))
        # generate two realizations of the noies: initial state and process noises, the noises are bounded within [-sqrt{3}, sqrt{3}] and with variance 1
        xi1 = [(torch.rand(self.M1, self.x_dim, 1, dtype=torch.float64) - 0.5) * 2 * np.sqrt(3 * self.variance) for _ in range(self.H+1)]
        xi2 = [(torch.rand(self.M1, self.x_dim, 1, dtype=torch.float64) - 0.5) * 2 * np.sqrt(3 * self.variance) for _ in range(self.H+1)]

        x2 = xi2[0]
        covariance[0] = x2 @ torch.transpose(x2, 1, 2)
        # use the pre-initialized P matrices
        perturbed_P[:, self.H, :, :] = torch.zeros(self.M1, self.x_dim, self.x_dim) + self.Q[self.H]
        perturbed_cost = torch.diagonal(perturbed_P[:, self.H, :, :] @ xi1[self.H] @ torch.transpose(xi1[self.H], 1, 2), offset=0, dim1=1, dim2=2).sum(-1)
        # Lyapunov equation
        for h in range(self.H-1, -1, -1):
            perturbed_temp_L[:, h, :, :] = temp_L[h] + perturbations[:, h, :, :]
            perturbed_P[:, h, :, :] = self.Q[h] + torch.transpose(temp_K[h], 0, 1) @ self.Ru[h] @ temp_K[:, h, :, :] - \
                                        torch.transpose(temp_L, 0, 1) @ self.Rw[h] @ temp_L + torch.transpose(self.A[h] - self.B[h] @ temp_K[h] - self.D[h] @ perturbed_temp_L[:, h, :, :], 1, 2) @ \
                                        perturbed_P[:, h+1, :, :] @ (self.A[h] - self.B[h] @ temp_K[h] - self.D[h] @ perturbed_temp_L[:, h, :, :])
            
            perturbed_cost += torch.diagonal(perturbed_P[:, h, :, :] @ xi1[h] @ torch.transpose(xi1[h], 1, 2), offset=0, dim1=1, dim2=2).sum(-1)

        for h in range(self.H):

            x2 = (self.A[h] - self.B[h] @ temp_K[h] - self.D[h] @ temp_L[h]) @ x2 + xi2[h+1]
            covariance[h+1] = x2 @ torch.transpose(x2, 1, 2)  
        
        # last cost
        perturbed_cost = perturbed_cost.unsqueeze(1)
        perturbed_cost = perturbed_cost.repeat(1, self.H * self.control_dim2 * self.x_dim)
        perturbed_cost = torch.reshape(perturbed_cost, (self.M2, self.H, self.control_dim2, self.x_dim))
        perturbations = perturbations * perturbed_cost
        
        # compute summation and times the perturbation vector
        # 1 / M2 \sum_{m=0}^{M2-1} d2 / r2 @ cost_M @ V_m
        #  sigma = 1 / M2 \sum_{m=0}^{M2-1} 
        # compute product with the inverse of the covariance matrix, covariance at H is not needed
        est_cov = [torch.sum(covariance[h], axis=0) / self.M2 for h in range(self.H)] 
        est_ng_K = [self.dK / (self.M2 * self.r2) * torch.sum(perturbations[:, h, :, :], axis=0) @ torch.inverse(est_cov[h]) for h in range(self.H)]
        # print("est ng K: ", est_ng_K)
        # return type is a list of blocks
        return est_ng_K



    def compute_optL(self, temp_K):
        # In this function, we compute the optimal L given K, this is basically solving a Riccati equation 

        # diag_Ks = tuple([temp_K[h] for h in range(self.H)])
        # compact_temp_K = torch.tensor(np.block([linalg.block_diag(*diag_Ks), np.zeros((self.control_dim1 * self.H, self.x_dim))]), dtype=torch.float64)
        opt_L_blocks = [None for _ in range(self.H)]
        P = [torch.zeros(self.x_dim, self.x_dim, dtype=torch.float64) for _ in range(self.H+1)]
        P[self.H] = self.Q[self.H]
        for h in range(self.H-1, -1, -1):
            P[h] = self.Q[h] + torch.transpose(temp_K[h], 0, 1) @ self.Ru[h] @ temp_K[h] + torch.transpose(self.A[h] - self.B[h] @ temp_K[h], 0, 1) @ \
                                            (P[h+1] + P[h+1] @ self.D[h] @ torch.inverse(self.Rw[h] - torch.transpose(self.D[h], 0, 1) @ P[h+1] @ self.D[h]) @ torch.transpose(self.D[h], 0, 1) @ P[h+1]) @ \
                                            (self.A[h] - self.B[h] @ temp_K[h])
        # P = linalg.solve_discrete_are(self.compact_A - self.compact_B @ compact_temp_K, self.compact_D, 
        #                               self.compact_Q + torch.transpose(compact_temp_K, 0, 1) @ self.compact_Ru @ compact_temp_K, -self.compact_Rw)
        for h in range(self.H):
            opt_L_blocks[h] = torch.inverse(-self.Rw[h] + torch.transpose(self.D[h], 0, 1) @ P[h+1] @ self.D[h]) @ torch.transpose(self.D[h], 0, 1) @ P[h+1] @ (self.A[h] - self.B[h] @ temp_K[h])
        # opt_L = torch.inverse(-self.compact_Rw + torch.transpose(self.compact_D, 0, 1) @ P @ self.compact_D) @ torch.transpose(self.compact_D, 0, 1) @ P @ (self.compact_A - self.compact_B @ compact_temp_K)

        # opt_L_blocks = [opt_L[h * self.control_dim2 : (h+1) * self.control_dim2, h * self.x_dim: (h+1) * self.x_dim] for h in range(self.H)]
        # print("opt L:", opt_L)
        # print("opt L blocks:", opt_L_blocks)
        return opt_L_blocks

       
    
    def compute_cost(self, temp_K, temp_L):
        # both K and L are blocks

        P_blocks = self.compute_P(temp_K, temp_L)
        temp_compact_P = torch.block_diag(*P_blocks)
        temp_cost = torch.trace(temp_compact_P @ self.compact_sigma0.double())

        return temp_cost


    def compute_lambda(self, temp_K, temp_L):
        # both K and L are block list
        # lambda_min(Rw - D^T P D)
        P_blocks = self.compute_P(temp_K, temp_L)
        temp_compact_P = torch.block_diag(*P_blocks)

        temp_H = self.compact_Rw - torch.transpose(self.compact_D, 0, 1) @ temp_compact_P @ self.compact_D
        temp_H = temp_H.numpy()
        temp_eigenvalues = np.linalg.eigvals(temp_H)
        temp_min_eigv = np.min(temp_eigenvalues)
       
        return temp_min_eigv


    def compute_P(self, temp_K, temp_L):

        P_blocks = [None for _ in range(self.H+1)]
        P_blocks[self.H] = self.Q[self.H]
        for h in range(self.H-1, -1, -1):
            P_blocks[h] = torch.transpose(self.A[h] - self.B[h] @ temp_K[h] - self.D[h] @ temp_L[h], 0, 1) @ P_blocks[h+1] @ (self.A[h] - self.B[h] @ temp_K[h] - self.D[h] @ temp_L[h]) \
                            + self.Q[h] + torch.transpose(temp_K[h], 0, 1) @ self.Ru[h] @ temp_K[h] - torch.transpose(temp_L[h], 0, 1) @ self.Rw[h] @ temp_L[h]

        
        # print("P blocks: ", P_blocks)
        # print("compare with Lyapunov function: ")
        # temp_compact_K = torch.cat((torch.block_diag(*temp_K), torch.zeros((self.control_dim1 * self.H, self.x_dim))), dim=1)
        # temp_compact_L = torch.cat((torch.block_diag(*temp_L), torch.zeros((self.control_dim2 * self.H, self.x_dim))), dim=1)
        # benchmark_P = linalg.solve_discrete_lyapunov(torch.transpose(self.compact_A - self.compact_B @ temp_compact_K - self.compact_D @ temp_compact_L, 0, 1), self.compact_Q + torch.transpose(temp_compact_K, 0, 1) @ self.compact_Ru @ temp_compact_K 
                                                    #  - torch.transpose(temp_compact_L, 0, 1) @ self.compact_Rw @ temp_compact_L)
        # print("benchmark P: ", benchmark_P)
        return P_blocks


    def compute_avg_ng(self, temp_K, T, prior_avg):

        temp_optL = self.compute_optL(temp_K)
        # norm function should be able to compute norm of a list
        temp_Kngrad = self.compute_K_ngrad(temp_K, temp_optL)

        temp_ng_grad_norm = torch.tensor(0., dtype=torch.float64)
        num_blocks = len(temp_Kngrad)
        for i in range(num_blocks):
            temp_ng_grad_norm += torch.norm(temp_Kngrad[i], p='fro')

        temp_ng_avg = (prior_avg * (T-1) + temp_ng_grad_norm**2) / T 

        return temp_ng_avg
