import numpy as np

class Riccati:
    def __init__(self, model_params: dict,):
        self.horizon_length = model_params["H"]
        self.x_dim = model_params["x_dim"]
        self.u_dim = model_params["control_dim1"]
        self.w_dim = model_params["control_dim2"]

        self.matrix_A_time_trajectory = np.zeros((self.horizon_length, self.x_dim, self.x_dim))
        self.matrix_B_time_trajectory = np.zeros((self.horizon_length, self.x_dim, self.u_dim))
        self.matrix_D_time_trajectory = np.zeros((self.horizon_length, self.x_dim, self.w_dim))

        self.matrix_Q_time_trajectory = np.zeros((self.horizon_length, self.x_dim, self.x_dim))
        self.matrix_Ru_time_trajectory = np.zeros((self.horizon_length, self.u_dim, self.u_dim))
        self.matrix_Rw_time_trajectory = np.zeros((self.horizon_length, self.w_dim, self.w_dim))
        
        self.matrix_P_time_trajectory = np.zeros((self.horizon_length, self.x_dim, self.x_dim))
        self.matrix_K_time_trajectory = np.zeros((self.horizon_length, self.u_dim, self.x_dim))
        self.matrix_L_time_trajectory = np.zeros((self.horizon_length, self.w_dim, self.x_dim))
        
        for time_step in range(self.horizon_length):
            self.matrix_A_time_trajectory[time_step, ...] = model_params["A"][time_step].numpy()
            self.matrix_B_time_trajectory[time_step, ...] = model_params["B"][time_step].numpy()
            self.matrix_D_time_trajectory[time_step, ...] = model_params["D"][time_step].numpy()
            self.matrix_Q_time_trajectory[time_step, ...] = model_params["Q"][time_step+1].numpy()
            self.matrix_Ru_time_trajectory[time_step, ...] = model_params["Ru"][time_step].numpy()
            self.matrix_Rw_time_trajectory[time_step, ...] = model_params["Rw"][time_step].numpy()

        self.compute_matrix_P_time_trajectory()
        self.compute_matrix_K_time_trajectory()
        self.compute_matrix_L_time_trajectory()

    def get_gain_K_time_trajectory(self,) -> np.ndarray:
        return self.matrix_K_time_trajectory.copy()
    
    def get_gain_L_time_trajectory(self,) -> np.ndarray:
        return self.matrix_L_time_trajectory.copy()
    
    def dynamics_of_P(self, matrix_P: np.ndarray, time_step: int) -> np.ndarray:
        matrix_A = self.matrix_A_time_trajectory[time_step]
        matrix_B = self.matrix_B_time_trajectory[time_step]
        matrix_D = self.matrix_D_time_trajectory[time_step]
        matrix_Q = self.matrix_Q_time_trajectory[time_step]
        matrix_Ru = self.matrix_Ru_time_trajectory[time_step]
        matrix_Rw = self.matrix_Rw_time_trajectory[time_step]

        matrix_Lambda = np.identity(self.x_dim) + (
            matrix_B @ np.linalg.inv(matrix_Ru) @ matrix_B.T - 
            matrix_D @ np.linalg.inv(matrix_Rw) @ matrix_D.T
        ) @ matrix_P
        matrix_P = matrix_Q + matrix_A.T @ matrix_P @ np.linalg.inv(matrix_Lambda) @ matrix_A
        return matrix_P

    def compute_matrix_P_time_trajectory(self,):
        self.matrix_P_time_trajectory[-1, ...] = self.matrix_Q_time_trajectory[-1, ...].copy()
        for time_step in range(self.horizon_length-1):
            self.matrix_P_time_trajectory[-1-time_step-1] = self.dynamics_of_P(
                matrix_P=self.matrix_P_time_trajectory[-1-time_step],
                time_step=time_step
            )

    def compute_matrix_K_time_trajectory(self,):
        for time_step in range(self.horizon_length):
            matrix_A = self.matrix_A_time_trajectory[time_step]
            matrix_B = self.matrix_B_time_trajectory[time_step]
            matrix_D = self.matrix_D_time_trajectory[time_step]
            matrix_Ru = self.matrix_Ru_time_trajectory[time_step]
            matrix_Rw = self.matrix_Rw_time_trajectory[time_step]
            matrix_P = self.matrix_P_time_trajectory[time_step]
            matrix_Lambda = np.identity(self.x_dim) + (
                matrix_B @ np.linalg.inv(matrix_Ru) @ matrix_B.T - 
                matrix_D @ np.linalg.inv(matrix_Rw) @ matrix_D.T
            ) @ matrix_P
            self.matrix_K_time_trajectory[time_step] = np.linalg.inv(matrix_Ru) @ matrix_B.T @ matrix_P @ np.linalg.inv(matrix_Lambda) @ matrix_A

    def compute_matrix_L_time_trajectory(self,):
        for time_step in range(self.horizon_length):
            matrix_A = self.matrix_A_time_trajectory[time_step]
            matrix_B = self.matrix_B_time_trajectory[time_step]
            matrix_D = self.matrix_D_time_trajectory[time_step]
            matrix_Ru = self.matrix_Ru_time_trajectory[time_step]
            matrix_Rw = self.matrix_Rw_time_trajectory[time_step]
            matrix_P = self.matrix_P_time_trajectory[time_step]
            matrix_Lambda = np.identity(self.x_dim) + (
                matrix_B @ np.linalg.inv(matrix_Ru) @ matrix_B.T - 
                matrix_D @ np.linalg.inv(matrix_Rw) @ matrix_D.T
            ) @ matrix_P
            self.matrix_L_time_trajectory[time_step] = - np.linalg.inv(matrix_Rw) @ matrix_D.T @ matrix_P @ np.linalg.inv(matrix_Lambda) @ matrix_A

def main():
    from sim_parameters import temp_model_params, temp_algo_params
    riccati = Riccati(temp_model_params)
    matrix_K_time_trajectory = riccati.get_gain_K_time_trajectory()
    matrix_L_time_trajectory = riccati.get_gain_L_time_trajectory()
    
    from zo_lq_model import ZO_LQ_model
    model = ZO_LQ_model(temp_model_params, temp_algo_params)

    nash_K = np.zeros((riccati.horizon_length, riccati.u_dim, riccati.x_dim))
    nash_L = np.zeros((riccati.horizon_length, riccati.w_dim, riccati.x_dim))
    nash_P = np.zeros((riccati.horizon_length+1, riccati.x_dim, riccati.x_dim))
    for time_step in range(riccati.horizon_length):
        nash_K[time_step, ...] = model.nash_K[time_step].numpy()
        nash_L[time_step, ...] = model.nash_L[time_step].numpy()

    from matplotlib import pyplot as plt 

    fig = plt.figure()
    axes = fig.subplots(riccati.u_dim, riccati.x_dim)

    for i in range(riccati.u_dim):
        for j in range(riccati.x_dim):
            axes[i, j].plot(matrix_K_time_trajectory[:, i, j])
            axes[i, j].plot(nash_K[:, i, j])

    fig.suptitle("Riccati eq.")

    plt.show()


if __name__ == "__main__":
    main()


