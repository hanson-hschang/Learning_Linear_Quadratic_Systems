import os
import pickle
from re import L
import numpy as np
from numpy.typing import NDArray
from typing import Literal, NamedTuple, List, Any

class Data:
    def __init__(self, value: np.ndarray, time: np.ndarray) -> None:
        self.value = value
        self.time = time
        self.number_of_data = self.time.shape[0]
        self.min_value = np.max([np.min(value) for value in self.value])
        self.max_value = np.min([np.max(value) for value in self.value])

    def get_time_curve(
        self, 
        num: int, 
        endpoint: bool = True, 
        spacing_type: Literal["linear", "log"] = "linear"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        match spacing_type:
            case "linear":
                value = np.linspace(self.min_value, self.max_value, num=num, endpoint=endpoint)
            case "log":
                value = np.logspace(np.log10(self.min_value), np.log10(self.max_value), num=num, endpoint=endpoint)
            case _:
                raise ValueError("Invalid spacing_type")
        interpolated_time = self.get_interpolated_time(value)
        time_average = np.mean(interpolated_time, axis=0)
        time_std = np.std(interpolated_time, axis=0)
        return value, time_average, time_std
    
    def get_standard_deviation_time(
        self,
        value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        interpolated_time = self.get_interpolated_time(value)
        time_average = np.mean(interpolated_time, axis=0)
        time_std = np.std(interpolated_time, axis=0)
        return time_average, time_std
    
    def get_interpolated_time(
        self, 
        value: np.ndarray,
    ) -> np.ndarray:
        interpolated_time = np.zeros((self.number_of_data, value.shape[0]))
        for i in range(self.number_of_data):
            interpolated_time[i, :] = np.interp(value, self.value[i, :], self.time[i, :])
        return interpolated_time

class RawData:
    def __init__(self, directory_name: str):
        self.directory_name = directory_name
        self.list_of_names_of_data_files = sorted([name for name in os.listdir(directory_name) if ".pk" in name])
        self.number_of_data_files = len(self.list_of_names_of_data_files)
        self.raw_data: List[Any] = [None] * self.number_of_data_files
        self.load_data()

        self.data_points_length = self.raw_data[0]["benchmark_time"].shape[0]

        self.benchmark_time = np.zeros((self.number_of_data_files, self.data_points_length))
        self.benchmark_normalized_K_error = np.zeros((self.number_of_data_files, self.data_points_length))
        self.benchmark_normalized_cost = np.zeros((self.number_of_data_files, self.data_points_length))

        for file_count, data in enumerate(self.raw_data):
            self.benchmark_time[file_count] = data["benchmark_time"]
            self.benchmark_normalized_K_error[file_count] = data["benchmark_normalized_K_error"]
            self.benchmark_normalized_cost[file_count] = data["benchmark_normalized_cost"]

    def load_data(self,) -> None:
        for file_count, data_file_name in enumerate(self.list_of_names_of_data_files):
            data_file = open(self.directory_name+data_file_name, "rb")
            self.raw_data[file_count] = pickle.load(data_file)

    def get_data(
        self, 
        key: Literal["K_error", "cost"], 
        reverse: bool = False, 
        start_index: int = 1
    ) -> Data:
        data_time = self.benchmark_time[:, start_index:]
        match key:
            case "K_error":
                data_value = self.benchmark_normalized_K_error[:, start_index:]
            case "cost":
                data_value = self.benchmark_normalized_cost[:, start_index:]
            case _:
                raise ValueError("Invalid key")
        
        if reverse:
            data_value = data_value[:, ::-1]
            data_time = data_time[:, ::-1]
        
        return Data(
            value=data_value, 
            time=data_time
        )

class BenchmarkData(NamedTuple):
    K_error: np.ndarray
    time_K_error: np.ndarray
    time_std_K_error: np.ndarray
    cost: np.ndarray
    time_cost: np.ndarray
    time_std_cost: np.ndarray

class DualEnKFData(NamedTuple):
    time_mean: np.ndarray
    time_std: np.ndarray
    K_error_mean: np.ndarray
    K_error_std: np.ndarray
    cost_mean: np.ndarray
    cost_std: np.ndarray

class L4DCData(NamedTuple):
    benchmark: BenchmarkData
    dual_enkf: DualEnKFData

def load_benchmark_data(
    directory_name: str,
    interpotate_num: int = 400,
    start_index: int = 1,
) -> BenchmarkData:
    benchmark_raw_data = RawData(directory_name=directory_name)

    # Process data
    data_K_error = benchmark_raw_data.get_data(key="K_error", reverse=True, start_index=start_index)
    K_error, time_K_error, time_std_K_error = data_K_error.get_time_curve(num=interpotate_num)
    # time_at_markers_K_error, time_std_at_markers_K_error = data_K_error.get_standard_deviation_time(markers_K_error)

    data_cost = benchmark_raw_data.get_data(key="cost", reverse=True, start_index=start_index)
    cost, time_cost , time_std_cost = data_cost.get_time_curve(num=interpotate_num)
    # time_at_markers_cost, time_std_at_markers_cost = data_cost.get_standard_deviation_time(markers_cost)
    
    return BenchmarkData(
        K_error, time_K_error, time_std_K_error,
        cost, time_cost, time_std_cost
    )

def load_dual_enkf_data_from_folder(
    folder_name: str,
) -> NDArray[np.number]:
    T=1 # [1, 10]
    DIMX = 3
    NSIM = 10 
    a_stats = np.zeros((2,2,NSIM,4,2)) # lqg or leqg, deterministic or random, 
    #  no of particles, cost terminal_error_gain(lqg) l1_gain(lqeg) time, 
    # mean or std

    basefilenames = ([[(folder_name + '/BRSys-LQG/'),
        (folder_name + '/RandomSys-LQG/')],
        [(folder_name + '/KZSys-LEQGP/'),
        (folder_name + '/RandomSys-LEQGP/')]])
    filename2 = [str(3) + "D/"+ 'LQG' + str(DIMX) + "D",str(3) + "D/" + 'LEQGP' + str(DIMX) + "D"]

    for i in range(0,2): # LQG or LEQG
        for j in range(0,1): # deterministic or random
            filename = basefilenames[i][j] + filename2[i]  + str(T) + "T"
            filenamenpy = filename + "mean" + ".npy"
            print(filenamenpy)
            meanfile = np.load(filenamenpy)
            # print(meanfile.shape)
            a_stats[i,j,:,:,0] = meanfile
            filenamenpy = filename + "cov" + ".npy"
            print(filenamenpy)
            covfile = np.load(filenamenpy)
            a_stats[i,j,:,:,1] = covfile
    return a_stats


def load_dual_enkf_data(
    folder_name: str, 
    cost_func: Literal["lqg", "leqg"], 
    system: Literal["deterministic", "random"] = "deterministic",
    selected_particles: tuple | None = None,
) -> DualEnKFData:
    cost_func_type = {"lqg": 0, "leqg": 1}
    system_type = {"deterministic": 0, "random": 1}
    measurements_type = {"cost": 0, "terminal": 1, "l1": 2,  "time": 3}
    statistics_type = {"mean": 0, "std": 1}
    K_error_type = {"lqg": "terminal", "leqg": "l1"}

    # data = np.load(folder_name)[cost_func_type[cost_func], system_type[system], ...]
    data: NDArray[np.number] = load_dual_enkf_data_from_folder(folder_name=folder_name)
    data = data[cost_func_type[cost_func], system_type[system], ...]
    selected_particles = tuple(np.arange(data.shape[0])) if selected_particles is None else selected_particles
    time_mean = data[selected_particles, measurements_type["time"], statistics_type["mean"]].squeeze()
    time_std = data[selected_particles, measurements_type["time"], statistics_type["std"]].squeeze()
    K_error_mean = data[selected_particles, measurements_type[K_error_type[cost_func]], statistics_type["mean"]].squeeze()
    K_error_std = data[selected_particles, measurements_type[K_error_type[cost_func]], statistics_type["std"]].squeeze()
    cost_mean = data[selected_particles, measurements_type["cost"], statistics_type["mean"]].squeeze()
    cost_std = data[selected_particles, measurements_type["cost"], statistics_type["std"]].squeeze()
    return DualEnKFData(time_mean, time_std, K_error_mean, K_error_std, cost_mean, cost_std)

class L4DCDataSelector(NamedTuple):
    dual_enkf_folder_name: str
    cost_func: Literal["lqg", "leqg"]
    benchmark_directory_name: str
    selected_particles: tuple | None = None
    start_index: int = 1

def load_data(
    l4dc_data_selector: L4DCDataSelector
) -> L4DCData:
    dula_enkf_data = load_dual_enkf_data(
        folder_name=l4dc_data_selector.dual_enkf_folder_name,
        cost_func=l4dc_data_selector.cost_func,
        selected_particles=l4dc_data_selector.selected_particles,
    )

    benchmark_data = load_benchmark_data(
        directory_name=l4dc_data_selector.benchmark_directory_name,
        start_index=l4dc_data_selector.start_index,
    )
    return L4DCData(benchmark_data, dula_enkf_data)

def load_smd_data(
    file_name: str,
    number_of_dimension: Literal[4, 10, 20],
    cost_func: Literal["lqg", "leqg+", "leqg-"], 
    selected_particles: tuple | None = None,
) -> DualEnKFData:
    number_of_dimension_type = {4: 0, 10: 1, 20: 2}
    cost_func_type = {"lqg": 0, "leqg+": 1, "leqg-": 2}
    measurements_type = {"cost": 0, "terminal": 1, "l1": 2,  "time": 3}
    statistics_type = {"mean": 0, "std": 1}
    K_error_type = {"lqg": "terminal", "leqg+": "l1", "leqg-": "l1"}

    data = np.load(file_name)[number_of_dimension_type[number_of_dimension], :, cost_func_type[cost_func], ...]
    selected_particles = np.arange(data.shape[0]) if selected_particles is None else selected_particles
    time_mean = data[selected_particles, measurements_type["time"], statistics_type["mean"]].squeeze()
    time_std = data[selected_particles, measurements_type["time"], statistics_type["std"]].squeeze()
    K_error_mean = data[selected_particles, measurements_type[K_error_type[cost_func]], statistics_type["mean"]].squeeze()
    K_error_std = data[selected_particles, measurements_type[K_error_type[cost_func]], statistics_type["std"]].squeeze()
    cost_mean = data[selected_particles, measurements_type["cost"], statistics_type["mean"]].squeeze()
    cost_std = data[selected_particles, measurements_type["cost"], statistics_type["std"]].squeeze()
    return DualEnKFData(time_mean, time_std, K_error_mean, K_error_std, cost_mean, cost_std)

def find_indices(
    array: np.ndarray,
    values: np.ndarray | float,
) -> np.ndarray:
    if isinstance(values, (int, float)):
        values_array = np.array([values])
    else:
        values_array = values.copy()
    indices = np.zeros(values_array.shape, dtype=int)
    for i, value in enumerate(values_array):
        indices[i] = np.argmin(np.abs(array-value))
    if isinstance(values, (int, float)):
        return indices[0]
    return indices

def main() -> None:
    
    lqg_l4dc_data_selector = L4DCDataSelector(
        dual_enkf_folder_name="L4DC_data/Results", # particle_list = [100, 500, 1000, 5000]
        cost_func="lqg",
        benchmark_directory_name="linear_quadratic_guassian/adaptive/neurips_data/prime_horizon_5000/",
        start_index=1,
    )
    lqg_data = load_data(
        l4dc_data_selector=lqg_l4dc_data_selector,
    )
    
    leqg_l4dc_data_selector = L4DCDataSelector(
        dual_enkf_folder_name="L4DC_data/Results", # particle_list = np.logspace(1, 3, 10)
        cost_func="leqg",
        benchmark_directory_name="linear_quadratic_games/neurips_data/",
        start_index=1,
    )
    leqg_data = load_data(
        l4dc_data_selector=leqg_l4dc_data_selector,
    )

if __name__ == "__main__":
    main()