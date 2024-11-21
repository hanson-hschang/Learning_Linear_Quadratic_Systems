import os
import pickle
from re import L
import h5py
import numpy as np
from numpy.typing import NDArray
from typing import Literal, NamedTuple, List, Any, Optional

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
    dual_enkf: DualEnKFData
    benchmark: Optional[BenchmarkData] = None

def load_benchmark_data(
    directory_name: str,
    interpotate_num: int = 400,
) -> BenchmarkData:
    benchmark_raw_data = RawData(directory_name=directory_name)

    # Process data
    data_cost = benchmark_raw_data.get_data(key="cost", reverse=True)
    cost, time_cost, time_std_cost = data_cost.get_time_curve(num=interpotate_num)
    # time_at_markers_cost, time_std_at_markers_cost = data_cost.get_standard_deviation_time(markers_cost)
    
    return BenchmarkData(
        cost, time_cost, time_std_cost
    )

def load_dual_enkf_data_from_folder(
    folder_name: str,
    DVEC: List = [10,20,40], # [4,10,20,40,50] # 8 is not working
) -> NDArray[np.number]:
    T = 10
    
    D = len(DVEC)
    NSIM = 4 # 100, 500, 1e3, 5e3 particles
    a_stats_smd = np.zeros((D,NSIM,3,4,2)) # dimension, no of particles, lqg or leqg,  cost term l1 time, mean or std

    basefilename = (os.path.dirname(os.path.realpath(__file__)) + "/" + folder_name + "/")
    # filename2 = [str(dim) + "D/"+ str(dim) + "D"]

    for dimcount, dim in enumerate(DVEC):
        filename = basefilename + str(dim) + "D/"+ str(dim) + "D" + str(int(T)) + "T"
        filenamenpy = filename + "mean" + ".npy"
        meanfile = np.load(filenamenpy)
        a_stats_smd[dimcount,:,:,:,0] = meanfile
        filenamenpy = filename + "cov" + ".npy"
        covfile = np.load(filenamenpy)
        a_stats_smd[dimcount,:,:,:,1] = covfile
    return a_stats_smd


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

    data: NDArray[np.number] = load_dual_enkf_data_from_folder(folder_name=folder_name)
    data = data[:, :, cost_func_type[cost_func], ...]
    selected_particles = tuple(np.arange(data.shape[1])) if selected_particles is None else selected_particles
    time_mean = data[:, selected_particles, measurements_type["time"], statistics_type["mean"]].squeeze()
    time_std = data[:, selected_particles, measurements_type["time"], statistics_type["std"]].squeeze()
    K_error_mean = data[:, selected_particles, measurements_type[K_error_type[cost_func]], statistics_type["mean"]].squeeze()
    K_error_std = data[:, selected_particles, measurements_type[K_error_type[cost_func]], statistics_type["std"]].squeeze()
    cost_mean = data[:, selected_particles, measurements_type["cost"], statistics_type["mean"]].squeeze()
    cost_std = data[:, selected_particles, measurements_type["cost"], statistics_type["std"]].squeeze()
    return DualEnKFData(time_mean, time_std, K_error_mean, K_error_std, cost_mean, cost_std)

class L4DCDataSelector(NamedTuple):
    dual_enkf_folder_name: str
    cost_func: Literal["lqg", "leqg"]
    benchmark_directory_name: str
    selected_particles: tuple | None = None

def load_data(
    l4dc_data_selector: L4DCDataSelector
) -> L4DCData:
    dula_enkf_data = load_dual_enkf_data(
        folder_name=l4dc_data_selector.dual_enkf_folder_name,
        cost_func=l4dc_data_selector.cost_func,
        selected_particles=l4dc_data_selector.selected_particles,
    )

    # benchmark_data = load_benchmark_data(
    #     directory_name=l4dc_data_selector.benchmark_directory_name,
    # )
    return L4DCData(dula_enkf_data, None)

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
        dual_enkf_folder_name="L4DC_data/Results/SMDTime", # particle_list = [100, 500, 1000, 5000]
        cost_func="lqg",
        benchmark_directory_name="MPPI",
    )
    lqg_data = load_data(
        l4dc_data_selector=lqg_l4dc_data_selector,
    )
    
if __name__ == "__main__":
    main()