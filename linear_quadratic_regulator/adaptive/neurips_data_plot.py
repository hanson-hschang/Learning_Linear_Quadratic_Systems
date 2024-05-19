import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

class Data:
    def __init__(self, value: np.ndarray, time: np.ndarray):
        self.value = value
        self.time = time
        self.number_of_data = self.time.shape[0]
        self.min_value = np.max([np.min(value) for value in self.value])
        self.max_value = np.min([np.max(value) for value in self.value])

    def get_average_time_curve(
        self, 
        num: int, endpoint: bool = True, 
        spacing_type: Literal["linear", "log"] = "linear"
    ) -> tuple[np.ndarray, np.ndarray]:
        match spacing_type:
            case "linear":
                value = np.linspace(self.min_value, self.max_value, num=num, endpoint=endpoint)
            case "log":
                value = np.logspace(np.log10(self.min_value), np.log10(self.max_value), num=num, endpoint=endpoint)
            case _:
                raise ValueError("Invalid spacing_type")
        interpolated_time = self.get_interpolated_time(value)
        time_average = np.mean(interpolated_time, axis=0)
        return value, time_average
    
    def get_standard_deviation_time(
        self,
        value: np.ndarray,
    ) -> np.ndarray:
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
        self.raw_data = [None] * self.number_of_data_files
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

    def get_data(self, key: Literal["K_error", "cost"], reverse: bool = False) -> Data:
        data_time = self.benchmark_time[:, 1:]
        match key:
            case "K_error":
                data_value = self.benchmark_normalized_K_error[:, 1:]
            case "cost":
                data_value = self.benchmark_normalized_cost[:, 1:]
            case _:
                raise ValueError("Invalid key")
        
        if reverse:
            data_value = data_value[:, ::-1]
            data_time = data_time[:, ::-1]
        
        return Data(
            value=data_value, 
            time=data_time
        )

class DataPlot:
    def __init__(self, kwargs_figure: dict = {}, kwargs_axes: dict = {}, kwargs_plots: dict = {}):
        self.kwargs_figure = kwargs_figure
        self.kwargs_axes = kwargs_axes
        self.kwargs_plots = kwargs_plots
        self.fig = plt.figure(**self.kwargs_figure)
        self.axes = self.fig.subplots(**self.kwargs_axes)

    def plot_time_curves(
        self,
        value: np.ndarray,
        time: np.ndarray,
        **kwargs
    ) -> None:
        for i in range(value.shape[0]):
            self.axes.plot(value[i, :], time[i, :], **kwargs)

    def plot_average_time_curve(
        self,
        value: np.ndarray,
        time: np.ndarray,
        **kwargs
    ) -> None:
        kwargs.setdefault("color", self.kwargs_plots['line_color'])
        self.axes.plot(value, time, **kwargs)

    def plot_error_bar(
        self,
        value: np.ndarray,
        time: np.ndarray,
        time_std: np.ndarray,
        **kwargs
    ) -> None:
        kwargs.setdefault("color", self.kwargs_plots['marker_color'])
        self.axes.errorbar(value, time, yerr=time_std, **kwargs)
    
    def plot_error_cross(
        self,
        value: np.ndarray,
        time: np.ndarray,
        value_std: np.ndarray,
        time_std: np.ndarray,
        **kwargs
    ) -> None:
        kwargs.setdefault("color", self.kwargs_plots['default_color'])
        self.axes.errorbar(value, time, xerr=value_std, yerr=time_std, **kwargs)

    def set_plot_settings(
        self,
        xlabel: str,
        ylabel: str,
        xscale: Literal["linear", "log"] = "linear",
        yscale: Literal["linear", "log"] = "linear",
    ) -> None:
        self.axes.set_xlabel(xlabel, fontsize=self.kwargs_plots['fontsize'])
        self.axes.set_ylabel(ylabel, fontsize=self.kwargs_plots['fontsize'])
        self.axes.tick_params(axis='both', labelsize=self.kwargs_plots['fontsize'])
        self.axes.set_xscale(xscale)
        self.axes.set_yscale(yscale)
        self.axes.legend(fontsize=self.kwargs_plots['fontsize'])

def main():
    """
    Main function for plotting NeurIPS data.

    This function retrieves data from the dataset, performs calculations, and plots the results.

    Args:
        None

    Returns:
        None
    """

    # Load data
    neurips_data_directory_name = "neurips_data/"
    raw_data = RawData(directory_name=neurips_data_directory_name)

    # Process data
    data_K_error = raw_data.get_data(key="K_error", reverse=True)
    K_error, time_K_error = data_K_error.get_average_time_curve(num=50)
    markers_K_error = np.linspace(K_error[0], K_error[-1], num=7)[1:-1]
    time_at_markers_K_error, time_std_at_markers_K_error = data_K_error.get_standard_deviation_time(markers_K_error)

    data_cost = raw_data.get_data(key="cost", reverse=True)
    cost, time_cost = data_cost.get_average_time_curve(num=50)
    markers_cost = np.linspace(cost[0], cost[-1], num=7)[1:-1]
    time_at_markers_cost, time_std_at_markers_cost = data_cost.get_standard_deviation_time(markers_cost)


    # Plots settings
    kwargs_figure = {"figsize": (8, 6)}
    kwargs_axes = {"nrows": 1, "ncols": 1}
    kwargs_plots = {
        "fontsize": 18, 
        "default_color": "#1C6AB1",
        "line_color": "#D097BA", 
        "marker_color": "#874F8D",
    }

    # Plotting data K_error
    data_plot_K_error = DataPlot(
        kwargs_figure=kwargs_figure, 
        kwargs_axes=kwargs_axes,
        kwargs_plots=kwargs_plots
    )
    data_plot_K_error.plot_average_time_curve(
        K_error,
        time_K_error,
        label='benchmark'
    )
    data_plot_K_error.plot_error_bar(
        markers_K_error,
        time_at_markers_K_error,
        time_std_at_markers_K_error,
        fmt="o",
        markersize=5,
        capsize=5,
        capthick=2,
        elinewidth=2,
    )
    data_plot_K_error.plot_error_cross(
        markers_K_error,
        0.01*time_at_markers_K_error,
        0.003 * np.ones_like(markers_K_error),
        0.01*time_std_at_markers_K_error,
        fmt="o",
        markersize=5,
        capsize=5,
        capthick=2,
        elinewidth=2,
        label='dual-EnKF'
    )
    data_plot_K_error.set_plot_settings(
        xlabel="normalized error", 
        ylabel="time [sec]",
        xscale="linear",
        yscale="log",
    )

    # Plotting data cost
    data_plot_cost = DataPlot(
        kwargs_figure=kwargs_figure, 
        kwargs_axes=kwargs_axes,
        kwargs_plots=kwargs_plots
    )
    data_plot_cost.plot_average_time_curve(
        cost,
        time_cost,
        label='benchmark'
    )
    data_plot_cost.plot_error_bar(
        markers_cost,
        time_at_markers_cost,
        time_std_at_markers_cost,
        fmt="o",
        markersize=5,
        capsize=5,
        capthick=2,
        elinewidth=2,
    )
    data_plot_cost.plot_error_cross(
        markers_cost,
        0.01*time_at_markers_cost,
        0.001 * np.ones_like(markers_cost),
        0.01*time_std_at_markers_cost,
        fmt="o",
        markersize=5,
        capsize=5,
        capthick=2,
        elinewidth=2,
        label='dual-EnKF'
    )
    data_plot_cost.set_plot_settings(
        xlabel="normalized cost", 
        ylabel="time [sec]",
        xscale="linear",
        yscale="log",
    )

    plt.show()    

if __name__ == "__main__":
    main()