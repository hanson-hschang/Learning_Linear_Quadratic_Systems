import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

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
            self.axes.plot(time[i, :], value[i, :], **kwargs)

    def plot_benchmark_time_curve(
        self,
        value: np.ndarray,
        time_mean: np.ndarray,
        time_std: np.ndarray,
        label: str,
        **kwargs
    ) -> None:
        kwargs.setdefault("color", self.kwargs_plots['mean_color'])
        self.axes.plot(time_mean, value, label=label, **kwargs)
        kwargs["color"] = self.kwargs_plots['std_color']
        self.axes.fill_betweenx(value, time_mean+time_std, time_mean-time_std, **kwargs)

    def plot_error_bar(
        self,
        value: np.ndarray,
        time: np.ndarray,
        time_std: np.ndarray,
        label: str,
        **kwargs
    ) -> None:
        kwargs.setdefault("color", self.kwargs_plots['default_color'])
        self.axes.errorbar(time, value, xerr=time_std, label=label, **kwargs)
    
    def plot_error_cross(
        self,
        value: np.ndarray,
        time: np.ndarray,
        value_std: np.ndarray,
        time_std: np.ndarray,
        label: str,
        **kwargs
    ) -> None:
        kwargs.setdefault("color", self.kwargs_plots['default_color'])
        self.axes.errorbar(value, time, xerr=value_std, yerr=time_std, label=label, **kwargs)

    def set_plot_settings(
        self,
        time_label: str,
        value_label: str,
        time_scale: Literal["linear", "log"] = "linear",
        value_scale: Literal["linear", "log"] = "linear",
    ) -> None:
        self.axes.set_xlabel(time_label, fontsize=self.kwargs_plots['fontsize'])
        self.axes.set_ylabel(value_label, fontsize=self.kwargs_plots['fontsize'])
        self.axes.tick_params(axis='both', labelsize=self.kwargs_plots['fontsize'])
        self.axes.set_xscale(time_scale)
        self.axes.set_yscale(value_scale)
        self.axes.grid(which='both', linestyle='--', linewidth=0.5)
        self.axes.legend(fontsize=self.kwargs_plots['fontsize'])

def create_plot(
    data: NeurIPSData,
    cost_func: Literal["lqg", "leqg"],
    savefig: bool = False
):
    # Plots settings
    kwargs_figure = {"figsize": (8, 6)}
    kwargs_axes = {"nrows": 1, "ncols": 1}
    kwargs_plots = {
        "fontsize": 18, 
        "default_color": "#1C6AB1",
        "mean_color": "#874F8D", 
        "std_color": "#D097BA",
    }

    # Plotting data K_error
    data_plot_K_error = DataPlot(
        kwargs_figure=kwargs_figure, 
        kwargs_axes=kwargs_axes,
        kwargs_plots=kwargs_plots,
    )
    data_plot_K_error.plot_benchmark_time_curve(
        data.benchmark.K_error,
        data.benchmark.time_K_error,
        data.benchmark.time_std_K_error,
        label='benchmark',
    )
    data_plot_K_error.plot_error_bar(
        data.dual_enkf.K_error_mean,
        data.dual_enkf.time_mean,
        data.dual_enkf.time_std,
        label='dual-EnKF',
        fmt="o",
        markersize=5,
        capsize=5,
        capthick=2,
        elinewidth=2,
    )
    data_plot_K_error.set_plot_settings(
        time_label="time [sec]",
        value_label="normalized error", 
        time_scale="log",
        value_scale="log",
    )
    if savefig:
        data_plot_K_error.fig.savefig("figs/"+cost_func+"_error.pdf")

    # Plotting data cost
    data_plot_cost = DataPlot(
        kwargs_figure=kwargs_figure, 
        kwargs_axes=kwargs_axes,
        kwargs_plots=kwargs_plots,
    )
    data_plot_cost.plot_benchmark_time_curve(
        data.benchmark.cost,
        data.benchmark.time_cost,
        data.benchmark.time_std_cost,
        label='benchmark',
    )
    data_plot_cost.plot_error_bar(
        data.dual_enkf.cost_mean,
        data.dual_enkf.time_mean,
        data.dual_enkf.time_std,
        label='dual-EnKF',
        fmt="o",
        markersize=5,
        capsize=5,
        capthick=2,
        elinewidth=2,
        color=kwargs_plots['default_color'],
    )
    data_plot_cost.set_plot_settings(
        time_label="time [sec]",
        value_label="normalized cost", 
        time_scale="log",
        value_scale="log",
    )
    if savefig:
        data_plot_K_error.fig.savefig("figs/"+cost_func+"_cost.pdf")

def main():
    pass

if __name__ == "__main__":
    main()