
import time
from tkinter import font
from turtle import color
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams.update({'font.family': 'serif'})
import matplotlib.patches as mpatches

from tool import PlanarPoint, PolarPoint
from L4DC_smd_data import L4DCData, L4DCDataSelector, load_data

class L4DCDataPlot:
    def __init__(
        self,
        data: L4DCData,
    ) -> None:
        self.data = data
        self.kwargs_figure = {"figsize": (8, 6)}
        self.kwargs_axes = {
            "nrows": 1, 
            "ncols": 1,
            "sharey": "row",
            "sharex": "col",
        }
        self.kwargs_plots =  {
            "fontsize": 18, 
            "default_color": "#1C6AB1",
            # "mean_color": "#874F8D", 
            # "std_color": "#D097BA",
            "std_color": "#874F8D",
        }
        self.fig = plt.figure(**self.kwargs_figure)
        self.ax = self.fig.subplots(**self.kwargs_axes)
        # self.ax_mppi = self.axes[0]
        # self.ax_dual_enkf = self.axes[1]
        self.colors = ['C0', 'C1', 'C6']
        self.number_of_dimensions = [10, 20, 40]

        self.kwargs_data = {
            "fmt": "o",
            "markersize": 5,
            "capsize": 5,
            "capthick": 2,
            "elinewidth": 2,
            "linewidth": 1,
        }

        self.set_axes()
        self.plot_dual_enkf_data(**self.kwargs_data)
        self.plot_benchmark_data(**self.kwargs_data)
        self.set_legend()

    # def set_colors(self, **kwargs,) -> None:
    #     cost_mean = self.data.dual_enkf.cost_mean
    #     number_of_dimension_options = cost_mean.shape[0]
    #     for i in range(number_of_dimension_options):
    #         self.colors.append("C" + str(i))

    def plot_dual_enkf_data(self, **kwargs) -> None:
        kwargs["fmt"] = "-" + kwargs["fmt"]
        ax: plt.Axes = self.ax
        cost_mean = self.data.dual_enkf.cost_mean
        cost_std = self.data.dual_enkf.cost_std
        # print(cost_std)
        # quit()
        cost_time = self.data.dual_enkf.time_mean
        number_of_dimension_options = cost_mean.shape[0]
        for i in range(number_of_dimension_options):
            ax.errorbar(
                x=cost_time[i],
                y=cost_mean[i]*100,
                yerr=cost_std[i]*100,
                color=self.colors[i],
                **kwargs,
                # fmt="o",
                # color=self.kwargs_plots["default_color"],
                # label="Dual EnKF",
            )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("error in cost (\%)")
        # ax.legend()

    def plot_benchmark_data(self, **kwargs) -> None:
        kwargs["fmt"] = "--" + kwargs["fmt"]

        ax: plt.Axes = self.ax
        cost_mean = self.data.benchmark.cost_mean
        cost_std = self.data.benchmark.cost_std
        cost_time = self.data.benchmark.time_cost
        number_of_dimension_options = cost_mean.shape[0]
        for i in range(number_of_dimension_options):
            ax.errorbar(
                x=cost_time[i],
                y=cost_mean[i],
                yerr=cost_std[i],
                color=self.colors[i],
                **kwargs,
                # fmt="o",
                # color=self.kwargs_plots["default_color"],
                # label="Dual EnKF",
            )

    def set_axes(self) -> None:
        self.set_ax(self.ax)
        # self.set_ax(self.ax_mppi)

    def set_ax(self, ax: plt.Axes) -> None:
        # ax.set_xlabel(ylabel, fontsize=self.kwargs_plots["fontsize"])
        ax.tick_params(axis="both", labelsize=self.kwargs_plots["fontsize"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="both", linestyle="--", linewidth=0.5)

    def set_legend(self) -> None:
        kwargs = {
            "fmt": "o",
            "markersize": 5,
            "capsize": 5,
            "capthick": 2,
            "elinewidth": 2,
        }
        ax = self.fig.subplots(1,1)
        number_of_dimensions = self.data.dual_enkf.cost_mean.shape[0]
        for i in range(number_of_dimensions):
            plt.errorbar(
                [100], 
                [100],
                yerr=1, 
                color=self.colors[i], 
                label=f"{self.number_of_dimensions[i]} Dim.",
                **kwargs
            )

        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(
            fontsize=self.kwargs_plots["fontsize"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
        )
        

        ax_algo = self.fig.subplots(1,1)
        ax_algo.plot([10, 10], [10, 10], "--", color="k", label="MPPI")
        ax_algo.plot([10, 10], [10, 10], "-", color="k", label="Dual EnKF")
        ax_algo.set_frame_on(False)
        ax_algo.set_xticks([])
        ax_algo.set_yticks([])
        ax_algo.patch.set_alpha(0.)
        ax_algo.set_xlim(0, 1)
        ax_algo.set_ylim(0, 1)
        ax_algo.legend(
            fontsize=self.kwargs_plots["fontsize"],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=1,
        )

        self.fig.tight_layout()


def main() -> None:    

    lqg_l4dc_data_selector = L4DCDataSelector(
        dual_enkf_folder_name="L4DC_data/Results/SMDTime", # particle_list = [100, 500, 1000, 5000]
        cost_func="lqg",
        benchmark_directory_name="MPPI",
    )
    lqg_data = load_data(
        l4dc_data_selector=lqg_l4dc_data_selector,
    )
    lqg_data_plot = L4DCDataPlot(
        data=lqg_data,
    )

    lqg_data_plot.fig.savefig("figs/mppi_comparison.pdf", dpi=300)
    
    # system_callback = h5py.File("MPPI/mppi/system.hdf5", "r")
    # cost_callback = h5py.File("MPPI/mppi/cost.hdf5", "r")

    plt.show()

if __name__ == "__main__":
    main()