
import time
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
            "nrows": 2, 
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
        self.axes: np.ndarray = self.fig.subplots(**self.kwargs_axes)
        self.ax_mppi = self.axes[0]
        self.ax_dual_enkf = self.axes[1]

        self.set_axes()
        self.plot_dual_enkf_data()

    def plot_dual_enkf_data(self,) -> None:
        ax: plt.Axes = self.ax_dual_enkf
        cost_mean = self.data.dual_enkf.cost_mean
        cost_std = self.data.dual_enkf.cost_std
        cost_time = self.data.dual_enkf.time_mean
        number_of_dimension_options = cost_mean.shape[0]
        for i in range(number_of_dimension_options):
            print(cost_std[i])
            ax.errorbar(
                x=cost_time[i],
                y=cost_mean[i],
                yerr=cost_std[i],
                # fmt="o",
                # color=self.kwargs_plots["default_color"],
                # label="Dual EnKF",
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cost")
        # ax.legend()

    def set_axes(self) -> None:
        self.set_dual_enkf_ax()

    def set_dual_enkf_ax(self) -> None:
        ax: plt.Axes = self.ax_dual_enkf
        # ax.set_xlabel(ylabel, fontsize=self.kwargs_plots["fontsize"])
        ax.tick_params(axis="both", labelsize=self.kwargs_plots["fontsize"])
        ax.set_xscale("log")
        ax.set_yscale("log")

    def set_legend(self) -> None:
        ax = self.fig.subplots(1,1)
        number_of_dimensions = self.data.dual_enkf.cost_mean.shape[0]

        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

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
    
    # system_callback = h5py.File("MPPI/mppi/system.hdf5", "r")
    # cost_callback = h5py.File("MPPI/mppi/cost.hdf5", "r")

    plt.show()

if __name__ == "__main__":
    main()