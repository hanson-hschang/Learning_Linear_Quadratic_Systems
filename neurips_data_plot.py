
from typing import NamedTuple
from cvxpy import pos
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams.update({'font.family': 'serif'})
import matplotlib.patches as mpatches

from neurips_data import load_data, NeurIPSData, NeurIPSDataSelector, find_indices

class PlanarPoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.theta = np.arctan2(self.y, self.x)
    
    def get_cartisian(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def get_length(self) -> float:
        return self.r
    
    def unit_vector(self) -> np.ndarray:
        return np.array([np.cos(self.theta), np.sin(self.theta)])
    
    def rotate_angle(self, angle: float) -> np.ndarray:
        angle = angle / 180 * np.pi
        return self.r * np.array([np.cos(self.theta+angle), np.sin(self.theta+angle)])

class PolarPoint(PlanarPoint):
    def __init__(self, r: float, theta: float):
        theta = theta / 180 * np.pi
        super().__init__(r*np.cos(theta), r*np.sin(theta))

class NeurIPSDataPlot:
    def __init__(
        self, 
        data: NeurIPSData, 
        init_plot: bool = True,
        broken_axis: bool = False,
    ):
        self.ratio_to_percentage = 100
        self.kwargs_figure = {"figsize": (8, 6)}
        self.broken_axis = broken_axis
        if self.broken_axis:
            self.kwargs_axes = {
                "nrows": 2, 
                "ncols": 2,
                "sharey": "row",
                "sharex": "col",
            }
        else:
            self.kwargs_axes = {
                "nrows": 1, 
                "ncols": 2,
                "sharey": True,
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
        if self.broken_axis:
            self.ax_cost: plt.Axes = self.axes[1, 0]
            self.ax_gain: plt.Axes = self.axes[1, 1]
            self.ax_cost_upper: plt.Axes = self.axes[0, 0]
            self.ax_gain_upper: plt.Axes = self.axes[0, 1]
        else:
            self.ax_cost: plt.Axes = self.axes[0]
            self.ax_gain: plt.Axes = self.axes[1]

        self.scatter_time_errorbar(
            self.ax_gain,
            data.dual_enkf.K_error_mean,
            data.dual_enkf.time_mean,
            data.dual_enkf.time_std
        )
        self.scatter_time_errorbar(
            self.ax_cost,
            data.dual_enkf.cost_mean,
            data.dual_enkf.time_mean,
            data.dual_enkf.time_std
        )
        # for i, number_of_particles in enumerate([100, 400, 600, 700, 800, 900]):
        # for i, number_of_particles in enumerate([100, 200, 600, 700, 800, 900]):
        #     print(
        #         r"& %d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\ \cline{2-8}" 
        #         % (number_of_particles, data.dual_enkf.cost_mean[i]*100, data.dual_enkf.time_mean[i], data.dual_enkf.time_std[i], 
        #         data.dual_enkf.K_error_mean[i]*100, data.dual_enkf.time_mean[i], data.dual_enkf.time_std[i])
        #     )
        # quit()

        if init_plot:
            self.plot_time_curve(
                self.ax_gain,
                data.benchmark.K_error,
                data.benchmark.time_K_error,
                data.benchmark.time_std_K_error,
            )
            
            self.plot_time_curve(
                self.ax_cost,
                data.benchmark.cost,
                data.benchmark.time_cost,
                data.benchmark.time_std_cost,
            )

    def scatter_time_errorbar(
        self,
        ax: plt.Axes,
        value: np.ndarray,
        time_mean: np.ndarray,
        time_std: np.ndarray,
        **kwargs,
    ) -> None:
        kwargs.setdefault("color", self.kwargs_plots["default_color"])
        kwargs.setdefault("fmt", "o")
        kwargs.setdefault("markersize", 3)
        kwargs.setdefault("capsize", 3)
        kwargs.setdefault("capthick", 1)
        kwargs.setdefault("elinewidth", 1)

        ax.errorbar(
            value*self.ratio_to_percentage, time_mean, yerr=time_std, 
            **kwargs,
        )

    def plot_time_curve(
        self,
        ax: plt.Axes,
        value: np.ndarray,
        time_mean: np.ndarray,
        time_std: np.ndarray,
        **kwargs,
    ) -> None:
        mean_kwargs = {**kwargs}
        mean_kwargs.setdefault("color", self.kwargs_plots["mean_color"])
        ax.plot(value*self.ratio_to_percentage, time_mean, **mean_kwargs)
        std_kwargs = {**kwargs}
        std_kwargs.setdefault("color", self.kwargs_plots["std_color"])
        ax.fill_between(value*self.ratio_to_percentage, time_mean+time_std, time_mean-time_std, **std_kwargs)
    
    def set_ax(self, ax: plt.Axes, ylabel: str) -> None:
        ax.set_xlabel(ylabel, fontsize=self.kwargs_plots["fontsize"])
        ax.tick_params(axis="both", labelsize=self.kwargs_plots["fontsize"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="both", linestyle="--", linewidth=0.5)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    
    def set_axes(self,) -> None:
        self.set_ax(self.ax_cost, "error in cost (\%)")
        self.set_ax(self.ax_gain, "error in gain (\%)")
        if self.broken_axis:
            self.set_ax(self.ax_cost_upper, "")
            self.set_ax(self.ax_gain_upper, "")
            self.ax_cost.spines['top'].set_visible(False)
            self.ax_gain.spines['top'].set_visible(False)
            self.ax_cost_upper.spines['bottom'].set_visible(False)
            self.ax_gain_upper.spines['bottom'].set_visible(False)
            self.ax_cost_upper.tick_params(
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
            )
            self.ax_gain_upper.tick_params(
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
            )
            d = .5  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                        linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            self.ax_cost_upper.plot([0, 1], [0, 0], transform=self.ax_cost_upper.transAxes, **kwargs)
            self.ax_gain_upper.plot([0, 1], [0, 0], transform=self.ax_gain_upper.transAxes, **kwargs)
            self.ax_cost.plot([0, 1], [1, 1], transform=self.ax_cost.transAxes, **kwargs)
            self.ax_gain.plot([0, 1], [1, 1], transform=self.ax_gain.transAxes, **kwargs)

    def set_legend(self, benchmark_label: str = "benchmark") -> plt.Axes:
        ax = self.fig.subplots(1, 1)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.)
        # benchmark_time_curve = plt.plot([], [], label=benchmark_label, color=self.kwargs_plots['mean_color'])[0]
        benchmark_scatter_plot = plt.errorbar(
            [100], [100], yerr=1, label=benchmark_label, color=self.kwargs_plots['std_color'],
            fmt="o",
            markersize=5,
            capsize=5,
            capthick=2,
            elinewidth=2,
        )
        dual_enkf_scatter_plot = plt.errorbar(
            [100], [100], yerr=1, label='dual EnKF', color=self.kwargs_plots['default_color'],
            fmt="o",
            markersize=5,
            capsize=5,
            capthick=2,
            elinewidth=2,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(
            handles=[dual_enkf_scatter_plot, benchmark_scatter_plot],
            fontsize=self.kwargs_plots['fontsize'],
            loc='upper right', 
            bbox_to_anchor=(1, 1.15),
            ncol=2, 
            # fancybox=True, 
            # shadow=True
        )
        plt.tight_layout()

        ax.set_ylabel("Time (s)", fontsize=self.kwargs_plots["fontsize"], rotation='horizontal')
        ax.yaxis.set_label_coords(0., 1.05)
        self.fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
        return ax
    
    def add_arrow_with_text(
        self,
        ax: plt.Axes,
        text: str,
        position: PlanarPoint | PolarPoint,
        direction: PlanarPoint | PolarPoint,
        color: str,
        text_offset_direction: float = 0.,
        text_offset_normal: float = 0.,
    ) -> None:
        ax.add_patch(
            mpatches.FancyArrowPatch(
                position.get_cartisian(), 
                position.get_cartisian()+direction.get_cartisian(),
                mutation_scale=10,
                arrowstyle='-|>',
                edgecolor=color,
                facecolor=color,
            )
        )
        
        normal = direction.rotate_angle(90) / direction.get_length()
        text_position = position.get_cartisian() + direction.get_cartisian()/2 + text_offset_direction*direction.unit_vector() + text_offset_normal*normal
        text_rotation_angle = np.arctan2(direction.get_cartisian()[1]*get_aspect(ax), direction.get_cartisian()[0])
        text_rotation_angle = text_rotation_angle if text_rotation_angle <= np.pi/2 and text_rotation_angle >= -np.pi/2 else text_rotation_angle - np.pi
        text_rotation_angle = text_rotation_angle*180/np.pi
        ax.text(
            text_position[0], 
            text_position[1],
            text,
            verticalalignment='center', 
            horizontalalignment='center',
            fontsize=self.kwargs_plots["fontsize"],
            rotation=text_rotation_angle,
            rotation_mode='anchor',
            color=color,
        )

def get_aspect(ax: plt.Axes = None) -> float:
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    axes_ratio = height / width
    aspect = axes_ratio / ax.get_data_ratio()

    return aspect

def create_neurips_lqg_plot(
    prime_horizon_list: list = [3000, 3500, 4000, 5000, 6000, 6500]
) -> None:
    
    benchmark_directory_name = "linear_quadratic_guassian/adaptive/neurips_data/"
    neurips_data_selector = NeurIPSDataSelector(
        dual_enkf_file_name="neurips_data/3dstats10lyap.npy",
        # dual_enkf_file_name="neurips_data/3dstats100_linear_500_new_new.npy",
        # dual_enkf_file_name="neurips_data/3dstats100_linear_100.npy",
        cost_func="lqg",
        benchmark_directory_name=benchmark_directory_name+"prime_horizon_{}/".format(prime_horizon_list[0]),
        # selected_particles=None,
        selected_particles=np.ix_([0, 3, 5, 6, 7, 8]), 
        start_index=50,
    )
    data=load_data(
        neurips_data_selector=neurips_data_selector,
    )

    neurips_data_plot = NeurIPSDataPlot(data=data, init_plot=False, broken_axis=True)

    for prime_horizon in prime_horizon_list:
        neurips_data_selector = neurips_data_selector._replace(
            benchmark_directory_name=benchmark_directory_name+"prime_horizon_{}/".format(prime_horizon)
        )
        data=load_data(
            neurips_data_selector=neurips_data_selector,
        )
        neurips_data_plot.scatter_time_errorbar(
            neurips_data_plot.ax_cost_upper,
            data.benchmark.cost[0],
            data.benchmark.time_cost[0],
            data.benchmark.time_std_cost[0],
            color=neurips_data_plot.kwargs_plots["std_color"],
        )
        neurips_data_plot.scatter_time_errorbar(
            neurips_data_plot.ax_gain_upper,
            data.benchmark.K_error[0],
            data.benchmark.time_K_error[0],
            data.benchmark.time_std_K_error[0],
            color=neurips_data_plot.kwargs_plots["std_color"],
        )
        # print(
        #     r"& %d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\ \cline{2-8}" 
        #     % (prime_horizon, data.benchmark.cost[0]*100, data.benchmark.time_cost[0], data.benchmark.time_std_cost[0], 
        #        data.benchmark.K_error[0]*100, data.benchmark.time_K_error[0], data.benchmark.time_std_K_error[0])
        # )
        # quit()
    
    neurips_data_plot.set_axes()
    virtual_ax = neurips_data_plot.set_legend(
        benchmark_label="[K19]",
    )

    neurips_data_plot.ax_cost.set_xticks([0.6, 1, 3])
    neurips_data_plot.ax_gain.set_xticks([7, 10, 20])
    neurips_data_plot.ax_cost.set_yticks([0.08, 0.1])
    neurips_data_plot.ax_cost_upper.set_yticks([0.7, 1., 3.])

    arrow_direction = PlanarPoint(-0.2, 0.3)
    neurips_data_plot.add_arrow_with_text(
        virtual_ax,
        "increase particles",
        PlanarPoint(0.3, 0.15),
        arrow_direction,
        text_offset_normal=-0.03,
        color=neurips_data_plot.kwargs_plots["default_color"],
    )
    neurips_data_plot.add_arrow_with_text(
        virtual_ax,
        "increase horizon",
        PlanarPoint(0.35, 0.5),
        arrow_direction,
        text_offset_normal=0.03,
        text_offset_direction=-0.02,
        color=neurips_data_plot.kwargs_plots["std_color"],
    )
    neurips_data_plot.add_arrow_with_text(
        virtual_ax,
        "increase particles",
        PlanarPoint(0.9, 0.15),
        arrow_direction,
        text_offset_normal=-0.03,
        color=neurips_data_plot.kwargs_plots["default_color"],
    )
    neurips_data_plot.add_arrow_with_text(
        virtual_ax,
        "increase horizon",
        PlanarPoint(0.9, 0.5),
        arrow_direction,
        text_offset_normal=0.03,
        text_offset_direction=-0.02,
        color=neurips_data_plot.kwargs_plots["std_color"],
    )

    neurips_data_plot.fig.savefig("figs/lqg_comparison.pdf")

def create_neurips_leqg_plot(
    number_of_results: int = 6,
    maximum_benchmark_cost: float = 0.1,
    maximum_benchmark_K_error: float = 0.6,
) -> None:
    
    neurips_data_selector = NeurIPSDataSelector(
        # dual_enkf_file_name="neurips_data/3dstats10.npy",
        dual_enkf_file_name="neurips_data/3dstats10lyap.npy",
        cost_func="leqg",
        benchmark_directory_name="linear_quadratic_games/neurips_data/",
        # selected_particles=None,
        # selected_particles=np.ix_([0, 3, 5, 6, 7, 8]),
        selected_particles=np.ix_([0, 1, 5, 6, 7, 8]),
        start_index=10,
    )
    data=load_data(
        neurips_data_selector=neurips_data_selector,
    )

    neurips_data_plot = NeurIPSDataPlot(data=data, init_plot=False, broken_axis=True)


    cost_final_index = find_indices(data.benchmark.cost, maximum_benchmark_cost)
    cost_indices = find_indices(
        data.benchmark.cost,
        np.linspace(data.benchmark.cost[0], data.benchmark.cost[cost_final_index], number_of_results)
    )
    neurips_data_plot.scatter_time_errorbar(
        neurips_data_plot.ax_cost_upper,
        data.benchmark.cost[cost_indices],
        data.benchmark.time_cost[cost_indices],
        data.benchmark.time_std_cost[cost_indices],
        color=neurips_data_plot.kwargs_plots["std_color"],
    )

    gain_final_index = find_indices(data.benchmark.K_error, maximum_benchmark_K_error)
    gain_indices = find_indices(
        data.benchmark.K_error,
        np.linspace(data.benchmark.K_error[0], data.benchmark.K_error[gain_final_index], number_of_results)
    )
    neurips_data_plot.scatter_time_errorbar(
        neurips_data_plot.ax_gain_upper,
        data.benchmark.K_error[gain_indices],
        data.benchmark.time_K_error[gain_indices],
        data.benchmark.time_std_K_error[gain_indices],
        color=neurips_data_plot.kwargs_plots["std_color"],
    )

    # for i in range(number_of_results):
    #     iteration_number = int(data.benchmark.time_cost[cost_indices[-1-i]]/data.benchmark.time_cost[0] * 10000)
    #     print(
    #         r"& %d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\ \cline{2-8}" 
    #         % (iteration_number, data.benchmark.cost[cost_indices[-1-i]]*100, data.benchmark.time_cost[cost_indices[-1-i]], data.benchmark.time_std_cost[cost_indices[-1-i]], 
    #            data.benchmark.K_error[gain_indices[-1-i]]*100, data.benchmark.time_K_error[gain_indices[-1-i]], data.benchmark.time_std_K_error[gain_indices[-1-i]])
    #     )
    # quit()

    
    neurips_data_plot.set_axes()
    virtual_ax = neurips_data_plot.set_legend(
        benchmark_label="[Z21]",
    )

    neurips_data_plot.ax_cost.set_yticks([0.07, 0.1])
    neurips_data_plot.ax_cost_upper.set_yticks([3600, 7200])
    # neurips_data_plot.ax_cost_upper.set_yticks([60, 1000, 3600])
    neurips_data_plot.ax_gain.set_xticks([1, 10, 100])
    xlim = neurips_data_plot.ax_gain.get_xlim()
    xlim = (0.91, xlim[1])
    neurips_data_plot.ax_gain.set_xlim(xlim)

    arrow_direction = PlanarPoint(-0.2, 0.3)
    neurips_data_plot.add_arrow_with_text(
        virtual_ax,
        "increase particles",
        PlanarPoint(0.4, 0.15),
        arrow_direction,
        text_offset_normal=-0.03,
        color=neurips_data_plot.kwargs_plots["default_color"],
    )
    neurips_data_plot.add_arrow_with_text(
        virtual_ax,
        "increase iterations",
        PlanarPoint(0.3, 0.6),
        arrow_direction,
        text_offset_normal=0.03,
        text_offset_direction=-0.02,
        color=neurips_data_plot.kwargs_plots["std_color"],
    )
    neurips_data_plot.add_arrow_with_text(
        virtual_ax,
        "increase particles",
        PlanarPoint(0.7, 0.1),
        PolarPoint(arrow_direction.get_length(), 100),
        text_offset_normal=-0.03,
        color=neurips_data_plot.kwargs_plots["default_color"],
    )
    neurips_data_plot.add_arrow_with_text(
        virtual_ax,
        "increase iterations",
        PlanarPoint(0.85, 0.55),
        PolarPoint(arrow_direction.get_length(), 100),
        text_offset_normal=0.03,
        text_offset_direction=-0.02,
        color=neurips_data_plot.kwargs_plots["std_color"],
    )

    neurips_data_plot.fig.savefig("figs/leqg_comparison.pdf")

def main():
    
    create_neurips_lqg_plot()
    create_neurips_leqg_plot()

    plt.show()    

if __name__ == "__main__":
    main()