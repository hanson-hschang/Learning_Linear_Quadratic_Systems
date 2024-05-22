
from typing import NamedTuple
from cvxpy import pos
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch import ne
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams.update({'font.family': 'serif'})
import matplotlib.patches as mpatches

from neurips_data import load_smd_data, DualEnKFData
from neurips_data_plot import PlanarPoint, PolarPoint, get_aspect

class NeurIPS_SMD_DataPlot:
    def __init__(
        self, 
        # data: DualEnKFData, 
    ):
        self.ratio_to_percentage = 100
        self.kwargs_figure = {"figsize": (8, 6)}
        self.kwargs_axes = {
            "nrows": 1, 
            "ncols": 2,
            "sharey": True,
        }
        self.kwargs_plots =  {
            "fontsize": 18, 
            "default_color": "#1C6AB1",
        }
        self.fig = plt.figure(**self.kwargs_figure)
        self.axes: np.ndarray = self.fig.subplots(**self.kwargs_axes)
        self.ax_cost: plt.Axes = self.axes[0]
        self.ax_gain: plt.Axes = self.axes[1]

    def plot_data(
        self, 
        data: DualEnKFData,
        color: str,
    ) -> None:
        self.scatter_time_errorbar(
            self.ax_gain,
            data.K_error_mean,
            data.time_mean,
            data.time_std,
            color=color,
        )
        self.scatter_time_errorbar(
            self.ax_cost,
            data.cost_mean,
            data.time_mean,
            data.time_std,
            color=color,
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

    def set_legend(self, handles) -> plt.Axes:
        ax = self.fig.subplots(1, 1)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.)


        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(
            handles=handles,
            fontsize=self.kwargs_plots['fontsize'],
            loc='upper right', 
            bbox_to_anchor=(1, 1.15),
            ncol=2,
        )
        plt.tight_layout()

        ax.set_ylabel("Time (s)", fontsize=self.kwargs_plots["fontsize"], rotation='horizontal')
        ax.yaxis.set_label_coords(0., 1.05)
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

def create_neurips_smd_plot() -> None:

    file_name = "neurips_data/smdstats.npy"
    for cost_func in ["lqg", "leqg+", "leqg-"]:
        neurips_smd_data_plot = NeurIPS_SMD_DataPlot()
        handles = []
        for number_of_dimension, color in zip(
            [4, 10, 20], ["#1C6AB1", "#FFA500", "#FF0000"]
        ):
            data = load_smd_data(
                file_name=file_name,
                number_of_dimension=number_of_dimension,
                cost_func=cost_func,
            )
            neurips_smd_data_plot.plot_data(
                data=data,
                color=color
            )
            # handles.append(
            #     neurips_smd_data_plot.plot_data(
            #         data=data,
            #         color=color
            #     )
            # )
        neurips_smd_data_plot.fig.suptitle(f"SMD {cost_func.upper()}")
        # ax = neurips_smd_data_plot.set_legend(handles)

def main():
    
    create_neurips_smd_plot()

    plt.show()    

if __name__ == "__main__":
    main()