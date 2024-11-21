import os
from pathlib import Path

import click
import numpy as np
from scipy.linalg import solve_continuous_are
from tqdm import tqdm
from time import perf_counter

from ss.control.cost import CostCallback
from ss.control.cost.quadratic import QuadraticCost
from ss.control.mppi import ModelPredictivePathIntegralController
from ss.system.state_vector import SystemCallback
from ss.system.state_vector.examples.mass_spring_damper import (
    ControlChoice,
    MassSpringDamperSystem,
)


@click.command()
@click.option(
    "--number-of-connections",
    type=click.IntRange(min=1),
    default=2,
    help="Set the number of connections (positive integers).",
)
@click.option(
    "--simulation-time",
    type=click.FloatRange(min=0),
    default=10.0,
    help="Set the simulation time (positive value).",
)
@click.option(
    "--time-step",
    type=click.FloatRange(min=0),
    default=0.1,
    help="Set the time step (positive value).",
)
@click.option(
    "--step-skip",
    type=click.IntRange(min=1),
    default=1,
    help="Set the step skip (positive integers).",
)
@click.option(
    "--number-of-systems",
    type=click.IntRange(min=1),
    default=15,
    help="Set the number of systems (positive integers).",
)
@click.option(
    "--number-of-samples",
    type=click.IntRange(min=1),
    default=1000,
    help="Set the number of samples (positive integers).",
)
def main(
    number_of_connections: int,
    simulation_time: float,
    time_step: float,
    step_skip: int,
    number_of_systems: int,
    number_of_samples: int,
) -> None:
    simulation_time_steps = int(simulation_time / time_step)
    system = MassSpringDamperSystem(
        number_of_connections=number_of_connections,
        time_step=time_step,
        control_choice=ControlChoice.ALL_FORCES,
        process_noise_covariance=0.01 * np.identity(2 * number_of_connections),
        number_of_systems=number_of_systems,
    )
    system.state = np.random.multivariate_normal(
        np.ones(2 * number_of_connections),
        np.identity(2 * number_of_connections),
        size=(number_of_systems,),
    ).squeeze()
    system_callback = SystemCallback(
        step_skip=step_skip,
        system=system,
    )
    cost = QuadraticCost(
        running_cost_state_weight=np.identity(2 * number_of_connections),
        running_cost_control_weight=np.identity(number_of_connections),
        time_step=time_step,
        number_of_systems=number_of_systems,
    )
    cost_callback = CostCallback(
        step_skip=step_skip,
        cost=cost,
    )
    controller = ModelPredictivePathIntegralController(
        system=system,
        cost=cost,
        time_horizon=100,
        number_of_samples=number_of_samples,
        temperature=100.0,
        base_control_confidence=0.98,
    )

    are_solution_inf = solve_continuous_are(
        a=system.state_space_matrix_A,
        b=system.state_space_matrix_B,
        q=cost.running_cost_state_weight,
        r=cost.running_cost_control_weight,
    )
    optimal_cost = np.trace(
        are_solution_inf @ system.process_noise_covariance
    )

    start_time = perf_counter()
    current_time = 0.0
    for k in tqdm(range(simulation_time_steps)):

        # Get the current state
        current_state = system.state
        if len(current_state.shape) == 1:
            current_state = current_state[np.newaxis, :]

        # Initialize control placeholder
        control = np.zeros_like(system.control)
        if len(control.shape) == 1:
            control = control[np.newaxis, :]
        for i in range(number_of_systems):
            # Compute the control for each system
            controller.reset_systems(current_state[i, :])
            control_trajectory = controller.compute_control()
            control[i, :] = control_trajectory[:, 0]

        # Set the control
        system.control = control.squeeze()
        system_callback.make_callback(k, current_time)

        # Compute the cost
        cost.state = current_state.squeeze()
        cost.control = control.squeeze()
        cost_callback.make_callback(k, current_time)

        # Update the system
        current_time = system.process(current_time)

    # Compute the terminal cost
    cost.set_terminal()
    cost.state = system.state
    cost_callback.make_callback(simulation_time_steps, current_time)
    system_callback.make_callback(simulation_time_steps, current_time)

    end_time = perf_counter()
    cost_callback.add_meta_data(
        {
            "execution_time (sec)": end_time - start_time,
            "number_of_connections" : number_of_connections,
            "simulation_time (sec)" : simulation_time,
            "time_step (sec)" : time_step,
            "optimal_cost" : optimal_cost,
        }
    )

    # Save the data
    parent_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    data_folder_directory = parent_directory / Path(__file__).stem
    system_callback.save(data_folder_directory / "system.hdf5")
    cost_callback.save(data_folder_directory / "cost.hdf5")



if __name__ == "__main__":
    main()
