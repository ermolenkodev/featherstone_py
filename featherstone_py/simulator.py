from typing import Union, List, Tuple

import numpy as np

from featherstone_py.forward_dynamics import InertiaMatrixMethod, InverseDynamicsUsingRNEA
from featherstone_py.model import MultibodyModel


class Simulator:
    """
    It is a toy forward dynamics simulator which uses the given forward dynamics algorithm to simulate the joints trajectory.\f
    It uses Euler integration scheme and it is quite inefficient

    Attributes
    ----------
    model: MultibodyModel
        Multibody system model in Featherstone's notation.
    forward_dynamics: InertiaMatrixMethod
        Forward dynamics algorithm which is used to calculate the joint accelerations.
    dt: float
        Simulation time step.
    integration_resolution: int
        Number of integration steps per simulation time step. Effectively, it makes time step dt smaller by factor of integration_resolution.
    curr_time: float
        Internal elapsed time, should be reset manually if you want to start simulation from the beginning
    gravity: np.ndarray
        3d gravity vector.
    """  # noqa: D301
    def __init__(self, model: MultibodyModel,
                 forward_dynamics: InertiaMatrixMethod = InverseDynamicsUsingRNEA(),
                 dt: float = 0.01,
                 integration_resolution: int = 1,
                 gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81])):
        self.integration_resolution = integration_resolution
        self.forward_dynamics = forward_dynamics
        self.model = model
        self.gravity = gravity
        self.curr_time = 0.
        self.dt = dt

    def advance_to(self, q_: np.ndarray, qd_: np.ndarray, tau_: np.ndarray, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance the simulation to the given time. It uses Euler integration scheme.
        
        :param q_: Current joint positions
        :param qd_: Current joint velocities
        :param tau_: Current applied joint torques
        :param time: Time to advance to. It should be greater than the internal time of the simulator.
        :return: Joint positions and velocities at the requested time
        :raises ValueError: If the requested time is less than the internal time of the simulator.
        """  # noqa: D301
        q, qd, tau = q_.copy(), qd_.copy(), tau_.copy()

        if time < self.curr_time:
            raise ValueError("Cannot go back in time")

        dt = self.dt / self.integration_resolution

        while self.curr_time < time:
            qdd = self.forward_dynamics(self.model, q, qd, tau, gravity=self.gravity).reshape(q_.shape)
            q += qd * dt
            qd += qdd * dt
            if self.curr_time + dt > time:
                dt = time - self.curr_time
            self.curr_time += dt

        return q, qd

    def reset_internal_time(self) -> None:
        """
        Reset the internal time of the simulator to zero.
        :return:
        """
        self.curr_time = 0.
