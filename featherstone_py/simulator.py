from typing import Union, List

import numpy as np

from featherstone_py.forward_dynamics import InertiaMatrixMethod, InverseDynamicsUsingRNEA
from featherstone_py.model import MultibodyModel


class Simulator:
    # TODO add validation of properties
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

    def advance_to(self, q_: np.ndarray, qd_: np.ndarray, tau_: np.ndarray, time: float):
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

        return q, qd, qdd
