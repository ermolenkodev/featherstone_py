from typing import List, Union, Optional

import numpy as np
from numpy import ndarray

from featherstone_py.inverse_dynamics import rnea
from featherstone_py.model import MultibodyModel
from abc import ABC, abstractmethod

from featherstone_py.spatial import colvec


def calculate_bias_forces(model: MultibodyModel, q: np.ndarray, qd: np.ndarray,
                          gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
                          f_ext: Optional[np.ndarray] = None) -> np.ndarray:
    qdd = np.zeros_like(q)
    return rnea(model, q, qd, qdd, gravity, f_ext)


def calculate_mass_matrix_using_rnea(model: MultibodyModel, q: np.ndarray) -> np.ndarray:
    n_bodies = model.n_bodies
    M = np.zeros([n_bodies, n_bodies])

    qd = np.zeros_like(q)
    g = [0., 0., 0.]
    identity = np.eye(n_bodies)
    for i in range(n_bodies):
        qdd = identity[:, i]
        M[:, [i]] = rnea(model, q, qd, qdd, gravity=g)

    return M


class InertiaMatrixMethod(ABC):
    @abstractmethod
    def __call__(self, model: MultibodyModel, q: np.ndarray, qd: np.ndarray, tau: np.ndarray,
                 f_ext: Optional[np.ndarray] = None,
                 gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81])) -> np.ndarray:
        pass


class InverseDynamicsUsingRNEA(InertiaMatrixMethod):
    def __call__(self, model: MultibodyModel, q: np.ndarray, qd: np.ndarray, tau: np.ndarray,
                 f_ext: Optional[np.ndarray] = None,
                 gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81])) -> np.ndarray:
        C = calculate_bias_forces(model, q, qd, gravity, f_ext)
        M = calculate_mass_matrix_using_rnea(model, q)

        return np.linalg.inv(M) @ (colvec(tau) - C)
