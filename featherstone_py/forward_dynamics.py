from typing import List, Union, Optional

import numpy as np

from featherstone_py.inverse_dynamics import rnea
from featherstone_py.model import MultibodyModel
from abc import ABC, abstractmethod

from featherstone_py.spatial import colvec


class InertiaMatrixMethod(ABC):
    """
    Interface for various implementations of the forward dynamics algorithms which based on calculation of inverse of
    the mass matrix (joint space inertia matrix).

    Methods
    ----------
    __call__(model, q, qd, tau, f_ext, gravity) -> qdd
    """
    @abstractmethod
    def __call__(self, model: MultibodyModel, q: np.ndarray, qd: np.ndarray, tau: np.ndarray,
                 f_ext: Optional[np.ndarray] = None,
                 gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81])) -> np.ndarray:
        """
        Calculate the joint accelerations of the multibody system, given the systems model and the current state.
        :param model: Multibody system model in Featherstone's notation.
        :param q: Joint positions vector.
        :param qd: Joint velocities vector.
        :param tau: Joint torques vector.
        :param f_ext: External forces acting on the links.
        :param gravity: Gravity vector.
        :return: Joint accelerations vector.
        """
        pass


class InverseDynamicsUsingRNEA(InertiaMatrixMethod):
    """
    Implementation of the forward dynamics algorithm based on the recursive Newton-Euler algorithm
    for calculating the mass matrix. It is the straightforward but inefficient implementation of the forward dynamics.
    """
    def __call__(self, model: MultibodyModel, q: np.ndarray, qd: np.ndarray, tau: np.ndarray,
                 f_ext: Optional[np.ndarray] = None,
                 gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81])) -> np.ndarray:
        """
        Calculate the joint accelerations of the multibody system using the recursive Newton-Euler algorithm for
        calculating the bias forces vector and the mass matrix.
        """
        C = calculate_bias_forces(model, q, qd, gravity, f_ext)
        M = calculate_mass_matrix_using_rnea(model, q)

        return np.linalg.inv(M) @ (colvec(tau) - C)


def calculate_bias_forces(model: MultibodyModel, q: np.ndarray, qd: np.ndarray,
                          gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
                          f_ext: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the bias forces vector using the recursive Newton-Euler algorithm.\f
    Conceptually this method is equivalent to substituting of zero joint accelerations
    into the manipulator equation of motion.\f
    tau = M(q) * [0,0,...,0] + C(q, qd)

    :param model: Multibody system model in Featherstone's notation.
    :param q: Joint positions vector.
    :param qd: Joint velocities vector.
    :param tau: Joint torques vector.
    :param f_ext: External forces acting on the links.
    :param gravity: Gravity vector.
    :return: Vector of bias forces.
    """
    qdd = np.zeros_like(q)
    return rnea(model, q, qd, qdd, gravity, f_ext)


def calculate_mass_matrix_using_rnea(model: MultibodyModel, q: np.ndarray) -> np.ndarray:
    """
    Calculate the mass matrix using the recursive Newton-Euler algorithm.\f
    Zeroing joint velocities, gravity and external forces we are effectively zeroing the bias forces vector
    in the manipulator equation of motion.\f
    On each iteration we calculate specific column of the mass matrix using the synthetic
    joint accelerations vector with the only one non-zero element.\f
    tau = M(q) * [0,0,..,1,..,0] - tau is effectively the column of the mass matrix.
    :param model: Multibody system model in Featherstone's notation.
    :param q: Joint positions vector.
    :return: Mass matrix.
    """
    n_bodies = model.n_bodies
    M = np.zeros([n_bodies, n_bodies])

    qd = np.zeros_like(q)
    g = [0., 0., 0.]
    identity = np.eye(n_bodies)
    for i in range(n_bodies):
        qdd = identity[:, i]
        M[:, [i]] = rnea(model, q, qd, qdd, gravity=g)

    return M
