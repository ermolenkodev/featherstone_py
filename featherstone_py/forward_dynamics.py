from typing import List, Union, Optional, Tuple, Dict

import numpy as np

from featherstone_py.inverse_dynamics import rnea, rnea_impl
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
    """  # noqa: D301, E501

    @abstractmethod
    def __call__(
        self,
        model: MultibodyModel,
        q: np.ndarray,
        qd: np.ndarray,
        tau: np.ndarray,
        f_tip: Optional[np.ndarray] = None,
        f_ext: Optional[Dict[int, np.ndarray]] = None,
        gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
    ) -> np.ndarray:
        """
        Calculate the joint accelerations of the multibody system, given the systems model and the current state.

        :param model: Multibody system model in Featherstone's notation.
        :param q: Joint positions vector.
        :param qd: Joint velocities vector.
        :param tau: Joint torques vector.
        :param f_tip: End-effector exerted force expressed in the end-effector frame.
            Note that end-effector frame may not be identical to the last link frame. The homogenous transformation matrix T_n_ee from the last link frame to the end-effector frame should be specified in the model.
        :param f_ext: External forces acting on the links.
        :param gravity: Gravity vector.
        :return: Joint accelerations vector.
        """  # noqa: D301, E501
        pass


class InverseDynamicsUsingRNEA(InertiaMatrixMethod):
    """
    Implementation of the forward dynamics algorithm based on the recursive Newton-Euler algorithm
    for calculating the mass matrix. It is the straightforward but inefficient implementation of the forward dynamics.
    """  # noqa: D301, E501

    def __call__(
        self,
        model: MultibodyModel,
        q: np.ndarray,
        qd: np.ndarray,
        tau: np.ndarray,
        f_tip: Optional[np.ndarray] = None,
        f_ext: Optional[Dict[int, np.ndarray]] = None,
        gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
    ) -> np.ndarray:
        """
        Calculate the joint accelerations of the multibody system using the recursive Newton-Euler algorithm for
        calculating the bias forces vector and the mass matrix.
        """  # noqa: D301, E501
        C = calculate_bias_forces(model, q, qd, gravity, f_tip, f_ext)
        M = calculate_mass_matrix_using_rnea(model, q)

        return np.linalg.inv(M) @ (colvec(tau) - C)


def calculate_bias_forces(
    model: MultibodyModel,
    q: np.ndarray,
    qd: np.ndarray,
    gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
    f_tip: Optional[np.ndarray] = None,
    f_ext: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
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
    :param f_tip: f_tip: End-effector exerted force expressed in the end-effector frame.
        Note that end-effector frame may not be identical to the last link frame. The homogenous transformation matrix T_n_ee from the last link frame to the end-effector frame should be specified in the model.
    :param gravity: Gravity vector.
    :return: Vector of bias forces.
    """  # noqa: D301, E501
    qdd = np.zeros_like(q)
    return rnea(model, q, qd, qdd, gravity, f_tip, f_ext)


def calculate_mass_matrix_using_rnea(
    model: MultibodyModel, q: np.ndarray
) -> np.ndarray:
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
    """  # noqa: D301, E501
    n_bodies = model.n_bodies
    M = np.zeros([n_bodies, n_bodies])

    qd = np.zeros_like(q)
    g = [0.0, 0.0, 0.0]
    identity = np.eye(n_bodies)
    for i in range(n_bodies):
        qdd = identity[:, i]
        M[:, [i]] = rnea(model, q, qd, qdd, gravity=g)

    return M


class InverseDynamicsUsingCRBA(InertiaMatrixMethod):
    """
    Implementation of the forward dynamics algorithm based on the composite rigid body algorithm.
    """  # noqa: D301, E501

    def __call__(
        self,
        model: MultibodyModel,
        q: np.ndarray,
        qd: np.ndarray,
        tau: np.ndarray,
        f_tip: Optional[np.ndarray] = None,
        f_ext: Optional[Dict[int, np.ndarray]] = None,
        gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
    ) -> np.ndarray:
        M, C = crba(model, q, qd, gravity, f_tip, f_ext)
        return np.linalg.inv(M) @ (colvec(tau) - C)


def crba(
    model: MultibodyModel,
    q: np.ndarray,
    qd: np.ndarray,
    gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
    f_tip: Optional[np.ndarray] = None,
    f_ext: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mass matrix and the bias forces vector using the composite rigid body algorithm.

    :param model: Multibody system model in Featherstone's notation.
    :param q: Joint positions vector.
    :param qd: Joint velocities vector.
    :param f_ext: External forces acting on the links.
    :param f_tip: f_tip: End-effector exerted force expressed in the end-effector frame.
        Note that end-effector frame may not be identical to the last link frame. The homogenous transformation matrix T_n_ee from the last link frame to the end-effector frame should be specified in the model.
    :param gravity: Gravity vector.
    :return: Mass matrix and vector of bias forces.
    """  # noqa: D301, E501
    # Here we calculate the bias forces vector using the recursive Newton-Euler algorithm.
    qdd = np.zeros_like(q)
    _, _, _, Xup, S, C = rnea_impl(model, q, qd, qdd, gravity, f_tip, f_ext)

    Ic = calculate_composite_inertia(model, Xup)
    M = calculate_mass_matrix_crba(model, S, Xup, Ic)

    return M, C


def calculate_composite_inertia(
    model: MultibodyModel, Xup: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Calculate the composite inertia for each composite body supported by the corresponding link.

    This function computes the composite inertia for each composite body in a multibody system. The composite inertia is the sum of the inertia of the link and the inertia of all the links that are supported by it. This artificially constructed quantity enables efficient calculation of the mass matrix using the Composite Rigid Body Algorithm (CRBA).

    :param model: A multibody system model in Featherstone's notation.
    :param Xup: A list of spatial transforms of frame parent[i] to frame i.
    :return: A list of composite inertia matrices.
    """  # noqa: D301, E501
    n_bodies, parent, I = model.n_bodies, model.parent, model.I

    Ic = [Ii.copy() for Ii in I]
    for i in range(n_bodies - 1, -1, -1):
        if parent[i] != -1:
            Ic[parent[i]] += Xup[i].T @ Ic[i] @ Xup[i]

    return Ic


def calculate_mass_matrix_crba(
    model: MultibodyModel,
    S: List[np.ndarray],
    Xup: List[np.ndarray],
    Ic: List[np.ndarray],
) -> np.ndarray:
    """
    Calculate the mass matrix of a multibody system using the Composite Rigid Body Algorithm (CRBA).

    This function computes the mass matrix for a given multibody system using the CRBA. The algorithm requires the system model, a list of joint motion vectors (S), a list of spatial transforms (Xup), and a list of composite inertia matrices (Ic).

    :param model: A MultibodyModel object representing the multibody system in Featherstone's notation.
    :param S: A list of np.ndarray objects representing the joint motion vectors for each body in the multibody system.
    :param Xup: A list of np.ndarray objects representing the spatial transforms from frame parent[i] to frame i for each body in the multibody system.
    :param Ic: A list of np.ndarray objects representing the composite inertia matrices for each body in the multibody system.
    :return: A np.ndarray object representing the mass matrix of the multibody system.
    """  # noqa: D301, E501
    n_bodies, parent = model.n_bodies, model.parent

    M = np.zeros([n_bodies, n_bodies])
    for i in range(n_bodies):
        F = Ic[i] @ S[i]
        M[i, i] = S[i].T @ F
        j = i
        while parent[j] != -1:
            F = Xup[j].T @ F
            j = parent[j]
            # the M[i, j] is scalar, so the next line is equivalent to M[i, j] = F.T @ S[j]
            M[i, j] = S[j].T @ F
            # note that if the joint have multiple degrees of freedom
            # M[i, j] is not scalar and the transpose is required (e.t. M[j, i] = M[i, j].T)
            M[j, i] = M[i, j]

    return M
