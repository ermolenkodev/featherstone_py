from typing import Optional, Dict, List, Union, NamedTuple

import numpy as np

from featherstone_py.external_forces import (
    apply_external_forces,
    apply_end_effector_exerted_force,
)
from featherstone_py.model import MultibodyModel
from featherstone_py.spatial import Vx, Vx_star, colvec


class CalculatedDynamicsQuantities(NamedTuple):
    """
    This class is used to collect all the quantities that are calculated during the inverse dynamics computation.

    Attributes
    ----------
    V : Dict[int, np.ndarray]
        Twist of each link expressed in that link frame.
    A : Dict[int, np.ndarray]
        Spatial acceleration of each link expressed in that link frame.
    F : Dict[int, np.ndarray]
        Wrench of each link expressed in that link frame.
    S : Dict[int, np.ndarray]
        Screw axis of each joint expressed in that joint frame.
    Xup : Dict[int, np.ndarray]
        Spatial transformation from i-th body frame to i+1-th body frame.
    """  # noqa: D301, E501

    V: Dict[int, np.ndarray]
    A: Dict[int, np.ndarray]
    F: Dict[int, np.ndarray]
    Xup: List[np.ndarray]
    S: List[np.ndarray]
    tau: np.ndarray


def rnea(
    model: MultibodyModel,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
    f_tip: Optional[np.ndarray] = None,
    f_ext: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """
    Calculate the torque vector needed to achieve the desired joint accelerations using the recursive Newton-Euler.\f
    Optionally the external forces acting on the links or the end-effector exerted force can be specified.
    :param model: Multibody system model in Featherstone's notation.
    :param q: Current joint positions vector.
    :param qd: Current joint velocities vector.
    :param qdd: Desired joint accelerations vector.
    :param gravity: 3D gravity vector.
    :param f_tip: End-effector exerted force expressed in the end-effector frame.
        Note that end-effector frame may not be identical to the last link frame. The homogenous transformation matrix T_n_ee from the last link frame to the end-effector frame should be specified in the model.
    :param f_ext: External forces acting on the links expressed in the base frame. The dictionary keys are the link indices.
    :return: Torque vector.
    """  # noqa: D301, E501
    return rnea_impl(model, q, qd, qdd, gravity, f_tip, f_ext).tau


def rnea_impl(
    model: MultibodyModel,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
    f_tip: Optional[np.ndarray] = None,
    f_ext: Optional[Dict[int, np.ndarray]] = None,
) -> CalculatedDynamicsQuantities:
    """
    It is actual implementation of the rnea function. It returns not only the torque vector but also the intermediate
    quantities that can be used for other purposes. For example the links twists or wrenches can be transformed to the
    different frames.
    :param model: Multibody system model in Featherstone's notation.
    :param q: Current joint positions vector.
    :param qd: Current joint velocities vector.
    :param qdd: Desired joint accelerations vector.
    :param gravity: 3D gravity vector.
    :param f_tip: End-effector exerted force expressed in the end-effector frame.
        Note that end-effector frame may not be identical to the last link frame. The homogenous transformation matrix T_n_ee from the last link frame to the end-effector frame should be specified in the model.
    :param f_ext: External forces acting on the links expressed in the base frame. The dictionary keys are the link indices.
    :return: CalculatedDynamicsQuantities tuple containing the torque vector and intermediate quantities such as links twists, spatial accelerations, spatial wrenches, spatial transformations from i-th body frame to i+1-th body frame and screw axes of joints.
    """  # noqa: D301, E501
    n_bodies, joints, parent, X_tree, I, _ = model

    # velocity of the base is zero
    V = {-1: np.zeros((6, 1))}

    # Note: we are assign acceleration of the base to the -gravity,
    # but it is just a way to incorporate gravity term to the recursive force propagation formula
    spatial_gravity = np.block([[np.zeros((3, 1))], [colvec(gravity)]])
    A = {-1: -spatial_gravity}

    F = {-1: np.zeros((6, 1))}
    Xup = [np.empty((6, 6))] * n_bodies
    S = [np.zeros((6, 1))] * n_bodies

    # All indexed from 0
    for i in range(n_bodies):
        Xj, S[i] = joints[i].joint_transform(q[i]), joints[i].screw_axis()
        Xup[i] = Xj @ X_tree[i]
        Vj = S[i] * qd[i]

        V[i] = Xup[i] @ V[parent[i]] + Vj
        A[i] = Xup[i] @ A[parent[i]] + S[i] * qdd[i] + Vx(V[i]) @ Vj

        F[i] = I[i] @ A[i] + Vx_star(V[i]) @ I[i] @ V[i]

    F = apply_end_effector_exerted_force(f_tip, model, F)
    F = apply_external_forces(f_ext, model, F, Xup)

    tau = np.zeros([n_bodies, 1])
    for i in range(n_bodies - 1, -1, -1):
        tau[i, 0] = S[i].T @ F[i]
        F[parent[i]] += Xup[i].T @ F[i]

    return CalculatedDynamicsQuantities(V, A, F, Xup, S, tau)
