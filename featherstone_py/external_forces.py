from typing import Dict, Optional

import numpy as np

from featherstone_py.model import MultibodyModel

from featherstone_py.spatial import colvec, Tinv, Ad, T
from featherstone_py.utils import take_last


def apply_external_forces(f_ext: Optional[Dict[int, np.ndarray]],
                          model: MultibodyModel, F: Dict[int, np.ndarray],
                          Xup: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Incorporates the external forces specified in f_ext into the calculations of a dynamics algorithm.\f
    f_ext is a dictionary of spatial forces, indexed by body id.\f
    If specified, f_ext[i] is the spatial force exerted on body i by the environment expressed in the base frame.\f
    if f_ext[i] is not specified, it is not included in the calculations.\f
    For each body algorithm is basically computes F_out[i] = F[i] - transformed f_ext[i], where F_out and F are dicts of forces expressed in link coordinates; so f_ext has to be transformed to link coordinates before use.
    :param f_ext: dictionary of spatial forces exerted on bodies by the environment
    :param model: Multibody system model in Featherstone's notation.
    :param F: dictionary of spatial forces to be updated with the external forces
    :param Xup: contain the spatial transformations from i-th body frame to i+1-th body frame. Should be precomputed. See inverse_dynamics.py for example.
    :return: dictionary of spatial forces updated with the external forces
    """  # noqa: D301
    if f_ext is None or not f_ext:
        return F

    Xi_0 = {-1: np.eye(6)}
    F_out = F.copy()

    for i in range(model.n_bodies):
        Xi_0[i] = Xup[i] @ Xi_0[model.parent[i]]
        if i in f_ext:
            # This part is quite tricky
            # Xi_0[i] is the spatial transform from the base frame to the i-th body frame
            # Xi_0[i].T is the wrench transform from the i-th body frame to the base frame
            # f_ext[i] is expressed in the base frame, and we want to transform it to the i-th body frame,
            # so we need to apply the inverse of the Xi_0[i].T to the f_ext[i]
            # it can be done as np.linalg.inv(Xi_0[i].T) @ colvec(f_ext[i])
            # but it's more efficient to use np.linalg.solve cause we don't need the actual inverse matrix here
            # this trick is used in the original code of Featherstone's spatialv2 library
            F_out[i] -= np.linalg.solve(Xi_0[i].T,  colvec(f_ext[i]))

    return F_out


def apply_end_effector_exerted_force(f_tip: Optional[np.ndarray],
                                     model: MultibodyModel,
                                     F: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    This API is not used in the original Featherstone's code, but it is used in Modern Robotics library and seems to be useful.\f
    Incorporates the end-effector exerted force specified in f_tip into the calculations of a dynamics.\f
    :param f_tip: spatial force exerted by the end-effector on the environment expressed in the end-effector frame.
    :param model: Multibody system model in Featherstone's notation. Note that you should specify the T_n_ee transform in the model which is the homogenous transformation matrix from the last body frame to the end-effector frame.
    :param F: dictionary of spatial forces to be updated
    :return: dictionary of spatial forces updated with the end-effector exerted force
    """  # noqa: D301
    if f_tip is None:
        return F

    if model.T_n_ee is None:
        raise ValueError("End-effector transform is not set can't apply end-effector exerted force")

    F_out = F.copy()
    end_effector_force = take_last(F_out)
    X_ee_n = Ad(Tinv(model.T_n_ee))
    end_effector_force += X_ee_n.T @ colvec(f_tip)

    return F_out
