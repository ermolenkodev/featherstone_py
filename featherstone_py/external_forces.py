from typing import Dict, Union, Optional

import numpy as np

from model import MultibodyModel

from functools import singledispatch

from spatial import Xinv, colvec, Tinv, Ad, T
from utils import take_last


def apply_external_forces(f_ext: Optional[Dict[int, np.ndarray]],
                          model: MultibodyModel, F: Dict[int, np.ndarray],
                          Xup: np.ndarray) -> Dict[int, np.ndarray]:
    if f_ext is None or not f_ext:
        return F

    Xi_0 = {-1: np.eye(6)}
    F_out = F.copy()

    for i in range(model.n_bodies):
        Xi_0[i] = Xup[i] @ Xi_0[model.parent[i]]
        if i in f_ext:
            # F_out[i] -= np.linalg.inv(Xi_0[i].T) @ colvec(f_ext[i])
            F_out[i] -= np.linalg.solve(Xi_0[i].T,  colvec(f_ext[i]))

    return F_out


def apply_end_effector_exerted_force(f_tip: Optional[np.ndarray],
                                     model: MultibodyModel,
                                     F: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    if f_tip is None:
        return F

    if model.T_n_ee is None:
        raise ValueError("End-effector transform is not set can't apply end-effector exerted force")

    F_out = F.copy()
    end_effector_force = take_last(F_out)
    end_effector_force += Ad(Tinv(model.T_n_ee)).T @ colvec(f_tip)

    return F_out
