from typing import Optional, Dict, List, Union

import numpy as np

from external_forces import apply_external_forces, apply_end_effector_exerted_force
from featherstone_py.model import MultibodyModel
from featherstone_py.spatial import Vx, Vx_star, colvec


def rnea(model: MultibodyModel, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
         gravity: Union[np.ndarray, List[float]] = np.array([0, 0, -9.81]),
         f_tip: Optional[np.ndarray] = None,
         f_ext: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
    return RNEAImpl().run(model, q, qd, qdd, gravity, f_tip, f_ext)


# Implementation of Recursive Newton-Euler Algorithm
# TODO apply_external_forces
class RNEAImpl:
    def __init__(self):
        self.executed = False
        self.V: Optional[Dict[int, np.ndarray]]
        self.A: Optional[Dict[int, np.ndarray]]
        self.F: Optional[List[np.ndarray]]
        self.Xup: Optional[List[np.ndarray]]
        self.S: Optional[List[np.ndarray]]

    def run(self, model: MultibodyModel, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, gravity: np.ndarray,
            f_tip: Optional[np.ndarray] = None,
            f_ext: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
        n_bodies, joints, parent, X_tree, I, _ = model

        # velocity of the base is zero
        V = {-1: np.zeros((6, 1))}

        # Note: we are assign acceleration of the base to the -gravity,
        # but it is just a way to incorporate gravity term to the recursive force propagation formula
        spatial_gravity = np.block([
            [np.zeros((3, 1))],
            [colvec(gravity)]
        ])
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
        for i in range(n_bodies-1, -1, -1):
            tau[i, 0] = S[i].T @ F[i]
            F[parent[i]] += Xup[i].T  @ F[i]

        self.V, self.A, self.F, self.Xup, self.S = V, A, F, Xup, S
        self.executed = True

        return tau
