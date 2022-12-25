from typing import Optional, Dict, List

import numpy as np
from model import MultibodyModel
from spatial import Vx, Vx_star


def rnea(model: MultibodyModel, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    return RNEAImpl().run(model, q, qd, qdd)


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

    def run(self, model: MultibodyModel, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        n_bodies, parent, joints, X_tree, I, gravity = model.as_tuple()

        # velocity of the base is zero
        V = {-1: np.zeros((6, 1))}

        # Note: we are assign acceleration of the base to the -gravity,
        # but it is just a way to incorporate gravity term to the recursive force propagation formula
        spatial_gravity = np.block([
            [np.zeros((3, 1))],
            [gravity]
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

        tau = np.zeros([n_bodies, 1])
        for i in range(n_bodies-1, -1, -1):
            tau[i, 0] = S[i].T @ F[i]
            F[parent[i]] += Xup[i].T  @ F[i]

        self.V, self.A, self.F, self.Xup, self.S = V, A, F, Xup, S
        self.executed = True

        return tau
