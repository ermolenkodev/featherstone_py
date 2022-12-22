from typing import Optional, List

import numpy as np
from model import MultibodyModel
from rnea import Vx, Vx_star


# TODO apply_external_forces
def rnea(model: MultibodyModel, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    n_bodies, parent, joints, X_tree, I, gravity = model.as_tuple()

    V = np.zeros((6, n_bodies+1))
    spatial_gravity = np.block([
        [np.zeros((3, 1))],
        [gravity]
    ])
    A = np.block([[-spatial_gravity, np.zeros((6, n_bodies))]])
    F = np.zeros((6, n_bodies+1))
    Xup: List[Optional[np.ndarray]] = [None] * (n_bodies+1)
    S = np.zeros((6, n_bodies+1))

    # joints, q, qd, qdd, I, X_tree are numbered from 0 to n_bodies-1
    # while Xup, S, V, A, F are numbered from 0 to n_bodies
    # TODO this is a bit confusing, maybe it's better to have consistent numbering
    for i in range(1, n_bodies+1):
        print("=======================================")
        # X_ipi = Ad(MatrixExp6(se3(model.S_ii[i] * -theta[i])) @ Tinv(model.poses[i]))
        Xj, S[:, [i]] = joints[i-1].joint_transform(q[i-1]), joints[i-1].screw_axis()
        Xup[i] = Xj @ X_tree[i-1]

        # V_ii.append(
        #     X_ipi @ V_ii[i-1] + model.S_ii[i] * theta_dot[i]
        # )
        Vj = S[:, [i]] * qd[i-1]
        V[:, [i]] = Xup[i] @ V[:, [parent[i-1]]] + Vj

        # A_ii.append(
        #     X_ipi @ A_ii[i-1] + model.S_ii[i] * theta_dot_dot[i] + Vx(V_ii[i]) @ model.S_ii[i] * theta_dot[i]
        # )
        A[:, [i]] = Xup[i] @ A[:, [parent[i-1]]] + S[:, [i]] * qdd[i-1] + Vx(V[:, [i]]) @ Vj

        # print(Xup[i])
        # print(A[:, [parent[i-1]]])
        # print(S[:, [i]])
        # print(qdd[i-1])
        # print(Vx(V[:, [i]]))
        # print(Vj)
        # F_ii[i] = model.I_ii[i] @ A_ii[i] + Vx_star(V_ii[i]) @ (model.I_ii[i] @  V_ii[i])
        F[:, [i]] = I[i-1] @ A[:, [i]] + Vx_star(V[:, [i]]) @ I[i-1] @ V[:, [i]]
        # print(F[:, [i]])

    tau = np.zeros([n_bodies, 1])
    for i in range(n_bodies, 0, -1):
        tau[i-1, 0] = S[:, [i]].T @ F[:, [i]]
        #     X_pii_star = Ad(MatrixExp6(se3(model.S_ii[i] * -theta[i])) @ Tinv(model.poses[i])).T
        #     F_ii[i-1] += X_pii_star @ F_ii[i]
        F[:, [parent[i-1]]] += Xup[i].T  @ F[:, [i]]

    return tau
