from math import cos, sin
from typing import Optional, Callable, List

import numpy as np
from enum import Enum

from mr import Adjoint, TransInv, MatrixExp6, VecTose3, ad


def T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    assert R.shape == (3, 3)
    assert p.shape == (3, 1)

    return np.block([
        [R, p],
        [np.zeros((1, 3)), 1]
    ])


def so3(v: np.ndarray) -> np.ndarray:
    assert v.shape == (3, 1)

    return np.array([
        [0, -v[2][0], v[1][0]],
        [v[2][0], 0, -v[0][0]],
        [-v[1][0], v[0][0], 0]
    ])


def Rot(omega: np.ndarray, theta: float) -> np.ndarray:
    assert omega.shape == (3, 1)
    omega_so3 = so3(omega)
    return np.eye(3) + np.sin(theta) * omega_so3 + (1 - np.cos(theta)) * omega_so3 @ omega_so3


class JointType(Enum):
    FIXED = 0
    REVOLUTE = 1


def link_pose(
        T0: np.ndarray,
        joint_type: JointType,
        joint_axis: Optional[np.ndarray] = None
) -> Callable[[float], np.ndarray]:
    if joint_type == JointType.FIXED:
        return lambda theta: T0

    return lambda theta: T0 @ T(Rot(joint_axis, theta), colvec([0, 0, 0]))


def colvec(coefficients: List[float]) -> np.ndarray:
    assert len(coefficients) == 3 or len(coefficients) == 6
    return np.array(coefficients).reshape(3, 1) if len(coefficients) == 3 else np.array(coefficients).reshape(6, 1)


def S(joint_type: JointType, joint_axis: Optional[np.ndarray] = None) -> np.ndarray:
    if joint_type == JointType.FIXED:
        return np.zeros((6, 1))

    return np.block([
        [joint_axis],
        [np.zeros((3, 1))]
    ])


def rotational_inertia(v: List[float]) -> np.ndarray:
    return np.diag(v)


def I_from_rotational_inertia(I_CC: np.ndarray, p_AC: np.ndarray, m: float) -> np.ndarray:
    assert I_CC.shape == (3, 3)
    assert p_AC.shape == (3, 1)

    return np.block([
        [I_CC + m * so3(p_AC) @ so3(p_AC).T, m * so3(p_AC)],
        [m * so3(p_AC).T, m * np.eye(3)]
    ])


class MultibodyDescription:
    def __init__(self):
        self.n_links = 3
        self.n_joints = 2
        self.parent = [-1, 0, 1]
        l0, l1, l2 = 0.5, 0.7, 0.4
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.poses = [T(R=np.eye(3), p=np.zeros((3, 1))),
                      T(R=np.eye(3), p=colvec([0, 0, l0])),
                      T(R=np.eye(3), p=colvec([0, 0, l1]))]
        self.Ts = [
            link_pose(
                T0=T(R=np.eye(3), p=np.zeros((3, 1))),
                joint_type=JointType.FIXED
            ),
            link_pose(
                T0=T(R=np.eye(3), p=colvec([0, 0, l0])),
                joint_type=JointType.REVOLUTE,
                joint_axis=colvec([1, 0, 0])
            ),
            link_pose(
                T0=T(R=np.eye(3), p=colvec([0, 0, l1])),
                joint_type=JointType.REVOLUTE,
                joint_axis=colvec([1, 0, 0])
            )
        ]
        self.S_ii = [
            S(joint_type=JointType.FIXED),
            S(joint_type=JointType.REVOLUTE, joint_axis=colvec([1, 0, 0])),
            S(joint_type=JointType.REVOLUTE, joint_axis=colvec([1, 0, 0]))
        ]
        m, r, w, h = 1., 0.05, 0.08, 0.06
        self.m = m
        self.r = r
        self.w = w
        self.h = h

        self.I_ii = [
            I_from_rotational_inertia(
                rotational_inertia([m * (3 * r ** 2 + l0 ** 2) / 12, m * (3 * r ** 2 + l0 ** 2) / 12, m * r ** 2 / 2]),
                colvec([0, 0, l0 / 2]),
                m
            ),
            I_from_rotational_inertia(
                rotational_inertia([m*(w**2+l1**2)/12, m*(h**2+l1**2)/12, m*(h**2+w**2)/12]),
                colvec([0, 0, l1 / 2]),
                m
            ),
            I_from_rotational_inertia(
                rotational_inertia([m*(w**2+l2**2)/12, m*(h**2+l2**2)/12, m*(h**2+w**2)/12]),
                colvec([0, 0, l2 / 2]),
                m
            )
        ]

    def to_modern_robotics_description(self):
        l0, l1, l2 = self.l0, self.l1, self.l2
        Mlist = np.array([
            T(R=np.eye(3), p=colvec([0, 0, l0/2 + l1/2])),
            T(R=np.eye(3), p=colvec([0, 0, l1/2 + l2/2])),
            T(R=np.eye(3), p=colvec([0, 0, l2 / 2]))
        ])
        Slist = np.hstack([
            colvec([1, 0, 0, 0, l0/2, 0]),
            colvec([1, 0, 0, 0, l0/2+l1, 0])
        ])

        frames_conversions = [
            T(R=np.eye(3), p=colvec([0, 0, -l0/2])),
            T(R=np.eye(3), p=colvec([0, 0, -l1/2])),
            T(R=np.eye(3), p=colvec([0, 0, -l2/2]))
        ]

        m, r, w, h = self.m, self.r, self.w, self.h
        Glist = [
            # centroidal_inertia(
            #     rotational_inertia([m * (3 * r ** 2 + l0 ** 2) / 12, m * (3 * r ** 2 + l0 ** 2) / 12, m * r ** 2 / 2]),
            #     m
            # ),
            centroidal_inertia(
                rotational_inertia(
                    [m * (w ** 2 + l1 ** 2) / 12, m * (h ** 2 + l1 ** 2) / 12, m * (h ** 2 + w ** 2) / 12]),
                m
            ),
            centroidal_inertia(
                rotational_inertia(
                    [m * (w ** 2 + l2 ** 2) / 12, m * (h ** 2 + l2 ** 2) / 12, m * (h ** 2 + w ** 2) / 12]),
                m
            )
        ]

        return Mlist, Slist, frames_conversions, Glist


def centroidal_inertia(rotational_inertia: np.ndarray, m: float) -> np.ndarray:
    assert rotational_inertia.shape == (3, 3)

    return np.block([
        [rotational_inertia, np.zeros((3, 3))],
        [np.zeros((3, 3)), m * np.eye(3)]
    ])


def Ad(T_AB: np.ndarray) -> np.ndarray:
    assert T_AB.shape == (4, 4)

    R = T_AB[0:3, 0:3]
    p = T_AB[0:3, 3].reshape(3, 1)

    return np.block([
        [R, np.zeros((3, 3))],
        [so3(p) @ R, R]
    ])


def Tinv(T: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4)

    R = T[0:3, 0:3]
    p = T[0:3, 3].reshape(3, 1)

    Rt = R.T

    return np.block([
        [R.T, -Rt @ p],
        [np.zeros((1, 3)), 1]
    ])


def Vx(V: np.ndarray) -> np.ndarray:
    omega = colvec(V[0:3, 0])
    v = colvec(V[3:6, 0])

    omega_so3 = so3(omega)

    return np.block([
        [omega_so3, np.zeros((3, 3))],
        [so3(v), omega_so3]
    ])


def Vx_star(V: np.ndarray) -> np.ndarray:
    return -Vx(V).T


def se3(V):
    return np.r_[
        np.c_[so3(colvec([V[0], V[1], V[2]])), [V[3], V[4], V[5]]],
        np.zeros((1, 4))
    ]


def Xinv(X: np.ndarray) -> np.ndarray:
    R = X[0:3, 0:3]
    p = colvec([0, 0, 0.5])

    return np.block([
        [R.T, np.zeros((3, 3))],
        [-R.T@so3(p), R.T]
    ])


def rnea(theta: np.ndarray, theta_dot: np.ndarray, theta_dot_dot: np.ndarray,
         model: MultibodyDescription) -> np.ndarray:
    V_ii = [np.zeros((6, 1))]
    A_ii = [-colvec([0, 0, 0, 0, 0, -9.8])]  # error fixed
    F_ii = [np.empty((6, 1)) for _ in range(model.n_links)]

    for i in range(1, model.n_links):
        X_ipi = Ad(MatrixExp6(se3(model.S_ii[i] * -theta[i])) @ Tinv(model.poses[i]))
        V_ii.append(
            X_ipi @ V_ii[i-1] + model.S_ii[i] * theta_dot[i]
        )
        A_ii.append(
            X_ipi @ A_ii[i-1] + model.S_ii[i] * theta_dot_dot[i] + Vx(V_ii[i]) @ model.S_ii[i] * theta_dot[i]
        )
        # print(f'n_link {i}')
        # print(model.I_ii[i] @ A_ii[i])
        # print(Vx_star(V_ii[i]) @ (model.I_ii[i] @  V_ii[i]))
        # print('++++++++++++++++++++++++++++++++++++++')
        F_ii[i] = model.I_ii[i] @ A_ii[i] + Vx_star(V_ii[i]) @ (model.I_ii[i] @  V_ii[i])

    tau = np.zeros([model.n_links, 1])
    for i in range(model.n_links - 1, 0, -1):
        tau[i, 0] = model.S_ii[i].T @ F_ii[i]
        X_pii_star = Ad(MatrixExp6(se3(model.S_ii[i] * -theta[i])) @ Tinv(model.poses[i])).T
        F_ii[i-1] += X_pii_star @ F_ii[i]

    return np.hstack(F_ii)


def InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist, conversions, model):
    n = len(thetalist)
    Mi = np.eye(4)
    Ai = np.zeros((6, n))
    AdTi = [[None]] * (n + 1)
    Vi = np.zeros((6, n + 1))
    Vdi = np.zeros((6, n + 1))
    Vdi[:, 0] = np.r_[[0, 0, 0], -np.array(g)]
    AdTi[n] = Adjoint(TransInv(Mlist[n]))
    Fi = np.array(Ftip).copy()
    taulist = np.zeros(n)
    for i in range(n):
        Mi = np.dot(Mi, Mlist[i])
        Ai[:, i] = np.dot(Adjoint(TransInv(Mi)), np.array(Slist)[:, i])
        AdTi[i] = Adjoint(np.dot(MatrixExp6(VecTose3(Ai[:, i] * \
                                                     -thetalist[i])), \
                                 TransInv(Mlist[i])))
        Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dthetalist[i]
        Vdi[:, i + 1] = np.dot(AdTi[i], Vdi[:, i]) \
                        + Ai[:, i] * ddthetalist[i] \
                        + np.dot(ad(Vi[:, i + 1]), Ai[:, i]) * dthetalist[i]

    F_ii = np.zeros((6, n))
    for i in range(n - 1, -1, -1):
        # I = I_from_rotational_inertia(
        #     Glist[i+1][0:3, 0:3],
        #     colvec([0, 0, 0.4 / 2]),
        #     1
        # )
        # print(f'n_link {i+1}')
        # print(np.dot(np.array(Glist[i]), Vdi[:, [i + 1]]))
        # print(I @ (Ad(Tinv(conversions[2])) @ Vdi[:, [i + 1]]))
        # print(np.dot(np.array(ad(Ad(Tinv(conversions[2])) @ Vi[:, [i + 1]])).T, \
        #        np.dot(I, Ad(Tinv(conversions[2])) @ Vi[:, [i + 1]])))
        # print('++++++++++++++++++++++++++++++++++++++')
        Fi = np.dot(np.array(AdTi[i + 1]).T, Fi) \
             + np.dot(np.array(Glist[i]), colvec(Vdi[:, i + 1])) \
             - np.dot(np.array(ad(Vi[:, i + 1])).T, \
                      np.dot(np.array(Glist[i]), colvec(Vi[:, i + 1])))
        F_ii[:6, i] = Fi[:, 0]
        taulist[i] = np.dot(np.array(Fi).T, Ai[:, i])

    return F_ii


if __name__ == '__main__':
    desc = MultibodyDescription()
    r = np.array([
        [1, 0, 0],
        [0, np.cos(0.2), -np.sin(0.2)],
        [0, np.sin(0.2), np.cos(0.2)]
    ])
    # print(desc.S_ii)
    # print(r)
    # print('=========================')
    # print(desc.Ts[1](0.2))
    # print('=========================')
    # print(Ad(desc.Ts[1](0.2)))

    thetalist = np.array([0.1, 0.1])
    dthetalist = np.array([0.1, 0.2])
    ddthetalist = np.array([2, 1.5])

    Mlist, Slist, conversions, Glist = desc.to_modern_robotics_description()
    Ftip = np.zeros((6, 1))
    g = np.array([0, 0, -9.8])

    print('=========================')
    Fi_ = rnea(np.concatenate(([0], thetalist)), np.concatenate(([0], dthetalist)),
               np.concatenate(([0], ddthetalist)), desc)
    print(Fi_)


    print('=========================')
    REsi = InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist, conversions, desc)
    for i in range(desc.n_links-1):
        # REsi[:, i] = Ad(Tinv(conversions[i])) @ REsi[:, i]
        REsi[:, [i]] = Ad(conversions[i+1]).T @ REsi[:, [i]]
    print(REsi)

    # thetalist = np.array([0.1, 0.1, 0.1])
    # dthetalist = np.array([0.1, 0.2, 0.3])
    # ddthetalist = np.array([2, 1.5, 1])
    # g = np.array([0, 0, -9.8])
    # Ftip = np.array([1, 1, 1, 1, 1, 1])
    # M01 = np.array([[1, 0, 0, 0],
    #                 [0, 1, 0, 0],
    #                 [0, 0, 1, 0.089159],
    #                 [0, 0, 0, 1]])
    # M12 = np.array([[0, 0, 1, 0.28],
    #                 [0, 1, 0, 0.13585],
    #                 [-1, 0, 0, 0],
    #                 [0, 0, 0, 1]])
    # M23 = np.array([[1, 0, 0, 0],
    #                 [0, 1, 0, -0.1197],
    #                 [0, 0, 1, 0.395],
    #                 [0, 0, 0, 1]])
    # M34 = np.array([[1, 0, 0, 0],
    #                 [0, 1, 0, 0],
    #                 [0, 0, 1, 0.14225],
    #                 [0, 0, 0, 1]])
    # G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
    # G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
    # G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
    # Glist = np.array([G1, G2, G3])
    # Mlist = np.array([M01, M12, M23, M34])
    # Slist = np.array([[1, 0, 1, 0, 1, 0],
    #                   [0, 1, 0, -0.089, 0, 0],
    #                   [0, 1, 0, -0.089, 0, 0.425]]).T
    #
    # InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist, None, None)




