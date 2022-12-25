from typing import List

import numpy as np

# TODO research shape safety

# Spatial coordinate transform (X-axis rotation).
# Rotation of frame B relative to frame S about the X-axis in S by an angle θ.
def rotx(theta: float) -> np.ndarray:
    # maybe it's better to instantiate a matrix and fill it with values directly
    # need to check performance
    return spatial_rotation(Rx(theta))


# Spatial coordinate transform (Y-axis rotation).
# Rotation of frame B relative to frame S about the Y-axis in S by an angle theta.
def roty(theta: float) -> np.ndarray:
    return spatial_rotation(Ry(theta))


# Spatial coordinate transform (Z-axis rotation).
# Rotation of frame B relative to frame S about the Z-axis in S by an angle theta.
def rotz(theta: float) -> np.ndarray:
    return spatial_rotation(Rz(theta))


# 3d rotation matrix about X-axis by an angle theta
def Rx(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])


# 3d rotation matrix about Y-axis by an angle theta
def Ry(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])


# 3d rotation matrix about Z-axis by an angle theta
def Rz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])


def spatial_rotation(R: np.ndarray) -> np.ndarray:
    return np.block([
        [R, np.zeros((3, 3))],
        [np.zeros((3, 3)), R]
    ])


def spatial_translation(p: np.ndarray) -> np.ndarray:
    R = np.eye(3)

    # TODO check if it's correct
    # TODO check how numpy handles copying
    return np.block([
        [R, np.zeros((3, 3))],
        [-so3(p), R]
    ])

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


def colvec(coefficients: List[float]) -> np.ndarray:
    assert len(coefficients) == 3 or len(coefficients) == 6
    return np.array(coefficients).reshape(3, 1) if len(coefficients) == 3 else np.array(coefficients).reshape(6, 1)


def rotational_inertia(v: List[float]) -> np.ndarray:
    return np.diag(v)


def I_from_rotational_inertia(I_CC: np.ndarray, p_AC: np.ndarray, m: float) -> np.ndarray:
    assert I_CC.shape == (3, 3)
    assert p_AC.shape == (3, 1)

    return np.block([
        [I_CC + m * so3(p_AC) @ so3(p_AC).T, m * so3(p_AC)],
        [m * so3(p_AC).T, m * np.eye(3)]
    ])


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
        [Rt, -Rt @ p],
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

