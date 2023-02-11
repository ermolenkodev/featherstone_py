from typing import List, Union

import numpy as np

# TODO research shape safety


def rotx(theta: float) -> np.ndarray:
    """
    Spatial coordinate transform (X-axis rotation).
    Rotation of frame B relative to frame S about the X-axis in S by an angle theta.
    :param theta:
    :return: X-axis spatial rotation matrix.
    """
    return spatial_rotation(Rx(theta))


def roty(theta: float) -> np.ndarray:
    """
    Spatial coordinate transform (Y-axis rotation).
    Rotation of frame B relative to frame S about the Y-axis in S by an angle theta.
    :param theta:
    :return: Y-axis spatial rotation matrix.
    """
    return spatial_rotation(Ry(theta))


def rotz(theta: float) -> np.ndarray:
    """
    Spatial coordinate transform (Z-axis rotation).
    Rotation of frame B relative to frame S about the Z-axis in S by an angle theta.
    :param theta:
    :return: Z-axis spatial rotation matrix.
    """
    return spatial_rotation(Rz(theta))


def Rx(theta: float) -> np.ndarray:
    """
    :param theta:
    :return: 3d rotation matrix about X-axis by an angle theta
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])


def Ry(theta: float) -> np.ndarray:
    """
    :param theta:
    :return: 3d rotation matrix about Y-axis by an angle theta
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])


def Rz(theta: float) -> np.ndarray:
    """
    :param theta:
    :return: 3d rotation matrix about Z-axis by an angle theta
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])


def spatial_rotation(R: np.ndarray) -> np.ndarray:
    """
    Constructs a spatial rotation matrix from a SO(3) matrix.
    :param R: 3d rotation matrix.
    :return: Spatial rotation matrix.
    """
    return np.block([
        [R, np.zeros((3, 3))],
        [np.zeros((3, 3)), R]
    ])


def spatial_translation(p: np.ndarray) -> np.ndarray:
    """
    Constructs a spatial translation matrix from a 3d displacement vector.
    :param p: 3d displacement vector.
    :return: Spatial translation matrix.
    """
    R = np.eye(3)

    # TODO check if it's correct
    # TODO check how numpy handles copying
    return np.block([
        [R, np.zeros((3, 3))],
        [-so3(p), R]
    ])


def T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Constructs a homogeneous transformation matrix from a rotation matrix and a displacement vector.
    :param R: SO(3) rotation matrix.
    :param p: 3d displacement vector.
    :return: Homogeneous transformation matrix.
    """
    assert R.shape == (3, 3)
    assert p.shape == (3, 1)

    return np.block([
        [R, p],
        [np.zeros((1, 3)), 1]
    ])


def so3(v: np.ndarray) -> np.ndarray:
    """
    Converts a 3d vector to a skew-symmetric representation.
    :param v: 3d vector.
    :return: Skew-symmetric matrix.
    """
    assert v.shape == (3, 1)

    return np.array([
        [0, -v[2][0], v[1][0]],
        [v[2][0], 0, -v[0][0]],
        [-v[1][0], v[0][0], 0]
    ])


def Rot(omega: np.ndarray, theta: float) -> np.ndarray:
    """
    Compute the rotation operator matrix corresponding to a rotation about the axis omega by an angle theta.
    :param omega: 3d rotation axis.
    :param theta: Rotation angle.
    :return: SO(3) rotation matrix.
    """
    assert omega.shape == (3, 1)
    omega_so3 = so3(omega)
    return np.eye(3) + np.sin(theta) * omega_so3 + (1 - np.cos(theta)) * omega_so3 @ omega_so3


def colvec(coefficients: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Converts a 1d input to a column vector.
    :param coefficients: Array like input.
    :return: column vector.
    """
    if isinstance(coefficients, list):
        return np.array(coefficients).reshape(len(coefficients), 1)
    elif isinstance(coefficients, np.ndarray):
        # TODO check if shape is already (n, 1)
        return coefficients.reshape(len(coefficients), 1)


def rotational_inertia(v: List[float]) -> np.ndarray:
    """
    Constructs a rotational inertia matrix from a list of principal moments of inertia.
    :param v: vector of principal moments of inertia.
    :return: 3d rotational inertia matrix.
    """
    return np.diag(v)


def I_from_rotational_inertia(I_CC: np.ndarray, p_AC: np.ndarray, m: float) -> np.ndarray:
    """
    Computes a spatial inertia matrix expressed in the link frame from a rotational inertia matrix, a displacement vector, and a mass.
    :param I_CC: central rotational inertia matrix expressed in frame with origin at C and axes aligned with link frame.
    :param p_AC: position of the center of mass relative to the link frame origin.
    :param m: mass of the link.
    :return: Spatial inertia matrix expressed in the link frame.
    """  # noqa: D301
    assert I_CC.shape == (3, 3)
    assert p_AC.shape == (3, 1)

    return np.block([
        [I_CC + m * so3(p_AC) @ so3(p_AC).T, m * so3(p_AC)],
        [m * so3(p_AC).T, m * np.eye(3)]
    ])


def centroidal_inertia(rotational_inertia: np.ndarray, m: float) -> np.ndarray:
    """
    Computes the spatial inertia matrix about CoM expressed in the CoM frame from a rotational inertia matrix and a mass.
    :param rotational_inertia: 3d rotational inertia matrix.
    :param m: mass of the link.
    :return: Spatial inertia matrix about CoM expressed in the CoM frame.
    """  # noqa: D301
    assert rotational_inertia.shape == (3, 3)

    return np.block([
        [rotational_inertia, np.zeros((3, 3))],
        [np.zeros((3, 3)), m * np.eye(3)]
    ])


def Ad(T_AB: np.ndarray) -> np.ndarray:
    """
    Computes the adjoint transformation matrix from a homogeneous transformation matrix.
    :param T_AB: homogeneous transformation matrix from frame A to frame B.
    :return: Corresponding adjoint transformation matrix.
    """
    assert T_AB.shape == (4, 4)

    R = T_AB[0:3, 0:3]
    p = T_AB[0:3, 3].reshape(3, 1)

    return np.block([
        [R, np.zeros((3, 3))],
        [so3(p) @ R, R]
    ])


def Tinv(T: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a homogeneous transformation matrix.
    :param T: homogeneous transformation matrix.
    :return: inverse of T.
    """
    assert T.shape == (4, 4)

    R = T[0:3, 0:3]
    p = T[0:3, 3].reshape(3, 1)

    Rt = R.T

    return np.block([
        [Rt, -Rt @ p],
        [np.zeros((1, 3)), 1]
    ])


def Vx(V: np.ndarray) -> np.ndarray:
    """
    Computes matrix corresponding to the spatial velocity cross product operation from twist V.
    :param V: twist
    :return: Spatial velocity cross product matrix.
    """
    omega = colvec(V[0:3, 0])
    v = colvec(V[3:6, 0])

    omega_so3 = so3(omega)

    return np.block([
        [omega_so3, np.zeros((3, 3))],
        [so3(v), omega_so3]
    ])


def Vx_star(V: np.ndarray) -> np.ndarray:
    """
    Computes matrix corresponding to the spatial force cross product operation from wrench twist V.
    :param V: twist
    :return: Spatial force cross product matrix.
    """
    return -Vx(V).T


def se3(V: np.ndarray) -> np.ndarray:
    """
    Converts a twist to a corresponding matrix representation. Analogues to skew-symmetric representation for angular velocity vectors.
    :param V: twist
    :return: Matrix representation of twist.
    """
    return np.r_[
        np.c_[so3(colvec([V[0], V[1], V[2]])), [V[3], V[4], V[5]]],
        np.zeros((1, 4))
    ]
