import numpy as np

from rnea import so3


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
