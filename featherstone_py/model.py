import numpy as np

from featherstone_py.joint import JointMetadata, RevoluteJoint, JointAxis
from typing import NamedTuple, List, Optional


# TODO make a proper docstring
# Multibody model description
class MultibodyModel(NamedTuple):
    # number of bodies excluding the base
    # it's equal to the number of joints, because the base is not counted
    n_bodies: int
    joints: List[JointMetadata]

    # parent[i] is the index of the parent of the i-th body
    # base is excluded from the list and the direct children of the base are having parent index -1
    parent: List[int]

    # spatial transform of frame parent[i] to the frame i when joint i is at zero position X_i_i-1
    X_tree: List[np.ndarray]

    # spatial inertia of body i, expressed in body i coordinates.
    I: List[np.ndarray]

    T_n_ee: Optional[np.ndarray] = None
