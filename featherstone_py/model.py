import numpy as np

from featherstone_py.joint import JointMetadata, RevoluteJoint, JointAxis
from typing import NamedTuple, List, Optional


class MultibodyModel(NamedTuple):
    """
    Multibody model description in Featherstone's notation.
    
    Attributes
    ----------
    n_bodies : int
        Number of bodies excluding the base. It's equal to the number of joints, because the base is not counted.
    joints: List[JointMetadata]
        List of objects describing the joints of the model. Each joint provides the screw axis and the joint transform.
    parent: List[int]
        parent[i] is the index of the parent of the i-th body.
        base is excluded from the list and the direct children of the base are having parent index -1
    X_tree: List[np.ndarray]
        X_tree[i] is a X_i_i-1(0) - spatial transforms of frame parent[i] to the frame i when joint i is at zero position.
    I: List[np.ndarray]
        I[i] is a spatial inertia of body i, expressed in body i coordinates.
    T_n_ee: Optional[np.ndarray]
        T_n_ee is a homogenous transform from the last body frame to the end-effector frame.
        It is required to consider the ee force exerted on the environment expressed in ee frame (Modern Robotics's Ftip) .
    """  # noqa: D301
    n_bodies: int
    joints: List[JointMetadata]
    parent: List[int]
    X_tree: List[np.ndarray]
    I: List[np.ndarray]
    T_n_ee: Optional[np.ndarray] = None
