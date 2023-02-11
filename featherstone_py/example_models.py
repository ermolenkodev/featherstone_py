from typing import List, Tuple

import numpy as np

from featherstone_py.joint import RevoluteJoint, JointAxis
from featherstone_py.spatial import colvec, T, I_from_rotational_inertia, rotational_inertia, centroidal_inertia, Ad, \
    Tinv

from featherstone_py.model import MultibodyModel


class DoublePendulum:
    """
    A class to represent a 2-link pendulum.
    The first link is cylindrical and the rest are boxes.

    Attributes
    ----------
    l0 : float
        length of the first link
    l1 : float
        length of the second link
    l2 : float
        length of the third link
    m : float
        mass of each link
    r : float
        radius of the first link
    w : float
        width of the second and third link
    h : float
        height of the second and third link

    Methods
    ----------
    to_modern_robotics_notation(self):
        Returns the quantities representing the model in the Modern Robotics book notation.

    to_featherstone_notation(self):
        Returns MultibodyModel the quantities representing the model in the Featherstone's book notation.
    """

    def __init__(self, l0=0.5, l1=0.7, l2=0.4, m=1., r=0.05, w=0.08, h=0.06):
        self.l0, self.l1, self.l2 = l0, l1, l2
        self.m, self.r, self.w, self.h = m, r, w, h

    def to_modern_robotics_notation(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Mlist: list of link frames i relative to i-1 at the home position.\f
        Glist: Spatial inertia matrices Gi of the links relative to the center of mass expressed in the link local frame.\f
        Slist: Screw axes Si of the joints in a space (inertial) frame, in the format of a matrix with axes as the columns.\f
        frames_conversions: the frames conversions from the link frames expected by Featherstone's algorithms to the link frames expected by Modern Robotics' algorithms. In the MR notation, the links are located in the center of mass, while in the Featherstone's notation, the link's frames are located at the joints.
        :returns: the quantities representing the model in the Modern Robotics book notation.
        """  # noqa: D301
        l0, l1, l2 = self.l0, self.l1, self.l2
        Mlist = np.array([
            T(R=np.eye(3), p=colvec([0, 0, l0 / 2 + l1 / 2])),
            T(R=np.eye(3), p=colvec([0, 0, l1 / 2 + l2 / 2])),
            T(R=np.eye(3), p=colvec([0, 0, l2 / 2]))
        ])
        Slist = np.hstack([
            colvec([1, 0, 0, 0, l0 / 2, 0]),
            colvec([1, 0, 0, 0, l0 / 2 + l1, 0])
        ])

        frames_conversions = [
            T(R=np.eye(3), p=colvec([0, 0, -l1 / 2])),
            T(R=np.eye(3), p=colvec([0, 0, -l2 / 2]))
        ]

        m, r, w, h = self.m, self.r, self.w, self.h
        Glist = np.array([
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
        ])

        return Mlist, Slist, frames_conversions, Glist

    def to_featherstone_notation(self) -> MultibodyModel:
        """
        :returns: the multibody model in the Featherstone's book notation.        """
        l0, l1, l2 = self.l0, self.l1, self.l2

        n_bodies = 2

        parent = [-1, 0]
        joints = [RevoluteJoint(JointAxis.X), RevoluteJoint(JointAxis.X)]

        X_tree = [
            Ad(T(R=np.eye(3), p=-colvec([0, 0, l0]))),
            Ad(T(R=np.eye(3), p=-colvec([0, 0, l1]))),
        ]

        m, r, w, h = self.m, self.r, self.w, self.h

        I = [
            I_from_rotational_inertia(
                rotational_inertia(
                    [m * (w ** 2 + l1 ** 2) / 12, m * (h ** 2 + l1 ** 2) / 12, m * (h ** 2 + w ** 2) / 12]),
                colvec([0, 0, l1 / 2]),
                m
            ),
            I_from_rotational_inertia(
                rotational_inertia(
                    [m * (w ** 2 + l2 ** 2) / 12, m * (h ** 2 + l2 ** 2) / 12, m * (h ** 2 + w ** 2) / 12]),
                colvec([0, 0, l2 / 2]),
                m
            )
        ]

        T_tree = [
            T(R=np.eye(3), p=colvec([0, 0, l0])),
            T(R=np.eye(3), p=colvec([0, 0, l1])),
            T(R=np.eye(3), p=colvec([0, 0, l2]))
        ]

        T_ee = T(R=np.eye(3), p=colvec([0, 0, l2]))

        return MultibodyModel(n_bodies, joints, parent, X_tree, I, T_ee)
