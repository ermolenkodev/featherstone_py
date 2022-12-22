from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

from rnea import colvec, T, I_from_rotational_inertia, rotational_inertia, centroidal_inertia, Ad
from spatial import rotx, roty, rotz


class JointAxis(Enum):
    X = 0
    Y = 1
    Z = 2


class JointMetadata(ABC):
    @abstractmethod
    def joint_transform(self, theta):
        pass

    @abstractmethod
    def screw_axis(self):
        pass


class RevoluteJoint(JointMetadata):
    def __init__(self, axis: JointAxis):
        super().__init__()
        self.axis = axis

    def joint_transform(self, theta):
        if self.axis == JointAxis.X:
            return rotx(theta)
        elif self.axis == JointAxis.Y:
            return roty(theta)
        elif self.axis == JointAxis.Z:
            return rotz(theta)

    def screw_axis(self):
        if self.axis == JointAxis.X:
            return colvec([1, 0, 0, 0, 0, 0])
        elif self.axis == JointAxis.Y:
            return colvec([0, 1, 0, 0, 0, 0])
        elif self.axis == JointAxis.Z:
            return colvec([0, 0, 1, 0, 0, 0])


# Multibody model description
# Currently it's a dummy model of a 2-link pendulum
# Later this model will be result of parsing of URDF file
class MultibodyModel:
    def __init__(self):
        # number of bodies excluding the base
        # it's equal to the number of joints
        self.n_bodies = 2

        self.parent = [0, 1]
        self.joints = [RevoluteJoint(JointAxis.X), RevoluteJoint(JointAxis.X)]

        l0, l1, l2 = 0.5, 0.7, 0.4
        self.l0, self.l1, self.l2 = l0, l1, l2
        # spatial transform of frame parent[i] to the frame i when joint i is at zero position X_i_i-1
        self.X_tree = [
            Ad(T(R=np.eye(3), p=-colvec([0, 0, l0]))),
            Ad(T(R=np.eye(3), p=-colvec([0, 0, l1])))
        ]

        m, r, w, h = 1., 0.05, 0.08, 0.06
        self.m, self.r, self.w, self.h = m, r, w, h
        # spatial inertia of body i, expressed in body i coordinates.
        self.I = [
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

        self.gravity = colvec([0, 0, -9.81])

    def as_tuple(self):
        return self.n_bodies, self.parent, self.joints, self.X_tree, self.I, self.gravity

    def to_modern_robotics_notation(self):
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