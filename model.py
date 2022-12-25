import numpy as np

from joint import RevoluteJoint, JointAxis
from spatial import colvec, T, I_from_rotational_inertia, rotational_inertia, centroidal_inertia, Ad


# Multibody model description
# Currently it's a dummy model of a 2-link pendulum
# Later this model will be result of parsing of URDF file
class MultibodyModel:
    def __init__(self):
        # number of bodies excluding the base
        # it's equal to the number of joints
        self.n_bodies = 2

        # parent[i] is the index of the parent of the i-th body
        # base is excluded from the list and the direct children of the base are having parent index -1
        self.parent = [-1, 0]
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
