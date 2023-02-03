import numpy as np

from featherstone_py.joint import JointMetadata, RevoluteJoint, JointAxis
from featherstone_py.spatial import colvec, T, I_from_rotational_inertia, rotational_inertia, centroidal_inertia, Ad

from featherstone_py.model import MultibodyModel


# A dummy model of a 2-link pendulum
# Later this model will be result of parsing of URDF file
# TODO add documentation
class DoublePendulum:
    def __init__(self, l0=0.5, l1=0.7, l2=0.4, m=1., r=0.05, w=0.08, h=0.06):
        self.l0, self.l1, self.l2 = l0, l1, l2
        self.m, self.r, self.w, self.h = m, r, w, h

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

    def to_featherstone_notation(self):
        l0, l1, l2 = self.l0, self.l1, self.l2

        n_bodies = 2

        parent = [-1, 0]
        joints = [RevoluteJoint(JointAxis.X), RevoluteJoint(JointAxis.X)]

        X_tree = [
            Ad(T(R=np.eye(3), p=-colvec([0, 0, l0]))),
            Ad(T(R=np.eye(3), p=-colvec([0, 0, l1])))
        ]

        m, r, w, h = self.m, self.r, self.w, self.h

        I = [
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

        return MultibodyModel(n_bodies, joints, parent, X_tree, I)
