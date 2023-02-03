from featherstone_py.spatial import rotx, roty, rotz, colvec
from enum import Enum
from abc import ABC, abstractmethod


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
