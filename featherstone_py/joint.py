import numpy as np

from featherstone_py.spatial import rotx, roty, rotz, colvec
from enum import Enum
from abc import ABC, abstractmethod


class JointAxis(Enum):
    X = 0
    Y = 1
    Z = 2


class JointMetadata(ABC):
    """
    Interface for different joint types.
    Each joint type should return the corresponding screw axis and joint transform.
    """  # noqa: D301, E501

    @abstractmethod
    def joint_transform(self, theta: float) -> np.ndarray:
        """
        :param theta:
        :return: the configuration dependent spatial transformation matrix corresponding to the motion of the joint.
        """  # noqa: D301, E501
        pass

    @abstractmethod
    def screw_axis(self) -> np.ndarray:
        """
        :returns: the screw axis of the joint.
        """
        pass


class RevoluteJoint(JointMetadata):
    """
    3d revolute joint

    Attributes:
        axis: the axis of rotation of the joint.
    """

    def __init__(self, axis: JointAxis):
        super().__init__()
        self.axis = axis

    def joint_transform(self, theta):
        """
        Joint transform for revolute joint is a spatial rotation about the joint axis.
        :param theta:
        :return: Spatial rotation matrix.
        """
        if self.axis == JointAxis.X:
            return rotx(theta)
        elif self.axis == JointAxis.Y:
            return roty(theta)
        elif self.axis == JointAxis.Z:
            return rotz(theta)

    def screw_axis(self) -> np.ndarray:
        """
        The screw axis of the revolute joint has a unit vector angular velocity component in the direction of the joint axis
        and zero linear velocity component.
        :returns: the screw axis of the joint.
        """  # noqa: D301, E501
        if self.axis == JointAxis.X:
            return colvec([1, 0, 0, 0, 0, 0])
        elif self.axis == JointAxis.Y:
            return colvec([0, 1, 0, 0, 0, 0])
        elif self.axis == JointAxis.Z:
            return colvec([0, 0, 1, 0, 0, 0])
