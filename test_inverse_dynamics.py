import unittest

import numpy as np

from inverse_dynamics import rnea
from model import MultibodyModel
from mr import InverseDynamics


class TestInverseDynamics(unittest.TestCase):
    def test_inverse_dynamics(self):
        q = np.array([0.1, 0.1])
        qd = np.array([0.1, 0.2])
        qdd = np.array([2, 1.5])

        model = MultibodyModel()

        Mlist, Slist, frames_conversions, Glist = model.to_modern_robotics_notation()
        Ftip = np.zeros(6)
        g = np.array([0, 0, -9.81])

        np.testing.assert_allclose(
            rnea(model, q, qd, qdd)[:, 0],
            InverseDynamics(q, qd, qdd, g, Ftip, Mlist, Glist, Slist)
        )
