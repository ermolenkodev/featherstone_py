import unittest

import numpy as np

from inverse_dynamics import rnea, RNEAImpl
from model import MultibodyModel
from modern_robotics import InverseDynamics, Adjoint, TransInv, MatrixExp6, VecTose3, ad
from spatial import Ad


class TestInverseDynamics(unittest.TestCase):
    def test_torques_calculation(self):
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

    def test_wrenches_calculation(self):
        q = np.array([0.1, 0.1])
        qd = np.array([0.1, 0.2])
        qdd = np.array([2, 1.5])

        model = MultibodyModel()

        Mlist, Slist, frames_conversions, Glist = model.to_modern_robotics_notation()
        Ftip = np.zeros(6)
        g = np.array([0, 0, -9.81])

        # Modern robotics inverse dynamics function modified to return wrenches instead of torques
        # It is useful only for demonstration purposes
        def InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
            n = len(thetalist)
            Mi = np.eye(4)
            Ai = np.zeros((6, n))
            AdTi = [[None]] * (n + 1)
            Vi = np.zeros((6, n + 1))
            Vdi = np.zeros((6, n + 1))
            Vdi[:, 0] = np.r_[[0, 0, 0], -np.array(g)]
            AdTi[n] = Adjoint(TransInv(Mlist[n]))
            Fi = np.array(Ftip).copy()

            for i in range(n):
                Mi = np.dot(Mi, Mlist[i])
                Ai[:, i] = np.dot(Adjoint(TransInv(Mi)), np.array(Slist)[:, i])
                AdTi[i] = Adjoint(np.dot(MatrixExp6(VecTose3(Ai[:, i] * -thetalist[i])), TransInv(Mlist[i])))
                Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dthetalist[i]
                Vdi[:, i + 1] = np.dot(AdTi[i], Vdi[:, i]) \
                                + Ai[:, i] * ddthetalist[i] \
                                + np.dot(ad(Vi[:, i + 1]), Ai[:, i]) * dthetalist[i]

            F = []
            for i in range(n - 1, -1, -1):
                Fi = np.dot(np.array(AdTi[i + 1]).T, Fi) \
                     + np.dot(np.array(Glist[i]), Vdi[:, i + 1]) \
                     - np.dot(np.array(ad(Vi[:, i + 1])).T, \
                              np.dot(np.array(Glist[i]), Vi[:, i + 1]))
                F.insert(0, Fi)

            return F

        F_mr = InverseDynamics(q, qd, qdd, g, Ftip, Mlist, Glist, Slist)

        algorithm = RNEAImpl()
        algorithm.run(model, q, qd, qdd)

        F = algorithm.F

        for i in range(len(F_mr)):
            np.testing.assert_allclose(Ad(frames_conversions[i]).T @ F_mr[i], F[i][:, 0])
