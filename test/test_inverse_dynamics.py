import pytest

import numpy as np

from example_models import DoublePendulum
from featherstone_py.inverse_dynamics import rnea, RNEAImpl
from modern_robotics import InverseDynamics, Adjoint, TransInv, MatrixExp6, VecTose3, ad
from featherstone_py.spatial import Ad, colvec, T, Tinv


@pytest.mark.parametrize(
    "q,qd,qdd,g,f_tip",
    [([0.1, 0.1], [0.1, 0.2], [2, 1.5], [0, 0, -9.81], np.zeros(6)),
     ([1, 0.2], [0., 0.2], [0.1, 0.], [0, 0, -9.81], np.zeros(6)),
     ([1, 0.2], [0., 0.2], [0.1, 0.], [0, 0, -9.81], None),
     ([1, 0.2], [0., 0.2], [0.1, 0.], [0, 0, -9.81], np.array([1, 2, 3, 4, 5, 6]))]
)
def test_torques_calculation(double_pendulum_models, q, qd, qdd, g, f_tip):
    q, qd, qdd, g = np.array(q), np.array(qd), np.array(qdd), np.array(g)

    m_feath_notation, m_mr_notation = double_pendulum_models
    Mlist, Slist, frames_conversions, Glist = m_mr_notation

    np.testing.assert_allclose(
        rnea(m_feath_notation, q, qd, qdd, g, f_tip)[:, 0],
        InverseDynamics(q, qd, qdd, g, np.zeros(6) if f_tip is None else f_tip, Mlist, Glist, Slist)
    )

@pytest.mark.parametrize(
    "q,qd,qdd,g,f_tip",
     [([0, 0], [0., 0], [0, 0.], [0, 0, -9.81], np.array([1, 2, 3, 4, 5, 6])),
      ([0, 0], [0., 0], [0, 0.], [0, 0, -9.81], np.array([1, 1, 1, 1, 1, 1]))]
)
def test_torques_calculation_with_eternal_forces(double_pendulum_models, q, qd, qdd, g, f_tip):
    q, qd, qdd, g = np.array(q), np.array(qd), np.array(qdd), np.array(g)

    m_feath_notation, m_mr_notation = double_pendulum_models
    Mlist, Slist, frames_conversions, Glist = m_mr_notation

    double_pendulum = DoublePendulum()
    l0, l1, l2 = double_pendulum.l0, double_pendulum.l1, double_pendulum.l2

    T_ee = T(R=np.eye(3), p=colvec([0, 0, l2]))
    X_0_n = Ad(Tinv(T(R=np.eye(3), p=colvec([0, 0, l0 + l1]))))

    f_ext = {1: X_0_n.T @ Ad(Tinv(T_ee)).T @ -colvec(f_tip)}

    np.testing.assert_allclose(
        rnea(m_feath_notation, q, qd, qdd, g, f_ext=f_ext)[:, 0],
        InverseDynamics(q, qd, qdd, g, f_tip, Mlist, Glist, Slist)
    )


@pytest.mark.parametrize(
    "q,qd,qdd,g,Ftip",
    [([1, 0.2], [0., 0.2], [0.1, 0.], [0, 0, -9.81], np.array([1, 2, 3, 4, 5, 6]))]
)
def test_wrenches_calculation(double_pendulum_models, q, qd, qdd, g, Ftip):
    q, qd, qdd, g = np.array(q), np.array(qd), np.array(qdd), np.array(g)

    m_feath_notation, m_mr_notation = double_pendulum_models
    Mlist, Slist, frames_conversions, Glist = m_mr_notation

    F_mr = wrenches_in_modern_robotics_notation(q, qd, qdd, g, Ftip, Mlist, Glist, Slist)

    algorithm = RNEAImpl()
    algorithm.run(m_feath_notation, q, qd, qdd, g, f_tip=Ftip)

    F = algorithm.F

    for i in range(len(F_mr)):
        np.testing.assert_allclose(Ad(frames_conversions[i]).T @ F_mr[i], F[i][:, 0])


# It is Modern robotics inverse dynamics function modified to return wrenches instead of torques
# It is useful only for demonstration purposes
def wrenches_in_modern_robotics_notation(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
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
