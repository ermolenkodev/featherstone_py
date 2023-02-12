import pytest

import numpy as np
from modern_robotics import ForwardDynamics

from featherstone_py.forward_dynamics import InverseDynamicsUsingRNEA


@pytest.mark.parametrize(
    "q,qd,g,tau,Ftip",
    [
        ([0.2, 0.2], [0.0, 0.0], [0, 0, -9.81], [0.0, 0.0], np.zeros(6)),
        ([1, 0.2], [0.0, 0.2], [0, 0, -9.8], [0.0, 0.0], np.zeros(6)),
    ],
)
def test_forward_dynamics(double_pendulum_models, q, qd, g, tau, Ftip):
    q, qd = np.array(q), np.array(qd)
    g = np.array(g)

    m_feath_notation, m_mr_notation = double_pendulum_models
    Mlist, Slist, frames_conversions, Glist = m_mr_notation

    tau = np.array(tau)

    qdd_mr = ForwardDynamics(q, qd, tau, g, Ftip, Mlist, Glist, Slist)

    alg = InverseDynamicsUsingRNEA()
    qdd_feath = alg(m_feath_notation, q, qd, tau, f_ext=Ftip, gravity=g)[:, 0]

    np.testing.assert_allclose(qdd_mr, qdd_feath)
