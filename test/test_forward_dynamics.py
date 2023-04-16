import pytest

import numpy as np
from modern_robotics import ForwardDynamics

from featherstone_py.forward_dynamics import InverseDynamicsUsingRNEA, InverseDynamicsUsingCRBA


@pytest.mark.parametrize(
    "q,qd,g,tau,Ftip,algorithm",
    [
        ([0.2, 0.2], [0.0, 0.0], [0, 0, -9.81], [0.0, 0.0], np.zeros(6), 'rnea'),
        ([1, 0.2], [0.0, 0.2], [0, 0, -9.8], [0.0, 0.0], np.zeros(6), 'rnea'),
        ([1, 0.2], [0.0, 0.2], [0, 0, -9.8], [0.0, 0.0], np.array([1, 2, 3, 4, 5, 6]), 'rnea'),
        ([1, 0], [1.0, 0.2], [0, 0, -9.81], [0.0, 0.0], np.array([1, 1, 1, 1, 1, 1]), 'rnea'),
        ([0.2, 0.2], [0.0, 0.0], [0, 0, -9.81], [0.0, 0.0], np.zeros(6), 'crba'),
        ([1, 0.2], [0.0, 0.2], [0, 0, -9.8], [0.0, 0.0], np.zeros(6), 'crba'),
        ([1, 0.2], [0.0, 0.2], [0, 0, -9.8], [0.0, 0.0], np.array([1, 2, 3, 4, 5, 6]), 'crba'),
        ([1, 0], [1.0, 0.2], [0, 0, -9.81], [0.0, 0.0], np.array([1, 1, 1, 1, 1, 1]), 'crba')
    ]
)
def test_forward_dynamics(double_pendulum_models, q, qd, g, tau, Ftip, algorithm):
    q, qd = np.array(q), np.array(qd)
    g = np.array(g)

    m_feath_notation, m_mr_notation = double_pendulum_models
    Mlist, Slist, frames_conversions, Glist = m_mr_notation

    tau = np.array(tau)

    qdd_mr = ForwardDynamics(q, qd, tau, g, Ftip, Mlist, Glist, Slist)

    if algorithm == 'rnea':
        alg = InverseDynamicsUsingRNEA()
    elif algorithm == 'crba':
        alg = InverseDynamicsUsingCRBA()
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')

    qdd_feath = alg(m_feath_notation, q, qd, tau, f_tip=Ftip, gravity=g)[:, 0]

    np.testing.assert_allclose(qdd_mr, qdd_feath)
