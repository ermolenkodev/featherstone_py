import pytest

from featherstone_py.example_models import DoublePendulum


@pytest.fixture
def double_pendulum_models():
    double_pendulum = DoublePendulum()
    model = double_pendulum.to_featherstone_notation()
    Mlist, Slist, frames_conversions, Glist = double_pendulum.to_modern_robotics_notation()
    return model, (Mlist, Slist, frames_conversions, Glist)
