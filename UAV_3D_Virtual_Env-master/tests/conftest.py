"""
Pytest configuration and fixtures for the quadrotor test suite.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def env():
    """Create a fresh quadrotor environment for testing."""
    from src.envs.quadrotor_env import quad
    
    environment = quad(t_step=0.01, n=100, direct_control=1, T=1)
    yield environment
    environment.close()


@pytest.fixture
def panda3d_env():
    """Create a Panda3D wrapper environment for testing."""
    from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
    
    environment = Panda3DQuadrotorEnv(
        panda3d_app=None,
        quad_model=None,
        render_node=None,
        t_step=0.01,
        n=100,
        enable_collisions=False
    )
    yield environment
    environment.close()


@pytest.fixture
def seed():
    """Provide a consistent seed for reproducible tests."""
    return 42
