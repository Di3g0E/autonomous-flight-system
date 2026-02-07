# UAV 3D Virtual Environment - Main Package
"""
Quadrotor simulation environment with Gymnasium interface and Panda3D visualization.
"""

__version__ = "1.0.0"
__author__ = "Rafael Costa Fernandes"

from src.envs.quadrotor_env import quad
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.envs.collision_detector import CollisionDetector, ObstacleManager
