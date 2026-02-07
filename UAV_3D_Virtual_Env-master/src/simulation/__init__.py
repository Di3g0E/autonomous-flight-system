# Simulation Module
"""
Panda3D simulation utilities: world setup, camera control, physics.
"""

# Quaternion utilities (no Panda3D dependency)
from src.simulation.quaternion_euler_utility import euler_quat, quat_euler, deriv_quat

# Optional Panda3D-dependent modules
try:
    from src.simulation.world_setup import world_setup, quad_setup
    from src.simulation.camera_control import camera_control
    PANDA3D_AVAILABLE = True
except ImportError:
    world_setup = None
    quad_setup = None
    camera_control = None
    PANDA3D_AVAILABLE = False

__all__ = ['euler_quat', 'quat_euler', 'deriv_quat',
           'world_setup', 'quad_setup', 'camera_control', 
           'PANDA3D_AVAILABLE']
