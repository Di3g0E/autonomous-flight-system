"""
Panda3D Wrapper for Quadrotor Gymnasium Environment

This module provides a wrapper that integrates the pure physics-based quadrotor
environment with Panda3D's 3D visualization and collision detection capabilities.
"""

import numpy as np
import gymnasium as gym
from src.envs.quadrotor_env import quad
from src.envs.collision_detector import CollisionDetector, ObstacleManager


class Panda3DQuadrotorEnv(gym.Env):
    """
    Panda3D-integrated Quadrotor Environment.
    
    This wrapper adds 3D visualization and collision detection to the base
    quadrotor environment while maintaining the Gymnasium API.
    
    The environment can run in two modes:
    - With rendering: Full 3D visualization and collision detection
    - Headless: Pure physics simulation (faster for training)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        panda3d_app=None,
        quad_model=None,
        render_node=None,
        t_step=0.01,
        n=1000,
        euler=0,
        direct_control=1,
        T=1,
        render_mode=None,
        enable_collisions=True,
        collision_radius=0.3,
        collision_penalty=-100.0
    ):
        """
        Initialize the Panda3D Quadrotor Environment.
        
        Args:
            panda3d_app: Panda3D ShowBase application instance (optional)
            quad_model: Panda3D quadrotor model NodePath (optional)
            render_node: Panda3D render node (optional)
            t_step: Integration time step
            n: Maximum number of steps per episode
            euler: Use Euler angles (1) or quaternions (0)
            direct_control: Direct motor control (1) or indirect (0)
            T: Number of warm-up steps
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            enable_collisions: Enable collision detection
            collision_radius: Radius of collision sphere around quadrotor
            collision_penalty: Reward penalty for collisions
        """
        super(Panda3DQuadrotorEnv, self).__init__()
        
        # Store Panda3D components
        self.panda3d_app = panda3d_app
        self.quad_model = quad_model
        self.render_node = render_node
        self.render_mode = render_mode
        
        # Collision settings
        self.enable_collisions = enable_collisions and (panda3d_app is not None)
        self.collision_radius = collision_radius
        self.collision_penalty = collision_penalty
        
        # Create base physics environment
        self.base_env = quad(
            t_step=t_step,
            n=n,
            euler=euler,
            direct_control=direct_control,
            T=T,
            render_mode=None  # Base env doesn't handle rendering
        )
        
        # Copy spaces from base environment
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
        # Initialize collision system if enabled
        self.collision_detector = None
        self.obstacle_manager = None
        
        if self.enable_collisions:
            self._setup_collision_system()
        
        # Collision state
        self.collision_occurred = False
        self.collision_info = {}
    
    def _setup_collision_system(self):
        """Setup the collision detection system."""
        if self.quad_model is None or self.render_node is None:
            print("Warning: Cannot setup collision system without Panda3D components")
            self.enable_collisions = False
            return
        
        # Create collision detector
        self.collision_detector = CollisionDetector(
            self.render_node,
            self.quad_model,
            collision_radius=self.collision_radius
        )
        
        # Create obstacle manager
        self.obstacle_manager = ObstacleManager(self.render_node)
    
    def add_box_obstacle(self, position, size, name="box_obstacle"):
        """
        Add a box obstacle to the environment.
        
        Args:
            position: (x, y, z) position
            size: (width, depth, height) dimensions
            name: Obstacle identifier
        
        Returns:
            NodePath or None if collisions disabled
        """
        if self.obstacle_manager:
            return self.obstacle_manager.add_box_obstacle(position, size, name)
        return None
    
    def add_sphere_obstacle(self, position, radius, name="sphere_obstacle"):
        """
        Add a sphere obstacle to the environment.
        
        Args:
            position: (x, y, z) position
            radius: Sphere radius
            name: Obstacle identifier
        
        Returns:
            NodePath or None if collisions disabled
        """
        if self.obstacle_manager:
            return self.obstacle_manager.add_sphere_obstacle(position, radius, name)
        return None
    
    def add_model_collision(self, model_node, name="model_obstacle"):
        """
        Add collision detection to an existing 3D model.
        
        Args:
            model_node: Panda3D model NodePath
            name: Obstacle identifier
        
        Returns:
            NodePath or None if collisions disabled
        """
        if self.obstacle_manager:
            return self.obstacle_manager.add_model_collision(model_node, name)
        return None
    
    def clear_obstacles(self):
        """Remove all obstacles from the environment."""
        if self.obstacle_manager:
            self.obstacle_manager.clear_obstacles()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options (can include 'det_state' for deterministic state)
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Reset base environment
        observation, info = self.base_env.reset(seed=seed, options=options)
        
        # Reset collision state
        self.collision_occurred = False
        self.collision_info = {}
        
        if self.collision_detector:
            self.collision_detector.reset()
        
        # Update Panda3D visualization if available
        if self.quad_model is not None:
            self._update_visualization()
        
        # Add collision info to info dict
        info['collision'] = self.collision_info
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
        
        Returns:
            observation: New observation
            reward: Reward obtained
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Additional information
        """
        # Execute step in base environment
        observation, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Update Panda3D visualization if available
        if self.quad_model is not None:
            self._update_visualization()
        
        # Check for collisions
        self.collision_occurred = False
        self.collision_info = {}
        
        if self.enable_collisions and self.collision_detector:
            self.collision_occurred = self.collision_detector.check_collisions()
            
            if self.collision_occurred:
                # Get detailed collision information
                self.collision_info = self.collision_detector.get_collision_info()
                
                # Apply collision penalty to reward
                reward += self.collision_penalty
                
                # Terminate episode on collision
                terminated = True
        
        # Add collision info to info dict
        info['collision'] = self.collision_info
        info['collision_occurred'] = self.collision_occurred
        
        return observation, reward, terminated, truncated, info
    
    def _update_visualization(self):
        """Update the Panda3D 3D model based on current state."""
        if self.quad_model is None:
            return
        
        # Get current state from base environment
        state = self.base_env.state
        
        # Extract position (x, y, z)
        position = state[0:6:2]
        
        # Extract quaternion and convert to Euler angles for Panda3D
        # Note: Panda3D uses HPR (Heading, Pitch, Roll)
        ang = self.base_env.ang  # Already in Euler angles
        
        # Set position (adjust z for visualization offset if needed)
        self.quad_model.setPos(
            float(position[0]),
            float(position[1]),
            float(position[2]) + 5  # Offset to match original visualization
        )
        
        # Set orientation (convert from our convention to Panda3D's HPR)
        # Our convention: [roll, pitch, yaw] in radians
        # Panda3D: (heading, pitch, roll) in degrees
        ang_deg = (
            float(ang[2] * 180 / np.pi),  # yaw -> heading
            float(ang[0] * 180 / np.pi),  # roll -> pitch (note the swap)
            float(ang[1] * 180 / np.pi)   # pitch -> roll (note the swap)
        )
        self.quad_model.setHpr(*ang_deg)
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            None or rgb_array depending on render_mode
        """
        if self.render_mode == 'human':
            # Panda3D handles rendering automatically
            return None
        elif self.render_mode == 'rgb_array':
            # TODO: Capture frame from Panda3D camera
            # This would require setting up an offscreen buffer
            return None
        return None
    
    def close(self):
        """Clean up resources."""
        self.base_env.close()
        
        if self.obstacle_manager:
            self.obstacle_manager.clear_obstacles()
    
    def enable_collision_debug(self):
        """Enable visual debugging of collision shapes."""
        if self.collision_detector:
            self.collision_detector.enable_debug_visualization()
        if self.obstacle_manager:
            self.obstacle_manager.enable_debug_visualization()
    
    def disable_collision_debug(self):
        """Disable visual debugging of collision shapes."""
        if self.collision_detector:
            self.collision_detector.disable_debug_visualization()
        if self.obstacle_manager:
            self.obstacle_manager.disable_debug_visualization()
    
    def get_collision_info(self):
        """
        Get information about the last collision.
        
        Returns:
            dict: Collision information
        """
        return self.collision_info
    
    def set_collision_penalty(self, penalty):
        """
        Set the reward penalty for collisions.
        
        Args:
            penalty: Negative reward value for collisions
        """
        self.collision_penalty = penalty
