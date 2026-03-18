"""
Panda3D Wrapper for Quadrotor Gymnasium Environment

This module provides a wrapper that integrates the pure physics-based quadrotor
environment with Panda3D's 3D visualization and collision detection capabilities.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
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
        collision_penalty=-100.0,
        # Camera parameters
        use_camera=False,
        camera_high_freq_obj=None,
        camera_low_freq_obj=None,
        camera_high_freq_size=(32, 32),
        camera_low_freq_size=(32, 32),
        physics_steps_per_high_freq_capture=1,
        physics_steps_per_low_freq_capture=10,
        # Depth parameters
        use_depth=False,
        depth_metric=False,
        # Target/goal parameters
        use_target=False,
        target_mode='fixed',
        target_range=3.0,
        target_radius=0.25,
        target_speed=0.2,
        search_timeout_steps=1000,
        # Filming mode: drone navigates purely by vision (no geometric attraction)
        filming_mode=True
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
            use_camera: Enable camera observation system
            camera_high_freq_obj: opencv_camera object for high-frequency captures
            camera_low_freq_obj: opencv_camera object for low-frequency captures
            camera_high_freq_size: Target size (width, height) for high-freq camera
            camera_low_freq_size: Target size (width, height) for low-freq camera
            physics_steps_per_high_freq_capture: Physics steps between high-freq captures
            physics_steps_per_low_freq_capture: Physics steps between low-freq captures
            use_depth: Enable depth buffer extraction (requires use_camera=True)
            use_target: Enable visual target tracking mode
            target_mode: 'fixed' (random static), 'waypoints' (sequential), 'moving' (circular)
            target_range: Maximum distance for random target placement
            target_radius: Visual radius of the target sphere (0.25 ≈ drone size)
            target_speed: Speed of moving target (m/s, only for 'moving' mode)
            search_timeout_steps: Max steps without seeing target before truncation (1000=10s)
            filming_mode: If True, disable geometric attraction (base env target stays at origin).
                          All navigation comes from visual rewards only.
            depth_metric: Use metric depth (meters) instead of normalized [0,1]
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
        
        # Camera settings
        self.use_camera = use_camera
        self.camera_high_freq_obj = camera_high_freq_obj
        self.camera_low_freq_obj = camera_low_freq_obj
        self.camera_high_freq_size = camera_high_freq_size
        self.camera_low_freq_size = camera_low_freq_size
        self.physics_steps_per_high_freq_capture = physics_steps_per_high_freq_capture
        self.physics_steps_per_low_freq_capture = physics_steps_per_low_freq_capture
        
        # Depth settings
        self.use_depth = use_depth and use_camera  # Depth requires camera to be enabled
        self.depth_metric = depth_metric
        
        # Camera state
        self._step_counter = 0
        self._last_high_freq_image = None
        self._last_low_freq_image = None
        self._last_high_freq_depth = None
        self._last_low_freq_depth = None
        
        # Initialize placeholder images (black frames) and depth maps
        if self.use_camera:
            self._last_high_freq_image = np.zeros(
                (*camera_high_freq_size[::-1], 3), dtype=np.uint8
            )
            self._last_low_freq_image = np.zeros(
                (*camera_low_freq_size[::-1], 3), dtype=np.uint8
            )
        
        if self.use_depth:
            self._last_high_freq_depth = np.zeros(
                (*camera_high_freq_size[::-1], 1), dtype=np.float32
            )
            self._last_low_freq_depth = np.zeros(
                (*camera_low_freq_size[::-1], 1), dtype=np.float32
            )
        
        # Create base physics environment
        self.base_env = quad(
            t_step=t_step,
            n=n,
            euler=euler,
            direct_control=direct_control,
            T=T,
            render_mode=None  # Base env doesn't handle rendering
        )
        
        # Copy action space from base environment
        self.action_space = self.base_env.action_space
        
        # Configure observation space based on camera and depth usage
        if self.use_camera:
            obs_dict = {
                "state": self.base_env.observation_space,
                "camera_high_freq": spaces.Box(
                    low=0, high=255,
                    shape=(*camera_high_freq_size[::-1], 3),
                    dtype=np.uint8
                ),
                "camera_low_freq": spaces.Box(
                    low=0, high=255,
                    shape=(*camera_low_freq_size[::-1], 3),
                    dtype=np.uint8
                )
            }
            
            # Add depth to observation space if enabled
            if self.use_depth:
                obs_dict["depth_high_freq"] = spaces.Box(
                    low=0, high=np.inf,
                    shape=(*camera_high_freq_size[::-1], 1),
                    dtype=np.float32
                )
                obs_dict["depth_low_freq"] = spaces.Box(
                    low=0, high=np.inf,
                    shape=(*camera_low_freq_size[::-1], 1),
                    dtype=np.float32
                )
            
            self.observation_space = spaces.Dict(obs_dict)
        else:
            # Backward compatibility: simple Box observation
            self.observation_space = self.base_env.observation_space
        
        # Initialize collision system if enabled
        self.collision_detector = None
        self.obstacle_manager = None
        
        if self.enable_collisions:
            self._setup_collision_system()
        
        # Collision state
        self.collision_occurred = False
        self.collision_info = {}
        
        # Filming mode
        self.filming_mode = filming_mode

        # Target/goal tracking
        self.use_target = use_target
        self.target_mode = target_mode
        self.target_range = target_range
        self.target_radius = target_radius
        self.target_speed = target_speed
        self.target_pos = np.zeros(3)
        self._target_node = None
        self._target_time = 0.0
        self._waypoint_idx = 0
        self._waypoints = []
        self._arrival_threshold = 0.3  # meters
        self.search_timeout_steps = search_timeout_steps
        self._target_ever_seen = False  # has the target been seen at least once?
        self._bird_camera = None  # set externally for recording
        
        # Create target marker if in target mode and Panda3D is available
        if self.use_target and self.panda3d_app is not None:
            self._create_target_marker()
    
    def _create_target_marker(self):
        """Create a visible sphere in the Panda3D scene as the target marker."""
        from panda3d.core import (
            GeomNode, CardMaker, Material, LColor,
            PointLight, NodePath
        )
        
        # Create a sphere using Panda3D's built-in geometry
        # We use a simple colored point light with a visible model
        try:
            # Try loading a sphere model
            sphere = self.panda3d_app.loader.loadModel("models/misc/sphere")
        except Exception:
            # Fallback: create a simple visual using CardMaker
            sphere = None
        
        if sphere is not None:
            self._target_node = sphere
            self._target_node.reparentTo(self.render_node)
            self._target_node.setScale(self.target_radius * 2)
            
            # Bright red/orange material so it's clearly visible
            mat = Material()
            mat.setEmission(LColor(1.0, 0.3, 0.0, 1.0))  # Bright orange glow
            mat.setDiffuse(LColor(1.0, 0.2, 0.0, 1.0))
            mat.setAmbient(LColor(1.0, 0.4, 0.0, 1.0))
            self._target_node.setMaterial(mat)
            self._target_node.setColor(1.0, 0.3, 0.0, 1.0)
            
            # Add a point light at the target for extra visibility
            plight = PointLight('target_light')
            plight.setColor(LColor(1.0, 0.5, 0.0, 1.0))
            plight.setAttenuation((1, 0.05, 0.01))
            plight_node = self._target_node.attachNewNode(plight)
            self.render_node.setLight(plight_node)
        else:
            # Headless fallback: no visual marker
            self._target_node = None
    
    def _randomize_target(self):
        """Generate a random target at the SAME HEIGHT as the drone, at a random angle.
        
        The target can be behind, beside, or in front of the drone (0-2π).
        It is clamped to the central zone (±3m) to stay in clean space.
        """
        drone_pos = self.base_env.state[0:5:2]  # [x, y, z]
        
        # Random angle in full circle and random distance
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(1.0, self.target_range)
        
        self.target_pos = np.array([
            drone_pos[0] + distance * np.cos(angle),
            drone_pos[1] + distance * np.sin(angle),
            drone_pos[2],  # SAME HEIGHT as drone
        ])
        
        # Clamp to central zone (stay away from city edges)
        self.target_pos = np.clip(self.target_pos, -3.0, 3.0)
        
        # For waypoints mode, generate a sequence at same height
        if self.target_mode == 'waypoints':
            n_waypoints = 5
            self._waypoints = []
            for _ in range(n_waypoints):
                a = np.random.uniform(0, 2 * np.pi)
                d = np.random.uniform(1.0, self.target_range)
                wp = np.array([
                    d * np.cos(a),
                    d * np.sin(a),
                    drone_pos[2],
                ])
                self._waypoints.append(np.clip(wp, -3.0, 3.0))
            self._waypoint_idx = 0
            self.target_pos = self._waypoints[0]
        
        # In filming mode, base env target stays at origin (navigation is purely visual).
        # In reach mode, base env shaping attracts toward target geometrically.
        if not self.filming_mode:
            self.base_env.set_target(self.target_pos)
        
        # Update visual marker position
        self._update_target_marker_pos()
    
    def _update_target_marker_pos(self):
        """Move the 3D target marker to the current target position."""
        if self._target_node is not None:
            self._target_node.setPos(
                float(self.target_pos[0]),
                float(self.target_pos[1]),
                float(self.target_pos[2]) + 5  # Same z-offset as drone visualization
            )
    
    def _update_target(self, dt=0.01):
        """
        Update target position based on mode.
        
        Args:
            dt: Time step for moving target
        """
        if not self.use_target:
            return
        
        drone_pos = self.base_env.state[0:5:2]
        dist_to_target = np.linalg.norm(drone_pos - self.target_pos)
        
        if self.target_mode == 'fixed':
            # Target doesn't move
            pass
        
        elif self.target_mode == 'waypoints':
            # Advance to next waypoint when close enough
            if dist_to_target < self._arrival_threshold and len(self._waypoints) > 0:
                self._waypoint_idx = (self._waypoint_idx + 1) % len(self._waypoints)
                self.target_pos = self._waypoints[self._waypoint_idx]
                if not self.filming_mode:
                    self.base_env.set_target(self.target_pos)
                self._update_target_marker_pos()
        
        elif self.target_mode == 'moving':
            # Random movement using Ornstein-Uhlenbeck process
            self._target_time += dt
            
            # Reversion to center and random noise
            theta = 0.15    # strength of reversion
            sigma = 0.3     # intensity of noise
            
            # Simple 2D OU process for x, y
            vel = theta * (np.zeros(2) - self.target_pos[:2]) * dt + \
                  sigma * np.random.randn(2) * np.sqrt(dt)
            
            self.target_pos[:2] += vel * self.target_speed * 10
            self.target_pos[2] = drone_pos[2]  # Keep at drone height
            
            self.target_pos = np.clip(self.target_pos, -3.0, 3.0)
            if not self.filming_mode:
                self.base_env.set_target(self.target_pos)
            self._update_target_marker_pos()
    
    def _goal_reward(self):
        """
        Compute goal-reaching reward.
        
        Returns:
            dict with target info; adds +500 arrival bonus.
        """
        drone_pos = self.base_env.state[0:5:2]
        dist = np.linalg.norm(drone_pos - self.target_pos)
        arrived = dist < self._arrival_threshold
        
        return {
            'target_pos': self.target_pos.copy(),
            'distance_to_target': float(dist),
            'arrived': arrived,
        }
    
    def _compute_visual_tracking_reward(self):
        """
        Compute dense per-step reward based on visual target tracking.
        
        Detects the orange sphere in the 32×32 camera image by color,
        then rewards:
          - Centering: how close the sphere centroid is to image center
          - Scale control: keeping the sphere at a consistent pixel size
          - Penalty for not seeing the target (encourages yaw search)
        
        Returns:
            reward (float), info (dict)
        """
        reward = 0.0
        info = {}
        
        img = self._last_high_freq_image  # (H, W, 3) uint8 RGB
        if img is None:
            return reward, info
        
        h, w = img.shape[:2]
        
        # Detect orange pixels via HSV thresholding
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Orange hue range in HSV (OpenCV H: 0-180)
        mask = cv2.inRange(hsv, (5, 100, 100), (25, 255, 255))
        
        pixel_count = int(np.sum(mask > 0))
        total_pixels = h * w
        target_visible = pixel_count > 2  # at least 3 orange pixels
        
        info['target_visible'] = target_visible
        info['target_pixels'] = pixel_count
        
        if target_visible:
            self._target_ever_seen = True
            
            # Compute centroid of detected pixels
            ys, xs = np.where(mask > 0)
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            center_x, center_y = w / 2.0, h / 2.0
            
            # Normalized distance from centroid to image center (0=perfect, 1=corner)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2) / max_dist
            
            # CENTERING BONUS: +3.0 when perfectly centered, → 0 at edge
            centering_reward = 3.0 * (1.0 - dist_to_center)
            reward += centering_reward
            
            # SCALE CONTROL: +3.0 at ideal 8%, proportionally negative when far/close
            target_fraction = pixel_count / total_pixels
            ideal_fraction = 0.08  # ~8% of image
            scale_error = abs(target_fraction - ideal_fraction) / max(ideal_fraction, 1e-6)
            scale_reward = 3.0 * (1.0 - scale_error)  # Proportional: positive at ideal, negative far away
            scale_reward = max(scale_reward, -3.0)     # Floor to prevent extreme penalties
            reward += scale_reward
            
            # PROXIMITY PENALTY: penalize getting TOO close (> 20% of image)
            if target_fraction > 0.20:
                reward -= 3.0 * (target_fraction - 0.20) / 0.20
            
            info['centering_reward'] = float(centering_reward)
            info['scale_reward'] = float(scale_reward)
            info['target_center'] = (cx, cy)
            info['target_fraction'] = float(target_fraction)
        else:
            # Penalty for not seeing the target (encourages rotation to search)
            reward -= 0.5
        
        return reward, info
    
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
    
    def _capture_camera_images(self, force_capture=False):
        """
        Capture camera images and depth buffers according to configured frequencies.
        
        Args:
            force_capture: If True, capture from both cameras regardless of step counter
        
        Returns:
            bool: True if at least one image was captured
        """
        captured_any = False
        
        if not self.use_camera:
            return captured_any
        
        # High-frequency camera capture
        if (force_capture or 
            self._step_counter % self.physics_steps_per_high_freq_capture == 0):
            if self.camera_high_freq_obj is not None:
                # Capture RGB image
                success, img = self.camera_high_freq_obj.get_image()
                if success and img is not None:
                    # Convert RGBA to RGB (remove alpha channel)
                    img_rgb = img[:, :, :3]
                    
                    # Resize to target size
                    # Note: cv2.resize expects (width, height)
                    img_resized = cv2.resize(
                        img_rgb, 
                        self.camera_high_freq_size,
                        interpolation=cv2.INTER_AREA
                    )
                    
                    # AXIS ALIGNMENT CHECK: Uncomment if images appear vertically flipped
                    # This is a common issue when converting OpenGL textures to NumPy arrays
                    # img_resized = cv2.flip(img_resized, 0)
                    
                    # Ensure uint8 dtype (raw 0-255 values, no normalization)
                    self._last_high_freq_image = img_resized.astype(np.uint8)
                    captured_any = True
                
                # Capture depth if enabled
                if self.use_depth:
                    success_depth, depth = self.camera_high_freq_obj.get_depth(
                        normalize=not self.depth_metric,
                        metric=self.depth_metric
                    )
                    if success_depth and depth is not None:
                        # Depth is already resized and in (H, W, 1) format from get_depth()
                        self._last_high_freq_depth = depth.astype(np.float32)
        
        # Low-frequency camera capture
        if (force_capture or 
            self._step_counter % self.physics_steps_per_low_freq_capture == 0):
            if self.camera_low_freq_obj is not None:
                # Capture RGB image
                success, img = self.camera_low_freq_obj.get_image()
                if success and img is not None:
                    # Convert RGBA to RGB
                    img_rgb = img[:, :, :3]
                    
                    # Resize to target size
                    img_resized = cv2.resize(
                        img_rgb,
                        self.camera_low_freq_size,
                        interpolation=cv2.INTER_AREA
                    )
                    
                    # AXIS ALIGNMENT CHECK: Uncomment if needed
                    # img_resized = cv2.flip(img_resized, 0)
                    
                    # Ensure uint8 dtype
                    self._last_low_freq_image = img_resized.astype(np.uint8)
                    captured_any = True
                
                # Capture depth if enabled
                if self.use_depth:
                    success_depth, depth = self.camera_low_freq_obj.get_depth(
                        normalize=not self.depth_metric,
                        metric=self.depth_metric
                    )
                    if success_depth and depth is not None:
                        # Depth is already resized and in (H, W, 1) format from get_depth()
                        self._last_low_freq_depth = depth.astype(np.float32)
        
        return captured_any

    
    def _build_observation(self, state):
        """
        Build observation according to configuration.
        
        Args:
            state: Physics state from base environment (np.array)
        
        Returns:
            observation: Either state alone (if use_camera=False) or Dict with state + images + depth
        """
        if not self.use_camera:
            # Backward compatibility mode
            return state
        
        # Build Dict observation with state and camera images
        obs = {
            "state": state,
            "camera_high_freq": self._last_high_freq_image.copy(),
            "camera_low_freq": self._last_low_freq_image.copy()
        }
        
        # Add depth if enabled
        if self.use_depth:
            obs["depth_high_freq"] = self._last_high_freq_depth.copy()
            obs["depth_low_freq"] = self._last_low_freq_depth.copy()
        
        return obs

    
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options (can include 'det_state' for deterministic state)
        
        Returns:
            observation: Initial observation (state or Dict with state + images)
            info: Additional information
        """
        # Reset base environment
        observation, info = self.base_env.reset(seed=seed, options=options)
        
        # Reset camera step counter
        self._step_counter = 0
        
        # Reset collision state
        self.collision_occurred = False
        self.collision_info = {}
        
        if self.collision_detector:
            self.collision_detector.reset()
        
        # Capture initial camera images (forced)
        if self.use_camera:
            self._capture_camera_images(force_capture=True)
        
        # Update Panda3D visualization if available
        if self.quad_model is not None:
            self._update_visualization()
        
        # Add collision info to info dict
        info['collision'] = self.collision_info
        
        # Randomize target if in goal mode
        if self.use_target:
            self._target_ever_seen = False
            self._randomize_target()
            self._target_time = 0.0
            info['target'] = self._goal_reward()
        
        # Build and return observation (with or without camera)
        observation = self._build_observation(observation)
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
        
        Returns:
            observation: New observation (state or Dict with state + images)
            reward: Reward obtained
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Additional information
        """
        # Execute step in base environment (physics simulation)
        observation, reward, terminated, truncated, info = self.base_env.step(action)

        # Neutralize +500 "solution achieved" bonus in filming mode
        # (base env rewards hovering at origin, irrelevant for visual tracking)
        if self.filming_mode and self.base_env.solved:
            reward -= 500
            self.base_env.solved = 0

        # Increment step counter for camera frame skip
        self._step_counter += 1
        
        # Update Panda3D visualization FIRST (move drone to new position)
        if self.quad_model is not None:
            self._update_visualization()
        
        # Force Panda3D to render the frame so camera buffers are updated
        if self.panda3d_app is not None:
            self.panda3d_app.graphicsEngine.renderFrame()
        
        # NOW capture camera images (after render, so they reflect current state)
        if self.use_camera:
            self._capture_camera_images()
        
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
        
        # Update target tracking
        if self.use_target:
            self._update_target(dt=self.base_env.t_step)
            goal_info = self._goal_reward()
            info['target'] = goal_info
            
            # Reward is now purely based on tracking quality
            # No arrival bonus to avoid collision behavior
        
        # Visual tracking reward (dense, per-step)
        if self.use_target and self.use_camera:
            visual_reward, visual_info = self._compute_visual_tracking_reward()
            reward += visual_reward
            info['visual_tracking'] = visual_info
        
        # Search timeout: truncate if target was never seen after 7s
        if (self.use_target and 
                self._step_counter >= self.search_timeout_steps and
                not self._target_ever_seen):
            truncated = True
        
        # Build observation (with or without camera)
        observation = self._build_observation(observation)
        
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
