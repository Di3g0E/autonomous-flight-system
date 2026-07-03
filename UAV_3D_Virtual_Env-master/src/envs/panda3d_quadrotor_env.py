"""
Panda3D Wrapper for Quadrotor Gymnasium Environment

This module provides a wrapper that integrates the pure physics-based quadrotor
environment with Panda3D's 3D visualization and collision detection capabilities.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from src.envs.quadrotor_env import quad
from src.envs.collision_detector import CollisionDetector, ObstacleManager
from src.simulation.quaternion_euler_utility import euler_quat


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
        filming_mode=True,
        # Lemniscate trajectory scale (half-width of the ∞; full width = 2×scale)
        lemniscate_scale=2.5,
        # Visual reward parameters (fraction-based)
        ideal_fraction=0.25,
        fraction_tolerance=0.05,
        max_visual_reward=1000.0,
        min_start_distance=3.0,
        # ── v2 reward & init parameters ──
        use_new_reward=False,
        initial_target_distance=2.0,
        constrained_init=False,
        init_pos_range=0.5,
        init_vel_range=0.25,
        init_ang_range=0.1,
        init_ang_vel_range=None,  # if None, defaults to init_ang_range (backward compat)
        # ── v3 centroid-obs mode ──
        centroid_obs=False,
        hover_height=1.394,
        camera_down=False,
        exclude_low_freq_camera=False,
        # ── v3+ reward selection ──
        reward_version='auto',  # 'auto', 'v2', 'v3'
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

        # Early v3 flags (needed before obs space config below)
        self.centroid_obs = centroid_obs
        self.exclude_low_freq_camera = exclude_low_freq_camera
        
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
        if self.use_camera and not self.centroid_obs:
            obs_dict = {
                "state": self.base_env.observation_space,
                "camera_high_freq": spaces.Box(
                    low=0, high=255,
                    shape=(*camera_high_freq_size[::-1], 3),
                    dtype=np.uint8
                ),
            }

            if not self.exclude_low_freq_camera:
                obs_dict["camera_low_freq"] = spaces.Box(
                    low=0, high=255,
                    shape=(*camera_low_freq_size[::-1], 3),
                    dtype=np.uint8
                )

            # Add depth to observation space if enabled
            if self.use_depth:
                obs_dict["depth_high_freq"] = spaces.Box(
                    low=0, high=np.inf,
                    shape=(*camera_high_freq_size[::-1], 1),
                    dtype=np.float32
                )
                if not self.exclude_low_freq_camera:
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

        # Lemniscate (∞) trajectory parameters
        self.lemniscate_scale = lemniscate_scale  # half-width 'a' (full width = 2a)
        self._lemniscate_phase = 0.0              # starting phase on the curve

        # Visual reward parameters (fraction-based)
        self.ideal_fraction = ideal_fraction
        self.fraction_tolerance = fraction_tolerance
        self.max_visual_reward = max_visual_reward
        self.min_start_distance = min_start_distance

        # ── v2 reward & init ──
        self.use_new_reward = use_new_reward
        self.initial_target_distance = initial_target_distance
        self.constrained_init = constrained_init
        self.init_pos_range = init_pos_range
        self.init_vel_range = init_vel_range
        self.init_ang_range = init_ang_range
        # Decouple angular-velocity init from angular-position init so we
        # can train recovery from high yaw rate without forcing huge
        # initial tilts. Defaults to init_ang_range if not specified.
        self.init_ang_vel_range = (init_ang_vel_range
                                   if init_ang_vel_range is not None
                                   else init_ang_range)
        self._target_visible_last_step = False  # for discovery bonus
        self._prev_centering_dist = 0.0  # for centering velocity reward

        # ── v3+ stabilization-only mode (set externally by curriculum) ──
        self.stabilization_only = False   # when True, reward ignores target
        self.reward_version = reward_version  # 'auto', 'v2', 'v3', 'v3.1'

        # ── v3.1 action smoothness tracking ──
        self._prev_action = None  # for jerk penalty in v3.1 reward

        # ── v3 centroid-obs mode ──
        self.hover_height = hover_height
        self.camera_down = camera_down
        self._prev_centroid = np.zeros(2, dtype=np.float32)
        self.store_transitions = True  # spiral controller sets False

        if self.centroid_obs:
            # 19-D flat: 13 state + cx, cy, frac, vis, dcx, dcy
            low = np.array(
                [-5, -10, -5, -10, -5, -10, -1, -1, -1, -1, -10, -10, -10,
                 -1, -1, 0, 0, -2, -2], dtype=np.float32)
            high = np.array(
                [5, 10, 5, 10, 5, 10, 1, 1, 1, 1, 10, 10, 10,
                 1, 1, 1, 1, 2, 2], dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high,
                                                dtype=np.float32)

        # Create target marker if in target mode and Panda3D is available
        if self.use_target and self.panda3d_app is not None:
            self._create_target_marker()
    
    def _create_target_marker(self):
        """Create a visible sphere in the Panda3D scene as the target marker."""
        from panda3d.core import Material, LColor
        
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
            self._target_node.setScale(self.target_radius)

            # Magenta self-illuminating material (H≈150 in HSV).
            # Magenta does not exist in any scene texture (brick, wood,
            # concrete, asphalt, dirt, sky are all warm browns/greys/blue),
            # guaranteeing zero false positives in the HSV detector.
            # setLightOff() keeps the colour pure regardless of lighting.
            mat = Material()
            mat.setEmission(LColor(1.0, 0.0, 1.0, 1.0))
            mat.setDiffuse(LColor(0.0, 0.0, 0.0, 1.0))
            mat.setAmbient(LColor(0.0, 0.0, 0.0, 1.0))
            self._target_node.setMaterial(mat)
            self._target_node.setColor(1.0, 0.0, 1.0, 1.0)
            self._target_node.setLightOff()
        else:
            # Headless fallback: no visual marker
            self._target_node = None
    
    def _lemniscate_point(self, t):
        """Compute a point on the Bernoulli lemniscate (∞) in the horizontal plane.

        Parametric form:
            x(t) = a * cos(t) / (1 + sin²(t))
            y(t) = a * sin(t) * cos(t) / (1 + sin²(t))

        The ∞ lies flat along the X axis (width = 2a, depth ≈ 2a/3).
        Natural symmetry forces equal left/right turns, preventing
        directional bias during training.

        Args:
            t: Parameter along the curve (radians).

        Returns:
            (x, y) position on the lemniscate.
        """
        a = self.lemniscate_scale
        denom = 1.0 + np.sin(t) ** 2
        x = a * np.cos(t) / denom
        y = a * np.sin(t) * np.cos(t) / denom
        return x, y

    def _randomize_target(self):
        """Place the target for the new episode.

        - fixed:     random static position near the drone.
        - waypoints: sequence of random static waypoints.
        - moving:    horizontal lemniscate (∞) trajectory.  Only the starting
                     phase is randomized per episode; the shape is controlled
                     by ``lemniscate_scale`` (constructor parameter).
        """
        drone_pos = self.base_env.state[0:5:2]  # [x, y, z]

        if self.target_mode == 'moving':
            # Sample starting phase ensuring min_start_distance from drone
            for _ in range(200):
                self._lemniscate_phase = np.random.uniform(0, 2 * np.pi)
                x, y = self._lemniscate_point(self._lemniscate_phase)
                dist = np.sqrt((x - drone_pos[0])**2 + (y - drone_pos[1])**2)
                if dist >= self.min_start_distance:
                    break

            # Height: Fixed at ground level (z=0) for consistency.
            # Repositioning of the drone relative to the target is handled in reset().
            target_z = 0.0
            self.target_pos = np.array([x, y, target_z])

        elif self.camera_down:
            # Target directly below drone at ground level (z=0)
            self.target_pos = np.array([
                drone_pos[0],
                drone_pos[1],
                0.0,
            ])
            self.target_pos = np.clip(self.target_pos, -3.0, 3.0)

        else:
            # Random angle in full circle; fixed or random distance
            angle = np.random.uniform(0, 2 * np.pi)
            if self.use_new_reward:
                distance = self.initial_target_distance
            else:
                distance = np.random.uniform(1.0, self.target_range)

            self.target_pos = np.array([
                drone_pos[0] + distance * np.cos(angle),
                drone_pos[1] + distance * np.sin(angle),
                drone_pos[2],  # SAME HEIGHT as drone
            ])
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
            # Lemniscate (∞) trajectory along the horizontal plane.
            # target_speed acts as angular-speed multiplier (curriculum-compatible).
            # At speed=0.3 → ω≈0.6 rad/s → full loop ≈1050 steps (10.5 s)
            self._target_time += dt
            t = self._lemniscate_phase + self._target_time * self.target_speed * 2.0

            x, y = self._lemniscate_point(t)
            self.target_pos[0] = x
            self.target_pos[1] = y
            # Z stays at the constant altitude set during reset
            # (no longer follows drone height dynamically)

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

        Detects the magenta sphere in the 32×32 camera image by colour,
        then computes a reward based on the fraction of the image that
        the target occupies:

          - Within ±fraction_tolerance of ideal_fraction:
            Positive exponential reward, max = max_visual_reward at ideal.
          - Outside that band:
            Negative exponential penalty that grows with the error.
          - Target not visible:
            Small fixed penalty to encourage searching.

        Returns:
            reward (float), info (dict)
        """
        import math

        reward = 0.0
        info = {}

        img = self._last_high_freq_image  # (H, W, 3) uint8
        if img is None:
            return reward, info

        h, w = img.shape[:2]

        # Detect magenta pixels via HSV thresholding
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))

        pixel_count = int(np.sum(mask > 0))
        total_pixels = h * w
        target_visible = pixel_count > 2

        info['target_visible'] = target_visible
        info['target_pixels'] = pixel_count

        if target_visible:
            self._target_ever_seen = True

            target_fraction = pixel_count / total_pixels
            error = abs(target_fraction - self.ideal_fraction)

            if error <= self.fraction_tolerance:
                # ── Positive reward: Gaussian peak at ideal_fraction ──
                # max_visual_reward at error=0, ≈1% of max at boundary
                normalized = error / self.fraction_tolerance  # 0→1
                reward = self.max_visual_reward * math.exp(-5.0 * normalized ** 2)
            else:
                # ── Negative reward: exponential penalty beyond tolerance ──
                excess = (error - self.fraction_tolerance) / self.fraction_tolerance
                reward = -self.max_visual_reward * 0.1 * (math.exp(excess) - 1)
                reward = max(reward, -self.max_visual_reward)  # floor

            # Centroid (for logging / debug overlays)
            ys, xs = np.where(mask > 0)
            cx, cy = float(np.mean(xs)), float(np.mean(ys))

            info['target_fraction'] = float(target_fraction)
            info['fraction_error'] = float(error)
            info['target_center'] = (cx, cy)
            info['scale_reward'] = float(reward)
        else:
            # Small penalty to encourage rotating to find the target
            reward = -5.0

        return reward, info

    # ================================================================= #
    #  v2 multi-component reward                                        #
    # ================================================================= #

    def _compute_new_reward(self):
        """Multi-component dense reward for visual target tracking.

        Components
        ----------
        R_survival   +0.05       always (stay alive incentive)
        R_stability  0 … +1.0   low angular velocity × low tilt
        R_centering  0 … +3.0   target centroid close to image centre
        R_scale      0 … +2.0   target fraction close to ideal (asymmetric σ)
        R_discovery  +3.0       each time target reappears after being lost
        R_not_visible -0.5      per step without seeing the target

        Returns
        -------
        total_reward : float
        info : dict   (component breakdown for logging)
        """
        info = {}

        # ── State-based components (always active) ────────────────────
        state = self.base_env.state
        ang = self.base_env.ang  # [roll, pitch, yaw]

        # R_survival
        r_survival = 0.05

        # R_stability — two separate Gaussians (don't mix units)
        ang_vel_norm = np.linalg.norm(state[10:13]) / 10.0   # normalised by BB_VEL
        tilt_norm = (abs(ang[0]) + abs(ang[1])) / (np.pi / 2) # normalised by BB_ANG
        r_stability = math.exp(-3.0 * ang_vel_norm ** 2) * math.exp(-3.0 * tilt_norm ** 2)

        # ── Visual detection ─────────────────────────────────────────
        img = self._last_high_freq_image
        r_centering = 0.0
        r_scale = 0.0
        r_discovery = 0.0
        r_not_visible = 0.0
        target_visible = False

        if img is not None:
            h, w = img.shape[:2]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))

            pixel_count = int(np.sum(mask > 0))
            total_pixels = h * w
            target_visible = pixel_count > 2

            info['target_visible'] = target_visible
            info['target_pixels'] = pixel_count

            if target_visible:
                self._target_ever_seen = True

                # ── Discovery bonus (repeatable) ──
                if not self._target_visible_last_step:
                    r_discovery = 3.0

                # ── Centroid & centering reward ──
                ys, xs = np.where(mask > 0)
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                cx_norm = (cx - w / 2) / (w / 2)   # [-1, 1]
                cy_norm = (cy - h / 2) / (h / 2)   # [-1, 1]
                dist_from_center = math.sqrt(cx_norm ** 2 + cy_norm ** 2)  # [0, √2]
                r_centering = 3.0 * math.exp(-3.0 * dist_from_center ** 2)

                # ── Scale reward (asymmetric Gaussian + proximity penalty) ──
                # Gaussian peak at ideal_fraction (0.25 by default).
                # σ_far  = 0.12  → gentle slope when too far (fraction < ideal)
                # σ_near = 0.06  → steep slope when too close (fraction > ideal)
                # This encourages keeping a safe distance: approaching too
                # close is punished ~4× faster than drifting too far.
                #
                # Additionally, fractions above 0.40 receive a linear penalty
                # (-2.0 per 0.10 excess) to actively discourage collision-risk
                # proximity.  The penalty is clamped at -2.0.
                target_fraction = pixel_count / total_pixels
                fraction_error = target_fraction - self.ideal_fraction
                abs_error = abs(fraction_error)
                sigma = 0.12 if fraction_error <= 0 else 0.06
                r_scale = 2.0 * math.exp(-0.5 * (abs_error / sigma) ** 2)

                # Proximity penalty: linear ramp beyond 0.40
                proximity_threshold = 0.40
                if target_fraction > proximity_threshold:
                    excess = target_fraction - proximity_threshold
                    r_scale -= min(2.0, 20.0 * excess)  # -2.0 per 0.10 excess

                info['target_fraction'] = float(target_fraction)
                info['fraction_error'] = float(fraction_error)
                info['target_center'] = (cx, cy)
                info['centering_dist'] = float(dist_from_center)
            else:
                r_not_visible = -0.5

        # Update visibility flag for next step's discovery check
        self._target_visible_last_step = target_visible

        # ── Total ─────────────────────────────────────────────────────
        total = r_survival + r_stability + r_centering + r_scale + r_discovery + r_not_visible

        info['r_survival'] = r_survival
        info['r_stability'] = float(r_stability)
        info['r_centering'] = float(r_centering)
        info['r_scale'] = float(r_scale)
        info['r_discovery'] = float(r_discovery)
        info['r_not_visible'] = float(r_not_visible)
        info['scale_reward'] = float(r_scale)             # compat with recorder
        info['centering_reward'] = float(r_centering)     # compat with recorder

        return float(total), info

    # ================================================================= #
    #  v3 hover-tracking reward (3 components, clean)                   #
    # ================================================================= #

    def _compute_hover_reward(self):
        """Simplified 3-component reward for hover-tracking with centroid obs.

        Components (when target visible)
        ---------------------------------
        R_stability   0 → +1.0   low angular velocity × low tilt
        R_centering   0 → +2.0   target centroid close to image centre
        R_scale       0 → +1.0   target fraction near ideal (0.25)

        When target NOT visible
        -----------------------
        R_stability   0 → +1.0   (always active)
        R_invisible   −1.0       fixed penalty per step

        Total range: −1.0 (invisible, unstable) to +4.0 (perfect tracking)

        Returns
        -------
        total_reward : float
        info : dict
        """
        info = {}
        state = self.base_env.state
        ang = self.base_env.ang  # [roll, pitch, yaw]

        # R_stability — always active
        ang_vel_norm = np.linalg.norm(state[10:13]) / 10.0
        tilt_norm = (abs(ang[0]) + abs(ang[1])) / (np.pi / 2)
        r_stability = math.exp(-3.0 * ang_vel_norm ** 2) * math.exp(-3.0 * tilt_norm ** 2)

        # Detect target (reuse centroid already computed for observation)
        cx, cy, frac, vis = self._detect_target_in_image()

        r_centering = 0.0
        r_scale = 0.0
        r_invisible = 0.0
        target_visible = vis > 0

        if target_visible:
            self._target_ever_seen = True

            # R_centering — Gaussian on distance from image centre
            dist_from_center = math.sqrt(cx ** 2 + cy ** 2)  # [0, √2]
            r_centering = 2.0 * math.exp(-3.0 * dist_from_center ** 2)

            # R_scale — asymmetric Gaussian (tight above ideal, loose below)
            fraction_error = abs(frac - self.ideal_fraction)
            sigma = 0.12 if frac <= self.ideal_fraction else 0.06
            r_scale = 1.0 * math.exp(-0.5 * (fraction_error / sigma) ** 2)

            info['target_fraction'] = float(frac)
            info['fraction_error'] = float(fraction_error)
            info['target_center'] = (cx, cy)
            info['centering_dist'] = float(dist_from_center)
        else:
            r_invisible = -1.0

        info['target_visible'] = target_visible
        info['r_stability'] = float(r_stability)
        info['r_centering'] = float(r_centering)
        info['r_scale'] = float(r_scale)
        info['r_invisible'] = float(r_invisible)
        info['scale_reward'] = float(r_scale)
        info['centering_reward'] = float(r_centering)

        total = r_stability + r_centering + r_scale + r_invisible
        return float(total), info

    # ================================================================= #
    #  v3+ hover-tracking reward (stabilization-aware + tighter center)  #
    # ================================================================= #

    def _compute_v3_reward(self):
        """Enhanced reward for hover-tracking v3 with stabilization-only mode.

        When ``self.stabilization_only`` is True (Phase 0 of curriculum),
        the target is ignored and the agent is rewarded purely for
        cancelling velocity and tilt — learning a motor "base skill"
        before attempting visual tracking.

        When stabilization_only is False (Phases A–C), this is an
        improved version of ``_compute_hover_reward`` with:
        * Tighter centering Gaussian  (4.0 × exp(−6 d²) vs 2.0 × exp(−3 d²))
        * Centering velocity bonus    (reward for reducing dist_from_center)
        * Explicit velocity-cancel component in R_stability

        Components — stabilization_only=True
        -------------------------------------
        R_survival    +0.05
        R_stability   0 → +1.0   low angular velocity × low tilt
        R_vel_cancel  0 → +2.0   low linear velocity (encourages hover)

        Components — stabilization_only=False
        --------------------------------------
        R_stability   0 → +1.0   low angular velocity × low tilt
        R_centering   0 → +4.0   tighter Gaussian on centroid distance
        R_center_vel  0 → +1.0   bonus for approaching image centre
        R_scale       0 → +1.0   target fraction near ideal (asymmetric)
        R_invisible   −1.0       per step without seeing the target

        Returns
        -------
        total_reward : float
        info : dict
        """
        info = {}
        state = self.base_env.state
        ang = self.base_env.ang  # [roll, pitch, yaw]

        # R_stability — always active
        ang_vel_norm = np.linalg.norm(state[10:13]) / 10.0
        tilt_norm = (abs(ang[0]) + abs(ang[1])) / (np.pi / 2)
        r_stability = math.exp(-3.0 * ang_vel_norm ** 2) * math.exp(-3.0 * tilt_norm ** 2)

        # ── Phase 0: stabilization only (no visual tracking) ──────────
        if self.stabilization_only:
            r_survival = 0.05
            # R_vel_cancel — reward for low linear velocity
            vel = state[1:6:2]  # [vx, vy, vz]
            vel_norm = np.linalg.norm(vel) / 2.0  # normalise (2 m/s → 1.0)
            r_vel_cancel = 2.0 * math.exp(-5.0 * vel_norm ** 2)

            total = r_survival + r_stability + r_vel_cancel

            info['target_visible'] = False
            info['r_stability'] = float(r_stability)
            info['r_vel_cancel'] = float(r_vel_cancel)
            info['r_centering'] = 0.0
            info['r_scale'] = 0.0
            info['r_invisible'] = 0.0
            info['centering_reward'] = 0.0
            info['scale_reward'] = 0.0
            return float(total), info

        # ── Phases A–C: visual tracking with tighter centering ────────
        cx, cy, frac, vis = self._detect_target_in_image()

        r_centering = 0.0
        r_center_vel = 0.0
        r_scale = 0.0
        r_invisible = 0.0
        target_visible = vis > 0

        if target_visible:
            self._target_ever_seen = True

            # R_centering — TIGHTER Gaussian (σ_eff ≈ 0.41 vs 0.58 in v2)
            dist_from_center = math.sqrt(cx ** 2 + cy ** 2)  # [0, √2]
            r_centering = 4.0 * math.exp(-6.0 * dist_from_center ** 2)

            # R_center_vel — bonus for reducing distance to centre
            delta = self._prev_centering_dist - dist_from_center
            r_center_vel = max(0.0, min(1.0, 3.0 * delta))
            self._prev_centering_dist = dist_from_center

            # R_scale — asymmetric Gaussian (unchanged from v2)
            fraction_error = abs(frac - self.ideal_fraction)
            sigma = 0.12 if frac <= self.ideal_fraction else 0.06
            r_scale = 1.0 * math.exp(-0.5 * (fraction_error / sigma) ** 2)

            info['target_fraction'] = float(frac)
            info['fraction_error'] = float(fraction_error)
            info['target_center'] = (cx, cy)
            info['centering_dist'] = float(dist_from_center)
        else:
            r_invisible = -1.0
            self._prev_centering_dist = 0.0  # reset on lost target

        info['target_visible'] = target_visible
        info['r_stability'] = float(r_stability)
        info['r_centering'] = float(r_centering)
        info['r_center_vel'] = float(r_center_vel)
        info['r_scale'] = float(r_scale)
        info['r_invisible'] = float(r_invisible)
        info['scale_reward'] = float(r_scale)
        info['centering_reward'] = float(r_centering)

        total = r_stability + r_centering + r_center_vel + r_scale + r_invisible
        return float(total), info

    # ================================================================= #
    #  v3.1 hover-tracking reward (stability-gated + action smoothness) #
    # ================================================================= #

    def _compute_v3_1_reward(self, action):
        """Stability-gated reward for hover-tracking v3.1.

        Fine-tune reward designed to fix the instability observed in v3
        medium/hard tiers.  Key changes over v3:

        1. **Multiplicative coupling** — tracking reward is scaled by
           stability, so aggressive corrections that destabilise the
           drone are self-penalised.
        2. **Action smoothness penalty** — penalises sudden motor
           changes (jerk) to encourage gradual corrections.
        3. **Velocity damping** — rewards low linear velocity during
           tracking phases, preventing momentum build-up.

        Phase 0 (stabilization_only=True)
        ----------------------------------
        Identical to v3: R_survival + R_stability + R_vel_cancel.

        Phases A–C (stabilization_only=False)
        --------------------------------------
        R_stability    0 → 1.0   low angular velocity × low tilt
        R_centering    0 → 4.0   tight Gaussian on centroid distance
        R_center_vel   0 → 1.0   bonus for approaching image centre
        R_scale        0 → 1.0   target fraction near ideal (asymmetric)
        R_vel_damp     0 → 0.5   low linear velocity bonus
        R_smooth      −0.3 → 0   jerk penalty (action delta)
        R_invisible   −1.0       per step without seeing the target

        Total = R_stability × (R_tracking + 0.5)
                + R_vel_damp + R_smooth + R_invisible

        where R_tracking = R_centering + R_center_vel + R_scale.

        Parameters
        ----------
        action : np.ndarray
            Current motor action (4-D), used for smoothness penalty.

        Returns
        -------
        total_reward : float
        info : dict
        """
        info = {}
        state = self.base_env.state
        ang = self.base_env.ang  # [roll, pitch, yaw]

        # R_stability — always active (identical to v3)
        ang_vel_norm = np.linalg.norm(state[10:13]) / 10.0
        tilt_norm = (abs(ang[0]) + abs(ang[1])) / (np.pi / 2)
        r_stability = math.exp(-3.0 * ang_vel_norm ** 2) * math.exp(-3.0 * tilt_norm ** 2)

        # ── Phase 0: stabilization only (identical to v3) ────────────
        if self.stabilization_only:
            r_survival = 0.05
            vel = state[1:6:2]
            vel_norm = np.linalg.norm(vel) / 2.0
            r_vel_cancel = 2.0 * math.exp(-5.0 * vel_norm ** 2)

            total = r_survival + r_stability + r_vel_cancel

            info['target_visible'] = False
            info['r_stability'] = float(r_stability)
            info['r_vel_cancel'] = float(r_vel_cancel)
            info['r_centering'] = 0.0
            info['r_scale'] = 0.0
            info['r_invisible'] = 0.0
            info['r_vel_damp'] = 0.0
            info['r_smooth'] = 0.0
            info['centering_reward'] = 0.0
            info['scale_reward'] = 0.0
            return float(total), info

        # ── Phases A–C: stability-gated visual tracking ──────────────
        cx, cy, frac, vis = self._detect_target_in_image()

        r_centering = 0.0
        r_center_vel = 0.0
        r_scale = 0.0
        r_invisible = 0.0
        target_visible = vis > 0

        if target_visible:
            self._target_ever_seen = True

            # R_centering — tight Gaussian (same as v3)
            dist_from_center = math.sqrt(cx ** 2 + cy ** 2)
            r_centering = 4.0 * math.exp(-6.0 * dist_from_center ** 2)

            # R_center_vel — bonus for reducing distance to centre
            delta = self._prev_centering_dist - dist_from_center
            r_center_vel = max(0.0, min(1.0, 3.0 * delta))
            self._prev_centering_dist = dist_from_center

            # R_scale — asymmetric Gaussian (same as v3)
            fraction_error = abs(frac - self.ideal_fraction)
            sigma = 0.12 if frac <= self.ideal_fraction else 0.06
            r_scale = 1.0 * math.exp(-0.5 * (fraction_error / sigma) ** 2)

            info['target_fraction'] = float(frac)
            info['fraction_error'] = float(fraction_error)
            info['target_center'] = (cx, cy)
            info['centering_dist'] = float(dist_from_center)
        else:
            r_invisible = -1.0
            self._prev_centering_dist = 0.0

        # R_vel_damp — reward low linear velocity during tracking
        vel = state[1:6:2]
        vel_norm = np.linalg.norm(vel) / 2.0
        r_vel_damp = 0.5 * math.exp(-4.0 * vel_norm ** 2)

        # R_smooth — penalise sudden action changes (jerk)
        r_smooth = 0.0
        if self._prev_action is not None:
            action_delta = float(np.linalg.norm(
                np.asarray(action) - self._prev_action))
            r_smooth = -0.3 * action_delta ** 2
        self._prev_action = np.asarray(action, dtype=np.float32).copy()

        # ── Multiplicative coupling ──
        # Stability gates all tracking rewards; base bonus 0.5 for hovering
        r_tracking = r_centering + r_center_vel + r_scale
        total = (r_stability * (r_tracking + 0.5)
                 + r_vel_damp + r_smooth + r_invisible)

        info['target_visible'] = target_visible
        info['r_stability'] = float(r_stability)
        info['r_centering'] = float(r_centering)
        info['r_center_vel'] = float(r_center_vel)
        info['r_scale'] = float(r_scale)
        info['r_vel_damp'] = float(r_vel_damp)
        info['r_smooth'] = float(r_smooth)
        info['r_invisible'] = float(r_invisible)
        info['scale_reward'] = float(r_scale)
        info['centering_reward'] = float(r_centering)

        return float(total), info

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

    
    def _detect_target_in_image(self):
        """Detect magenta target in the high-freq camera image via HSV.

        Returns
        -------
        cx, cy : float  centroid in [-1, 1] (normalised image coords)
        frac   : float  fraction of image occupied [0, 1]
        vis    : float  1.0 if visible, 0.0 otherwise
        """
        img = self._last_high_freq_image
        if img is None:
            return 0.0, 0.0, 0.0, 0.0

        h, w = img.shape[:2]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))

        pixel_count = int(np.sum(mask > 0))
        total_pixels = h * w

        if pixel_count <= 2:
            # Invisible: fraction=0 NEVER occurs with visible=1 → unambiguous
            return 0.0, 0.0, 0.0, 0.0

        ys, xs = np.where(mask > 0)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        # Normalise to [-1, 1] (centre of image = 0)
        cx_norm = (cx - w / 2) / (w / 2)
        cy_norm = (cy - h / 2) / (h / 2)
        frac = pixel_count / total_pixels
        return cx_norm, cy_norm, frac, 1.0

    def _build_observation(self, state):
        """
        Build observation according to configuration.

        Args:
            state: Physics state from base environment (np.array)

        Returns:
            observation: Either state alone (if use_camera=False) or Dict with state + images + depth
        """
        # ── v3 centroid-obs: flat 19-D vector ──
        if self.centroid_obs:
            cx, cy, frac, vis = self._detect_target_in_image()
            dcx = cx - self._prev_centroid[0] if vis > 0 else 0.0
            dcy = cy - self._prev_centroid[1] if vis > 0 else 0.0
            self._prev_centroid[:] = [cx, cy]
            extras = np.array([cx, cy, frac, vis, dcx, dcy], dtype=np.float32)
            return np.concatenate([state.astype(np.float32), extras])

        if not self.use_camera:
            # Backward compatibility mode
            return state

        # Build Dict observation with state and camera images
        obs = {
            "state": state,
            "camera_high_freq": self._last_high_freq_image.copy(),
        }

        if not self.exclude_low_freq_camera:
            obs["camera_low_freq"] = self._last_low_freq_image.copy()

        # Add depth if enabled
        if self.use_depth:
            obs["depth_high_freq"] = self._last_high_freq_depth.copy()
            if not self.exclude_low_freq_camera:
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
        # Constrained init: override options to produce near-hover start
        if self.constrained_init and (options is None or 'det_state' not in options):
            init_state = np.zeros(13)
            pr, vr, ar = self.init_pos_range, self.init_vel_range, self.init_ang_range
            avr = self.init_ang_vel_range
            init_state[0:5:2] = (np.random.rand(3) - 0.5) * 2 * pr
            init_state[1:6:2] = (np.random.rand(3) - 0.5) * 2 * vr
            q = euler_quat((np.random.rand(3) - 0.5) * 2 * ar)
            init_state[6:10] = q.flatten()
            init_state[10:13] = (np.random.rand(3) - 0.5) * 2 * avr
            options = {'det_state': init_state}

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
        if self.centroid_obs:
            self._prev_centroid[:] = 0.0
            self._prev_centering_dist = 0.0
        self._prev_action = None

        if self.use_target:
            self._target_ever_seen = False
            self._target_visible_last_step = False
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

        # In filming mode, discard ALL base env rewards (position shaping
        # toward origin, +500 arrival bonus, etc.).  Only keep the boundary
        # violation penalty so the agent learns not to leave the map.
        if self.filming_mode:
            boundary_penalty = -10.0 if self.use_new_reward else -200.0
            reward = boundary_penalty if terminated else 0.0
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
            if self.reward_version == 'v3.1':
                visual_reward, visual_info = self._compute_v3_1_reward(action)
            elif self.reward_version == 'v3':
                visual_reward, visual_info = self._compute_v3_reward()
            elif self.centroid_obs:
                visual_reward, visual_info = self._compute_hover_reward()
            elif self.use_new_reward:
                visual_reward, visual_info = self._compute_new_reward()
            else:
                visual_reward, visual_info = self._compute_visual_tracking_reward()
            reward += visual_reward
            info['visual_tracking'] = visual_info
        
        # Search timeout: truncate if target was never seen after 7s
        # (skip in stabilization-only mode — no target to find)
        if (self.use_target and not self.stabilization_only and
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
