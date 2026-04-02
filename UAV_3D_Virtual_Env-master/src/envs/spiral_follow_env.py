"""
Spiral Follow Environment for training trajectory-following neural networks.

Wraps a quadrotor environment with an Archimedes spiral reference trajectory.
The drone must follow the expanding spiral at constant altitude, simulating
a search pattern activated when the visual target is lost.

Design constraint (0.5 m vision radius)
----------------------------------------
Arm spacing = r_growth × 2π/ω  ≤  2 × vision_radius = 1.0 m

With ω = 1.8 rad/s, r_growth = 0.12 m/s:
    spacing ≈ 0.42 m   (58 % overlap → robust coverage)

Coverage limit (adaptive ω reduces at larger radii):
    r ≈ 0.5 m  →  spacing ≈ 0.42 m  ✓
    r ≈ 2.0 m  →  spacing ≈ 0.82 m  ✓
    r ≈ 3.0 m  →  spacing ≈ 1.01 m  ⚠ borderline
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SpiralFollowEnv(gym.Wrapper):
    """Quadrotor wrapper for Archimedes spiral trajectory following.

    Spiral reference
    ----------------
    r(t) = r_growth · t + r0          (radius grows linearly)
    θ(t) = ∫ω(t) dt                   (angle accumulates with adaptive ω)
    ω(t) = min(ω_base, √(a_budget / r))  (cap centripetal acceleration)

    Observation (18-D)
    ------------------
    [0:13]  drone state  (x, vx, y, vy, z, vz, q0, q1, q2, q3, wx, wy, wz)
    [13:15] position error to reference (dx, dy), normalised by vision_radius
    [15:17] reference velocity direction (vx_n, vy_n), unit vector
    [17]    altitude error (dz), normalised by hover_height

    Reward components
    -----------------
    R_tracking    0 → +2.0   exponential on position error
    R_velocity    0 → +1.0   velocity direction matching (cosine similarity)
    R_altitude    0 → +1.0   hover height maintenance
    R_stability   0 → +1.0   low angular velocity and tilt
    R_progress    +0.1       constant survival incentive
    R_off_track   −0.5       penalty when pos_error > vision_radius

    Parameters
    ----------
    env           : gym.Env   Inner quadrotor environment.
    omega         : float     Base angular rate (rad/s).
    r_growth      : float     Radial expansion (m/s).
    hover_height  : float     Target altitude (m).
    vision_radius : float     Drone's visual detection radius (m).
    max_tilt      : float     Tilt safety clamp (rad).
    dt            : float     Physics time step (s).
    omega_scale   : float     Curriculum multiplier on omega (set externally).
    """

    def __init__(self, env, omega=1.8, r_growth=0.12, hover_height=1.39,
                 vision_radius=0.5, max_tilt=0.25, dt=0.01):
        super().__init__(env)
        self.omega_base = omega
        self.r_growth = r_growth
        self.hover_height = hover_height
        self.vision_radius = vision_radius
        self.max_tilt = max_tilt
        self.dt = dt
        self.omega_scale = 1.0  # curriculum multiplier

        # Override observation space: 13 (state) + 5 (spiral ref) = 18
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(18,), dtype=np.float32)

        # Spiral state (reset in reset())
        self.spiral_step = 0
        self._theta_accum = 0.0
        self._center_x = 0.0
        self._center_y = 0.0

        # Cached reference (updated once per step)
        self._ref_x = 0.0
        self._ref_y = 0.0
        self._ref_vx = 0.0
        self._ref_vy = 0.0
        self._ref_w = 0.0
        self._ref_r = 0.05

    # ── reference computation ─────────────────────────────────────────

    def _update_reference(self):
        """Advance spiral one time-step and cache reference values."""
        t = self.spiral_step * self.dt
        r = self.r_growth * t + 0.05
        omega = self.omega_base * self.omega_scale

        # Adaptive ω: cap so centripetal accel stays within 70 % of tilt budget
        a_budget = 0.70 * 9.82 * math.sin(self.max_tilt)
        w_max = math.sqrt(a_budget / max(r, 0.05))
        w = min(omega, w_max)

        self._theta_accum += w * self.dt
        theta = self._theta_accum

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dr = self.r_growth

        self._ref_x = self._center_x + r * cos_t
        self._ref_y = self._center_y + r * sin_t
        self._ref_vx = dr * cos_t - r * w * sin_t
        self._ref_vy = dr * sin_t + r * w * cos_t
        self._ref_w = w
        self._ref_r = r

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_euler(q):
        """Quaternion [q0, q1, q2, q3] → Euler [roll, pitch, yaw]."""
        q0, q1, q2, q3 = q
        roll = math.atan2(2 * (q0 * q1 + q2 * q3),
                          1 - 2 * (q1 ** 2 + q2 ** 2))
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        yaw = math.atan2(2 * (q0 * q3 + q1 * q2),
                         1 - 2 * (q2 ** 2 + q3 ** 2))
        return roll, pitch, yaw

    def _get_state(self, obs):
        """Extract flat state vector from observation (dict or array)."""
        if isinstance(obs, dict):
            return obs['state'].copy()
        return obs.copy()

    def _augment_obs(self, state):
        """Concatenate state with spiral reference error → 18-D vector."""
        dx = (self._ref_x - state[0]) / self.vision_radius
        dy = (self._ref_y - state[2]) / self.vision_radius

        v_mag = max(math.sqrt(self._ref_vx ** 2 + self._ref_vy ** 2), 1e-3)
        vx_n = self._ref_vx / v_mag
        vy_n = self._ref_vy / v_mag

        dz = (self.hover_height - state[4]) / max(self.hover_height, 0.1)

        ref = np.array([dx, dy, vx_n, vy_n, dz], dtype=np.float32)
        return np.concatenate([state.astype(np.float32), ref])

    # ── reward ────────────────────────────────────────────────────────

    def _compute_spiral_reward(self, state):
        """Multi-component dense reward for spiral following."""
        dx = state[0] - self._ref_x
        dy = state[2] - self._ref_y
        pos_err = math.sqrt(dx ** 2 + dy ** 2)

        # R_tracking: exponential on normalised position error
        r_tracking = 2.0 * math.exp(-5.0 * (pos_err / self.vision_radius) ** 2)

        # R_velocity: cosine similarity with reference velocity
        vx_d, vy_d = state[1], state[3]
        v_ref = math.sqrt(self._ref_vx ** 2 + self._ref_vy ** 2)
        v_drone = math.sqrt(vx_d ** 2 + vy_d ** 2)
        if v_ref > 0.01 and v_drone > 0.01:
            cos_sim = (vx_d * self._ref_vx + vy_d * self._ref_vy) / (v_drone * v_ref)
            r_velocity = 0.5 * (1.0 + cos_sim)  # [0, 1]
        else:
            r_velocity = 0.5

        # R_altitude: Gaussian on height error
        alt_err = abs(state[4] - self.hover_height)
        r_altitude = 1.0 * math.exp(-10.0 * alt_err ** 2)

        # R_stability: low angular velocity × low tilt
        roll, pitch, _ = self._quat_to_euler(state[6:10])
        ang_vel_norm = np.linalg.norm(state[10:13]) / 10.0
        tilt_norm = (abs(roll) + abs(pitch)) / (np.pi / 2)
        r_stability = (math.exp(-3.0 * ang_vel_norm ** 2)
                       * math.exp(-3.0 * tilt_norm ** 2))

        # R_progress: constant survival incentive
        r_progress = 0.1

        # Penalty when too far from reference
        r_off_track = -0.5 if pos_err > self.vision_radius else 0.0

        total = (r_tracking + r_velocity + r_altitude
                 + r_stability + r_progress + r_off_track)

        info = {
            'r_tracking': float(r_tracking),
            'r_velocity': float(r_velocity),
            'r_altitude': float(r_altitude),
            'r_stability': float(r_stability),
            'r_progress': float(r_progress),
            'r_off_track': float(r_off_track),
            'pos_error': float(pos_err),
            'alt_error': float(alt_err),
            'spiral_radius': float(self._ref_r),
            'omega': float(self._ref_w),
            'x_ref': float(self._ref_x),
            'y_ref': float(self._ref_y),
        }
        return float(total), info

    # ── Gym API ───────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self._get_state(obs)

        self.spiral_step = 0
        self._theta_accum = 0.0
        self._center_x = float(state[0])
        self._center_y = float(state[2])

        # Initial reference at step 0 (θ = 0, r = r0)
        r0 = 0.05
        omega = self.omega_base * self.omega_scale
        self._ref_x = self._center_x + r0
        self._ref_y = self._center_y
        self._ref_vx = self.r_growth
        self._ref_vy = r0 * omega
        self._ref_w = omega
        self._ref_r = r0

        return self._augment_obs(state), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        state = self._get_state(obs)

        self.spiral_step += 1
        self._update_reference()

        spiral_reward, spiral_info = self._compute_spiral_reward(state)

        # Override reward; keep boundary penalty on termination
        reward = (-10.0 + spiral_reward) if term else spiral_reward
        info['spiral'] = spiral_info

        return self._augment_obs(state), reward, term, trunc, info
