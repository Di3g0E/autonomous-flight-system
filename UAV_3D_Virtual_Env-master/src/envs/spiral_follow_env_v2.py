"""
Spiral Follow Environment v2 — minimal delta from v1.

Two and only two differences vs v1:

  1. Altitude target is RELATIVE to the reset altitude:
         target_z = z0 + climb_offset
     (v1 used a fixed absolute hover_height = 1.39 m, which is why the
     v1 controller could not be deployed from arbitrary altitudes.)

  2. CLIMB phase: for the first `climb_duration_steps` simulation steps
     the spiral reference is frozen at the centre (r = 0, ref velocity
     zero). The drone uses that time to gain `climb_offset` metres of
     altitude before the Archimedes spiral starts expanding.

Everything else — observation layout, reward structure, omega curriculum,
adaptive ω cap — is byte-identical to v1.

Observation (18-D, same as v1)
------------------------------
[0:13]  drone state (ABSOLUTE x, vx, y, vy, z, vz, q0..q3, wx, wy, wz)
[13:15] dx, dy   normalized error to spiral reference
[15:17] vx_n, vy_n   reference velocity direction (zero during climb)
[17]    dz   altitude error to (z0 + climb_offset), normalized by climb_offset
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SpiralFollowEnvV2(gym.Wrapper):
    def __init__(self, env, omega=1.8, r_growth=0.12,
                 climb_offset=0.8, climb_duration_steps=100,
                 vision_radius=0.5, max_tilt=0.25, dt=0.01):
        super().__init__(env)
        self.omega_base = omega
        self.r_growth = r_growth
        self.climb_offset = float(climb_offset)
        self.climb_duration_steps = int(climb_duration_steps)
        self.vision_radius = vision_radius
        self.max_tilt = max_tilt
        self.dt = dt
        self.omega_scale = 1.0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(18,), dtype=np.float32)

        # Spiral state
        self.spiral_step = 0
        self._theta_accum = 0.0
        self._center_x = 0.0
        self._center_y = 0.0
        self._z0 = 0.0
        self._target_z = 0.0

        # Cached reference (ABSOLUTE frame, same as v1)
        self._ref_x = 0.0
        self._ref_y = 0.0
        self._ref_vx = 0.0
        self._ref_vy = 0.0
        self._ref_w = 0.0
        self._ref_r = 0.05

    # ── reference computation ─────────────────────────────────────────

    def _update_reference(self):
        # CLIMB phase: hold reference at centre with zero velocity.
        if self.spiral_step < self.climb_duration_steps:
            self._ref_x = self._center_x
            self._ref_y = self._center_y
            self._ref_vx = 0.0
            self._ref_vy = 0.0
            self._ref_w = 0.0
            self._ref_r = 0.0
            return

        # SPIRAL phase — identical to v1, clock starts at end of climb.
        t = (self.spiral_step - self.climb_duration_steps) * self.dt
        r = self.r_growth * t + 0.05
        omega = self.omega_base * self.omega_scale

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
        q0, q1, q2, q3 = q
        roll = math.atan2(2 * (q0 * q1 + q2 * q3),
                          1 - 2 * (q1 ** 2 + q2 ** 2))
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        yaw = math.atan2(2 * (q0 * q3 + q1 * q2),
                         1 - 2 * (q2 ** 2 + q3 ** 2))
        return roll, pitch, yaw

    def _get_state(self, obs):
        if isinstance(obs, dict):
            return obs['state'].copy()
        return obs.copy()

    def _augment_obs(self, state):
        # Same as v1: absolute state + 5 ref features. The only line that
        # differs from v1 is the dz computation (relative target).
        dx = (self._ref_x - state[0]) / self.vision_radius
        dy = (self._ref_y - state[2]) / self.vision_radius

        v_mag = max(math.sqrt(self._ref_vx ** 2 + self._ref_vy ** 2), 1e-3)
        vx_n = self._ref_vx / v_mag
        vy_n = self._ref_vy / v_mag

        dz = (self._target_z - state[4]) / max(self.climb_offset, 0.1)

        ref = np.array([dx, dy, vx_n, vy_n, dz], dtype=np.float32)
        return np.concatenate([state.astype(np.float32), ref])

    # ── reward ────────────────────────────────────────────────────────

    def _compute_spiral_reward(self, state):
        dx = state[0] - self._ref_x
        dy = state[2] - self._ref_y
        pos_err = math.sqrt(dx ** 2 + dy ** 2)

        r_tracking = 2.0 * math.exp(-5.0 * (pos_err / self.vision_radius) ** 2)

        vx_d, vy_d = state[1], state[3]
        v_ref = math.sqrt(self._ref_vx ** 2 + self._ref_vy ** 2)
        v_drone = math.sqrt(vx_d ** 2 + vy_d ** 2)
        if v_ref > 0.01 and v_drone > 0.01:
            cos_sim = (vx_d * self._ref_vx + vy_d * self._ref_vy) / (v_drone * v_ref)
            r_velocity = 0.5 * (1.0 + cos_sim)
        else:
            r_velocity = 0.5

        alt_err = abs(state[4] - self._target_z)
        r_altitude = 1.0 * math.exp(-10.0 * alt_err ** 2)

        roll, pitch, _ = self._quat_to_euler(state[6:10])
        ang_vel_norm = np.linalg.norm(state[10:13]) / 10.0
        tilt_norm = (abs(roll) + abs(pitch)) / (np.pi / 2)
        r_stability = (math.exp(-3.0 * ang_vel_norm ** 2)
                       * math.exp(-3.0 * tilt_norm ** 2))

        r_progress = 0.1
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
            'target_z': float(self._target_z),
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
        self._z0 = float(state[4])
        self._target_z = self._z0 + self.climb_offset

        # Step 0 → climb phase, ref at centre with zero velocity.
        self._ref_x = self._center_x
        self._ref_y = self._center_y
        self._ref_vx = 0.0
        self._ref_vy = 0.0
        self._ref_w = 0.0
        self._ref_r = 0.0

        return self._augment_obs(state), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        state = self._get_state(obs)

        self.spiral_step += 1
        self._update_reference()

        spiral_reward, spiral_info = self._compute_spiral_reward(state)

        reward = (-10.0 + spiral_reward) if term else spiral_reward
        info['spiral'] = spiral_info

        return self._augment_obs(state), reward, term, trunc, info
