"""
Spiral Follow Environment v2 — position- and altitude-invariant search pattern
with an explicit climb phase before the spiral starts expanding.

Differences from v1
-------------------
- Observation is built in a frame **relative to the reset position**:
  the policy receives (x - x0, y - y0, z - z0) instead of absolute world
  coordinates. Quadrotor dynamics (no collisions, constant gravity) are
  translation-invariant, so the policy is invariant to where the search
  is triggered — both in XY and in altitude.
- The spiral has two phases inside one episode:
    * CLIMB   [steps 0 .. climb_duration_steps):   r = 0, drone hovers
              at its initial XY while gaining `climb_offset` metres
              of altitude.
    * SPIRAL  [climb_duration_steps .. end]:       Archimedes spiral
              expands as in v1 (same omega, r_growth, adaptive ω cap).
- `hover_height` is replaced by a relative `climb_offset`. The altitude
  target is always `z0 + climb_offset`, where z0 is the drone's z at
  reset (or, in deployment, the z at which the target was lost).

Observation (18-D, same shape as v1, different content)
-------------------------------------------------------
[0:13]  drone state with positions in relative frame
        (x-x0, vx, y-y0, vy, z-z0, vz, q0, q1, q2, q3, wx, wy, wz)
[13:14] dx, dy   normalized position error to spiral reference
[15:17] vx_n, vy_n   reference velocity direction (unit vector; zero
        during climb because the spiral hasn't started)
[17]    dz   altitude error to (z0 + climb_offset), normalized by
        climb_offset

Reward components (unchanged structure, slightly retuned for the climb)
----------------------------------------------------------------------
R_tracking    0 → +2.0   exponential on XY position error
R_velocity    0 → +1.0   cosine similarity (skipped during climb)
R_altitude    0 → +1.0   exponential on altitude error
R_stability   0 → +1.0   low angular velocity and tilt
R_progress    +0.1       constant survival incentive
R_off_track   −0.5       penalty when XY pos_error > vision_radius

Parameters
----------
env                  : gym.Env  Inner quadrotor environment.
omega                : float    Base angular rate (rad/s).
r_growth             : float    Radial expansion (m/s).
climb_offset         : float    Metres to climb before spiral starts.
climb_duration_steps : int      Sim steps reserved for the climb phase.
vision_radius        : float    Drone's visual detection radius (m).
max_tilt             : float    Tilt safety clamp (rad).
dt                   : float    Physics time step (s).
omega_scale          : float    Curriculum multiplier on omega.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SpiralFollowEnvV2(gym.Wrapper):
    """Quadrotor wrapper for relative-frame spiral with a pre-climb phase."""

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

        # Origin of the relative frame (set on reset)
        self._x0 = 0.0
        self._y0 = 0.0
        self._z0 = 0.0

        # Spiral / climb state
        self.spiral_step = 0
        self._theta_accum = 0.0

        # Cached reference in RELATIVE frame
        self._ref_x_rel = 0.0
        self._ref_y_rel = 0.0
        self._ref_vx = 0.0
        self._ref_vy = 0.0
        self._ref_w = 0.0
        self._ref_r = 0.0

    # ── reference computation ─────────────────────────────────────────

    def _update_reference(self):
        """Advance one time-step. Reference stays at the origin during the
        climb phase, then expands as a normal Archimedes spiral."""
        if self.spiral_step < self.climb_duration_steps:
            # CLIMB phase — drone should hover at relative origin.
            self._ref_x_rel = 0.0
            self._ref_y_rel = 0.0
            self._ref_vx = 0.0
            self._ref_vy = 0.0
            self._ref_w = 0.0
            self._ref_r = 0.0
            return

        # SPIRAL phase — clock starts at end of climb.
        t_spiral = (self.spiral_step - self.climb_duration_steps) * self.dt
        r = self.r_growth * t_spiral + 0.05
        omega = self.omega_base * self.omega_scale

        # Adaptive ω: cap centripetal accel within tilt budget.
        a_budget = 0.70 * 9.82 * math.sin(self.max_tilt)
        w_max = math.sqrt(a_budget / max(r, 0.05))
        w = min(omega, w_max)

        self._theta_accum += w * self.dt
        theta = self._theta_accum

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dr = self.r_growth

        self._ref_x_rel = r * cos_t
        self._ref_y_rel = r * sin_t
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
        """Build 18-D observation in a frame relative to the reset position."""
        state_rel = state.astype(np.float32).copy()
        state_rel[0] = state[0] - self._x0
        state_rel[2] = state[2] - self._y0
        state_rel[4] = state[4] - self._z0

        dx = (self._ref_x_rel - state_rel[0]) / self.vision_radius
        dy = (self._ref_y_rel - state_rel[2]) / self.vision_radius

        v_mag = max(math.sqrt(self._ref_vx ** 2 + self._ref_vy ** 2), 1e-3)
        vx_n = self._ref_vx / v_mag
        vy_n = self._ref_vy / v_mag

        # Altitude error to the climb target (relative frame).
        dz_norm = (self.climb_offset - state_rel[4]) / max(self.climb_offset, 0.1)

        ref = np.array([dx, dy, vx_n, vy_n, dz_norm], dtype=np.float32)
        return np.concatenate([state_rel, ref])

    # ── reward ────────────────────────────────────────────────────────

    def _compute_spiral_reward(self, state):
        # Position error in the relative frame (numerically identical
        # to absolute since both terms are translated by the same x0,y0).
        x_rel = state[0] - self._x0
        y_rel = state[2] - self._y0
        z_rel = state[4] - self._z0

        dx = x_rel - self._ref_x_rel
        dy = y_rel - self._ref_y_rel
        pos_err = math.sqrt(dx ** 2 + dy ** 2)

        # R_tracking
        r_tracking = 2.0 * math.exp(-5.0 * (pos_err / self.vision_radius) ** 2)

        # R_velocity — meaningless during climb (ref is zero); skip then.
        in_climb = self.spiral_step < self.climb_duration_steps
        if in_climb:
            r_velocity = 0.5  # neutral
        else:
            vx_d, vy_d = state[1], state[3]
            v_ref = math.sqrt(self._ref_vx ** 2 + self._ref_vy ** 2)
            v_drone = math.sqrt(vx_d ** 2 + vy_d ** 2)
            if v_ref > 0.01 and v_drone > 0.01:
                cos_sim = (vx_d * self._ref_vx + vy_d * self._ref_vy) / (v_drone * v_ref)
                r_velocity = 0.5 * (1.0 + cos_sim)
            else:
                r_velocity = 0.5

        # R_altitude — target is z0 + climb_offset, i.e. z_rel == climb_offset.
        alt_err = abs(z_rel - self.climb_offset)
        r_altitude = 1.0 * math.exp(-10.0 * alt_err ** 2)

        # R_stability
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
            'in_climb': bool(in_climb),
            'x_ref_rel': float(self._ref_x_rel),
            'y_ref_rel': float(self._ref_y_rel),
            'z_target_rel': float(self.climb_offset),
        }
        return float(total), info

    # ── Gym API ───────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self._get_state(obs)

        self._x0 = float(state[0])
        self._y0 = float(state[2])
        self._z0 = float(state[4])

        self.spiral_step = 0
        self._theta_accum = 0.0

        # Step 0 → climb phase, ref at origin.
        self._ref_x_rel = 0.0
        self._ref_y_rel = 0.0
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
