"""
Spiral Search Controller v2 — minimal delta from v1.

Two differences vs v1 (mirror of SpiralFollowEnvV2):

  1. Altitude target captured at SEARCH activation:
         target_z = z_at_lost + climb_offset
     instead of the fixed hover_height = 1.394 m hardcoded in v1.

  2. CLIMB phase before the spiral: for `climb_duration_steps`
     simulation steps the reference stays at (x_lost, y_lost) with
     zero velocity, so the drone climbs `climb_offset` metres before
     the Archimedes spiral starts expanding.

Everything else (state machine, replay-buffer flag, handoff blending,
observation layout with ABSOLUTE state) is identical to v1.
"""

import math
import numpy as np
from stable_baselines3 import PPO


class SpiralSearchControllerV2:

    TRACK = 'track'
    SEARCH = 'search'
    HANDOFF = 'handoff'

    def __init__(
        self,
        spiral_model_path,
        omega=1.8,
        r_growth=0.12,
        climb_offset=0.8,
        climb_duration_steps=100,
        vision_radius=0.5,
        max_tilt=0.25,
        invisible_threshold=20,
        handoff_steps=15,
    ):
        self.invisible_threshold = invisible_threshold
        self.handoff_steps = handoff_steps

        self.spiral_model = PPO.load(str(spiral_model_path))

        self.omega = omega
        self.r_growth = r_growth
        self.climb_offset = float(climb_offset)
        self.climb_duration_steps = int(climb_duration_steps)
        self.vision_radius = vision_radius
        self.max_tilt = max_tilt

        # Runtime state
        self._state = self.TRACK
        self._invisible_count = 0
        self._handoff_step = 0
        self._last_spiral_action = np.zeros(4)

        # Spiral reference state (ABSOLUTE frame, mirrors SpiralFollowEnvV2)
        self._spiral_step = 0
        self._theta_accum = 0.0
        self._center_x = 0.0
        self._center_y = 0.0
        self._z0 = 0.0
        self._target_z = 0.0
        self._ref_x = 0.0
        self._ref_y = 0.0
        self._ref_vx = 0.0
        self._ref_vy = 0.0

        self.store_transitions = True

    # ── Spiral reference (mirrors SpiralFollowEnvV2) ──────────────────

    def _reset_spiral(self, drone_x, drone_y, drone_z):
        self._spiral_step = 0
        self._theta_accum = 0.0
        self._center_x = float(drone_x)
        self._center_y = float(drone_y)
        self._z0 = float(drone_z)
        self._target_z = self._z0 + self.climb_offset
        self._ref_x = self._center_x
        self._ref_y = self._center_y
        self._ref_vx = 0.0
        self._ref_vy = 0.0

    def _advance_spiral(self, dt=0.01):
        self._spiral_step += 1
        if self._spiral_step < self.climb_duration_steps:
            self._ref_x = self._center_x
            self._ref_y = self._center_y
            self._ref_vx = 0.0
            self._ref_vy = 0.0
            return

        t = (self._spiral_step - self.climb_duration_steps) * dt
        r = self.r_growth * t + 0.05
        a_budget = 0.70 * 9.82 * math.sin(self.max_tilt)
        w_max = math.sqrt(a_budget / max(r, 0.05))
        w = min(self.omega, w_max)

        self._theta_accum += w * dt
        theta = self._theta_accum
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dr = self.r_growth

        self._ref_x = self._center_x + r * cos_t
        self._ref_y = self._center_y + r * sin_t
        self._ref_vx = dr * cos_t - r * w * sin_t
        self._ref_vy = dr * sin_t + r * w * cos_t

    def _build_spiral_obs(self, state_13d):
        # Translate the raw state into the spiral's anchor frame so the
        # network sees a distribution that matches training (drone
        # initialized near origin). The dx/dy/dz features are already
        # frame-invariant, but state[0], state[2], state[4] are not —
        # without this translation, deploying the spiral at arbitrary
        # (x, y, z) feeds the policy out-of-distribution coordinates,
        # which produces erratic motor commands.
        s = state_13d.astype(np.float32).copy()
        s[0] = s[0] - self._center_x
        s[2] = s[2] - self._center_y
        s[4] = s[4] - self._z0

        dx = (self._ref_x - state_13d[0]) / self.vision_radius
        dy = (self._ref_y - state_13d[2]) / self.vision_radius

        v_mag = max(math.sqrt(self._ref_vx ** 2 + self._ref_vy ** 2), 1e-3)
        vx_n = self._ref_vx / v_mag
        vy_n = self._ref_vy / v_mag

        dz = (self._target_z - state_13d[4]) / max(self.climb_offset, 0.1)

        ref = np.array([dx, dy, vx_n, vy_n, dz], dtype=np.float32)
        return np.concatenate([s, ref])

    # ── Public API ────────────────────────────────────────────────────

    def get_action(self, obs_19d, target_visible, tracking_model, env):
        state_13d = env.base_env.state.copy()

        if target_visible:
            self._invisible_count = 0
            if self._state == self.SEARCH:
                # handoff_steps <= 0 disables the blending phase: switch
                # straight from SEARCH to TRACK.
                if self.handoff_steps <= 0:
                    self._state = self.TRACK
                else:
                    self._state = self.HANDOFF
                    self._handoff_step = 0
            elif self._state == self.HANDOFF:
                self._handoff_step += 1
                if self._handoff_step >= self.handoff_steps:
                    self._state = self.TRACK
        else:
            self._invisible_count += 1
            if self._invisible_count >= self.invisible_threshold:
                if self._state == self.TRACK:
                    self._reset_spiral(
                        state_13d[0], state_13d[2], state_13d[4])
                    self._state = self.SEARCH
                elif self._state == self.HANDOFF:
                    self._state = self.SEARCH

        if self._state == self.TRACK:
            self.store_transitions = True
            action, _ = tracking_model.predict(obs_19d, deterministic=True)
            return action

        if self._state == self.SEARCH:
            self.store_transitions = False
            self._advance_spiral(dt=env.base_env.t_step)
            spiral_obs = self._build_spiral_obs(state_13d)
            action, _ = self.spiral_model.predict(
                spiral_obs, deterministic=True)
            self._last_spiral_action = action.copy()
            return action

        # HANDOFF
        self.store_transitions = True
        alpha = self._handoff_step / self.handoff_steps

        self._advance_spiral(dt=env.base_env.t_step)
        spiral_obs = self._build_spiral_obs(state_13d)
        action_spiral, _ = self.spiral_model.predict(
            spiral_obs, deterministic=True)

        action_rl, _ = tracking_model.predict(obs_19d, deterministic=True)

        return (1 - alpha) * action_spiral + alpha * action_rl

    def reset(self):
        self._state = self.TRACK
        self._invisible_count = 0
        self._handoff_step = 0
        self._last_spiral_action = np.zeros(4)
        self.store_transitions = True

    @property
    def current_state(self):
        return self._state

    @property
    def in_climb(self):
        return (self._state == self.SEARCH
                and self._spiral_step < self.climb_duration_steps)
