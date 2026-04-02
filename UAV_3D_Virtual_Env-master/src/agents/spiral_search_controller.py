"""
Spiral Search Controller — uses a pre-trained PPO model to execute an
Archimedes spiral when the visual target is lost.

Integration with the tracking RL policy
----------------------------------------
During evaluation / deployment the controller wraps the tracking policy:

    controller = SpiralSearchController(spiral_model_path, spiral_env_cfg)

    obs = env.reset()
    while not done:
        action = controller.get_action(obs, target_visible, env)
        obs, reward, term, trunc, info = env.step(action)

Internally it manages three states:
  TRACK   – target visible, RL policy drives
  SEARCH  – target lost for K consecutive steps, spiral model drives
  HANDOFF – target just re-acquired, blending spiral → RL over D steps

Replay-buffer safety
--------------------
``controller.store_transitions`` is False during SEARCH (the spiral
model's actions should NOT enter the tracking model's buffer) and True
during TRACK / HANDOFF.
"""

import math
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from src.envs.spiral_follow_env import SpiralFollowEnv


class SpiralSearchController:
    """Manages switching between a tracking RL policy and a spiral search policy.

    Parameters
    ----------
    spiral_model_path : str
        Path to the trained spiral PPO model (.zip).
    omega : float
        Base angular rate for the spiral (rad/s).
    r_growth : float
        Radial expansion rate (m/s).
    hover_height : float
        Altitude the spiral tries to maintain (m).  Overridden at runtime
        to the drone's altitude when the target is lost.
    vision_radius : float
        Detection radius used during spiral training (m).
    invisible_threshold : int
        Consecutive invisible steps before activating search (K).
    handoff_steps : int
        Number of blending steps when transitioning search → track (D).
    """

    # States
    TRACK = 'track'
    SEARCH = 'search'
    HANDOFF = 'handoff'

    def __init__(
        self,
        spiral_model_path,
        omega=1.8,
        r_growth=0.12,
        hover_height=1.39,
        vision_radius=0.5,
        invisible_threshold=20,
        handoff_steps=15,
    ):
        self.invisible_threshold = invisible_threshold
        self.handoff_steps = handoff_steps

        # Load pre-trained spiral model
        self.spiral_model = PPO.load(str(spiral_model_path))

        # Spiral env config (used to build observations for the spiral model)
        self.omega = omega
        self.r_growth = r_growth
        self.hover_height = hover_height
        self.vision_radius = vision_radius

        # Runtime state
        self._state = self.TRACK
        self._invisible_count = 0
        self._handoff_step = 0
        self._last_spiral_action = np.zeros(4)

        # Spiral reference state (replicate SpiralFollowEnv logic)
        self._spiral_step = 0
        self._theta_accum = 0.0
        self._center_x = 0.0
        self._center_y = 0.0
        self._ref_x = 0.0
        self._ref_y = 0.0
        self._ref_vx = 0.0
        self._ref_vy = 0.0

        # Public flag — training script checks this
        self.store_transitions = True

    # ── Spiral reference (mirrors SpiralFollowEnv) ────────────────────

    def _reset_spiral(self, drone_x, drone_y, drone_z):
        """Initialise spiral centered on the drone's current XY position."""
        self._spiral_step = 0
        self._theta_accum = 0.0
        self._center_x = drone_x
        self._center_y = drone_y
        # Lock hover height to current altitude (not the training default)
        self.hover_height = drone_z

        r0 = 0.05
        self._ref_x = self._center_x + r0
        self._ref_y = self._center_y
        self._ref_vx = self.r_growth
        self._ref_vy = r0 * self.omega

    def _advance_spiral(self, dt=0.01):
        """Advance one time-step and return 18-D observation for the spiral model."""
        self._spiral_step += 1
        t = self._spiral_step * dt
        r = self.r_growth * t + 0.05
        a_budget = 0.70 * 9.82 * math.sin(0.25)
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
        """Build 18-D observation expected by the spiral model."""
        dx = (self._ref_x - state_13d[0]) / self.vision_radius
        dy = (self._ref_y - state_13d[2]) / self.vision_radius

        v_mag = max(math.sqrt(self._ref_vx ** 2 + self._ref_vy ** 2), 1e-3)
        vx_n = self._ref_vx / v_mag
        vy_n = self._ref_vy / v_mag

        dz = (self.hover_height - state_13d[4]) / max(self.hover_height, 0.1)

        ref = np.array([dx, dy, vx_n, vy_n, dz], dtype=np.float32)
        return np.concatenate([state_13d.astype(np.float32), ref])

    # ── Public API ────────────────────────────────────────────────────

    def get_action(self, obs_19d, target_visible, tracking_model, env):
        """Return motor action [4] and update internal state machine.

        Parameters
        ----------
        obs_19d : np.ndarray  (19,) flat observation from the centroid env
        target_visible : bool
        tracking_model : SB3 model with .predict()
        env : the Panda3DQuadrotorEnv (for reading base_env.state)

        Returns
        -------
        action : np.ndarray (4,)
        """
        state_13d = env.base_env.state.copy()

        # ── State machine transitions ─────────────────────────────────
        if target_visible:
            self._invisible_count = 0
            if self._state == self.SEARCH:
                # Target found → start blending
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
                    # Lost → initialise spiral at current position
                    self._reset_spiral(state_13d[0], state_13d[2], state_13d[4])
                    self._state = self.SEARCH
                elif self._state == self.HANDOFF:
                    # Lost again during handoff → back to search
                    self._state = self.SEARCH

        # ── Action computation ────────────────────────────────────────
        if self._state == self.TRACK:
            self.store_transitions = True
            action, _ = tracking_model.predict(obs_19d, deterministic=True)
            return action

        if self._state == self.SEARCH:
            self.store_transitions = False
            self._advance_spiral(dt=env.base_env.t_step)
            spiral_obs = self._build_spiral_obs(state_13d)
            action, _ = self.spiral_model.predict(spiral_obs, deterministic=True)
            self._last_spiral_action = action.copy()
            return action

        # HANDOFF: linear blend spiral → RL
        self.store_transitions = True
        alpha = self._handoff_step / self.handoff_steps  # 0 → 1

        self._advance_spiral(dt=env.base_env.t_step)
        spiral_obs = self._build_spiral_obs(state_13d)
        action_spiral, _ = self.spiral_model.predict(spiral_obs, deterministic=True)

        action_rl, _ = tracking_model.predict(obs_19d, deterministic=True)

        action = (1 - alpha) * action_spiral + alpha * action_rl
        return action

    def reset(self):
        """Reset controller state for a new episode."""
        self._state = self.TRACK
        self._invisible_count = 0
        self._handoff_step = 0
        self._last_spiral_action = np.zeros(4)
        self.store_transitions = True

    @property
    def current_state(self):
        return self._state