#!/usr/bin/env python
"""
Fine-tune hover-track v4 — Moving target tracking (lemniscate).

Loads a pre-trained SAC model and fine-tunes it to track a moving
target that follows a Bernoulli lemniscate (∞) trajectory.

The reward reuses the v3.1 stability-gated structure unchanged:
  total = R_stability × (R_tracking + 0.5) + R_vel_damp + R_smooth + R_invisible

R_vel_damp (weight 0.5) is gentle at tracking velocities — at 0.3 m/s
the drone only loses 0.04/step vs R_centering up to 4.0/step — so the
agent naturally learns to move with the target.

No BRAKE or HANDOFF states:  the hard curriculum tier (high velocity,
large offset) trains the agent to recover from post-spiral conditions
directly.

Curriculum (3 phases, FOV-area-based offset — v6)
-------------------------------------------------
  A [0 %, 20 %)    speed 0.05→0.15, target in 0–10 % FOV area (center)
  B [20 %, 65 %)   speed 0.15→0.25, target in 10–75 % FOV area (mid-ring)
  C [65 %, 100 %]  speed capped @ 0.25 m/s, target in 75–100+% FOV
                    (periphery + partially outside, still visible)

v6 reward adjustments (additive, on top of v3.1 base):
  • Stability bonus:  +w_stab × r_stability   (default w_stab=2.0)
  • Extra jerk pen.:  -w_jerk × ||Δaction||²   (default w_jerk=1.2)
  • Altitude penalty:  -w_alt × |Δz|/h_hover   (default w_alt=1.0)

Key changes vs. v4.1:
  • Base checkpoint: v4.1 @ 150K (best eval: 40% global survival,
    60% in medium/fast tiers — already saw the moving target).
  • Crash penalty DISABLED (default 0.0): v4.1 analysis showed it
    created an incentive to shorten episodes (70% < 100 steps in
    Phase B/C). Removing it lets the agent learn long-horizon tracking.
  • Phase A shortened 30% → 20%: the 150K base already knows Phase A.
  • Phase B extended 75% → 80% of the shorter training: more time
    consolidating medium-speed tracking.
  • Timesteps reduced 750K → 400K (starting from a more advanced base).
  • Output directory: ./models/hover_track_v4_2 (preserves v4.1).

Key changes vs. v4 original:
  • Phase C max speed reduced 0.40 → 0.25 m/s (realistic for physics).
  • v3.1 stability-gated reward unchanged (validated architecture).

Usage:
    python scripts/train_hover_track_v4.py
    python scripts/train_hover_track_v4.py --timesteps 500000
    python scripts/train_hover_track_v4.py --max-speed-c 0.30
"""

import argparse
import csv
import json
import os
import sys
import time
import cv2
import numpy as np
from collections import deque
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: F401
from panda3d.core import Filename, loadPrcFile, loadPrcFileData

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from direct.showbase.ShowBase import ShowBase
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.simulation.quaternion_euler_utility import euler_quat
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.utils.episode_recorder import EpisodeRecorder


# ══════════════════════════════════════════════════════════════════════
# Find best available model
# ══════════════════════════════════════════════════════════════════════

def find_best_model():
    """Return path to the best available pre-trained model.

    Preference order (v4.2 defaults):
      1. v4.1 checkpoint @ 150K steps — evaluation data shows this
         checkpoint has 40% global survival and 60% survival in both
         medium (0.25 m/s) and fast (0.40 m/s) tiers, already adapted
         to moving-target tracking.
      2. v3.1 checkpoint @ 400K — fallback for static-target base
         (93% global survival, higher than 500K best_model.zip).
      3. Fallback to v3.1 best_model.zip (typically 500K).
      4. Fallback to v3 checkpoints.
    """
    candidates = [
        './models/hover_track_v4_1/checkpoints/model_150000_steps.zip',
        './models/hover_track_v3_1/checkpoints/model_400000_steps.zip',
        './models/hover_track_v3_1/best_model.zip',
        './models/hover_track_v3/checkpoints/model_900000_steps.zip',
        './models/hover_track_v3/best_model.zip',
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[0]  # will fail with clear error


# ══════════════════════════════════════════════════════════════════════
# Panda3D render callback
# ══════════════════════════════════════════════════════════════════════

class Panda3DRenderCallback(BaseCallback):
    def __init__(self, app, verbose=0):
        super().__init__(verbose)
        self.app = app

    def _on_step(self):
        self.app.taskMgr.step()
        return True


# ══════════════════════════════════════════════════════════════════════
# Video recording callback
# ══════════════════════════════════════════════════════════════════════

class VideoRecordCallback(BaseCallback):
    """Records periodic episodes to video during training."""

    def __init__(self, raw_env, ext_camera, curriculum_cb, output_dir,
                 record_interval=25, fps=10, verbose=0):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.ext_camera = ext_camera
        self.curriculum_cb = curriculum_cb
        self.record_interval = record_interval
        self.fps = fps
        self.frame_step = max(1, 100 // fps)

        self.recorder = EpisodeRecorder(
            output_dir=str(Path(output_dir) / 'recordings'),
            fps=fps,
            resolution=(480, 360),
        )

        self._episode_count = 0
        self._step_in_ep = 0
        self._is_recording = False
        self._last_phase = 'A'
        self._force_next = False

    def _should_record(self):
        if self._force_next:
            self._force_next = False
            return True
        return (self._episode_count % self.record_interval) == 0

    def _on_step(self):
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        info = infos[0] if infos else {}

        cur_phase = self.curriculum_cb.current_phase
        if cur_phase != self._last_phase:
            self._force_next = True
            self._last_phase = cur_phase

        if self._is_recording:
            self._step_in_ep += 1
            if self._step_in_ep % self.frame_step == 0:
                self._capture(info, rewards[0])

        if dones[0]:
            if self._is_recording:
                self.recorder.end_episode()
                self._is_recording = False

            self._episode_count += 1

            if self._should_record():
                self._is_recording = True
                self._step_in_ep = 0
                self.recorder.start_episode(self._episode_count)
                self._position_ext_camera()

        return True

    def _capture(self, info, reward):
        fpv_img = self.raw_env._last_high_freq_image
        bird_img = None
        if self.ext_camera is not None:
            ok, rgba = self.ext_camera.get_image()
            if ok:
                bird_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

        vt = info.get('visual_tracking', {})
        overlay = {
            'visual_tracking': vt,
            'target': info.get('target', {}),
            'Step': self._step_in_ep,
            'Timestep': self.num_timesteps,
            'Reward': round(float(reward), 2),
            'Phase': self.curriculum_cb.current_phase,
            'Speed': round(self.raw_env.target_speed, 3),
        }
        self.recorder.capture_frame(fpv_img, bird_img, overlay)

    def _position_ext_camera(self):
        if self.ext_camera is None:
            return
        drone_pos = self.raw_env.base_env.state[0:5:2]
        target_pos = self.raw_env.target_pos
        mx = (drone_pos[0] + target_pos[0]) / 2
        my = (drone_pos[1] + target_pos[1]) / 2
        mz = drone_pos[2] + 5
        cam_dist = 6.0
        self.ext_camera.cam.setPos(
            mx - cam_dist * 0.6,
            my - cam_dist * 0.6,
            mz + cam_dist * 0.4,
        )
        self.ext_camera.cam.lookAt(float(mx), float(my), float(mz - 0.5))

    def compile_timelapse(self):
        if self.recorder.episode_files:
            return self.recorder.compile_timelapse(
                "v4_training_timelapse.mp4", max_frames_per_ep=150)
        return None


# ══════════════════════════════════════════════════════════════════════
# 3-phase curriculum + metrics callback
# ══════════════════════════════════════════════════════════════════════

class CurriculumV4Callback(BaseCallback):
    """Three-phase curriculum for v4: slow → medium → fast target.

    Phase boundaries (fraction of total timesteps)
    -----------------------------------------------
      A: [0, phase_b)          — slow target, easy init
      B: [phase_b, phase_c)    — medium target, medium init
      C: [phase_c, 1.0]        — fast target, hard init
    """

    CSV_HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'visibility_pct', 'mean_centering_dist', 'mean_fraction',
        'mean_action_mag', 'mean_action_jerk',
        'r_stability', 'r_centering', 'r_scale',
        'r_vel_damp', 'r_smooth',
        'phase', 'target_speed', 'target_offset',
        'init_pos', 'init_vel', 'init_ang',
    ]

    def __init__(self, raw_env, output_dir,
                 phase_b=0.30, phase_c=0.75, max_speed_c=0.25,
                 metrics_window=50, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_b = phase_b
        self.phase_c = phase_c
        self.max_speed_c = max_speed_c

        self.episode_count = 0
        self.episode_rewards = deque(maxlen=metrics_window)

        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        # Start in Phase A
        self.current_phase = 'A'
        self.cur_speed_range = (0.05, 0.15)
        # FOV geometry: r_max = hover_height × (half_film_h / focal_length)
        self.fov_radius = raw_env.hover_height * 12.0 / 45.0
        self.target_sphere_radius = raw_env.target_radius  # 0.25m
        self.target_offset_range = (0.0, np.sqrt(0.10) * self.fov_radius)
        self.cur_pos_range = 0.2
        self.cur_vel_range = 0.10
        self.cur_ang_range = 0.03

        self._reset_accum()
        self.start_time = time.time()
        self.last_log_time = time.time()

    def _reset_accum(self):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_visible = 0
        self._ep_centering = []
        self._ep_fractions = []
        self._ep_action_mags = []
        self._ep_action_jerks = []
        self._ep_r_stability = []
        self._ep_r_centering = []
        self._ep_r_scale = []
        self._ep_r_vel_damp = []
        self._ep_r_smooth = []
        self._prev_action = None

    def _on_training_start(self):
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADERS)

    def _on_rollout_end(self):
        total = self.locals.get(
            'total_timesteps',
            getattr(self.model, '_total_timesteps', 1))
        progress = min(self.num_timesteps / max(total, 1), 1.0)

        prev_phase = self.current_phase

        if progress < self.phase_b:
            # ── Phase A: slow target, easy init ──
            self.current_phase = 'A'
            p = progress / self.phase_b
            speed_lo = 0.05 + p * 0.05              # 0.05 → 0.10
            speed_hi = 0.10 + p * 0.05              # 0.10 → 0.15
            self.cur_speed_range = (speed_lo, speed_hi)
            # Phase A: target in 0–10% FOV area (centered)
            self.target_offset_range = (0.0, np.sqrt(0.10) * self.fov_radius)
            self.cur_pos_range = 0.2 + p * 0.1         # 0.2 → 0.3 m
            self.cur_vel_range = 0.10 + p * 0.05        # 0.10 → 0.15
            self.cur_ang_range = 0.03 + p * 0.02        # 0.03 → 0.05

        elif progress < self.phase_c:
            # ── Phase B: medium target, medium init ──
            self.current_phase = 'B'
            p = (progress - self.phase_b) / (self.phase_c - self.phase_b)
            speed_lo = 0.10 + p * 0.05              # 0.10 → 0.15
            speed_hi = 0.15 + p * 0.10              # 0.15 → 0.25
            self.cur_speed_range = (speed_lo, speed_hi)
            # Phase B: target in 10–75% FOV area (mid-ring)
            self.target_offset_range = (np.sqrt(0.10) * self.fov_radius,
                                        np.sqrt(0.75) * self.fov_radius)
            self.cur_pos_range = 0.3 + p * 0.2         # 0.3 → 0.5 m
            self.cur_vel_range = 0.15 + p * 0.15        # 0.15 → 0.30
            self.cur_ang_range = 0.05 + p * 0.05        # 0.05 → 0.10

        else:
            # ── Phase C: fast target, hard recovery ──
            # Speed hi is capped at max_speed_c (default 0.25 m/s —
            # drone physics cannot reliably follow faster targets).
            self.current_phase = 'C'
            p = (progress - self.phase_c) / (1.0 - self.phase_c)
            max_hi = self.max_speed_c
            max_lo = max(0.15, max_hi - 0.05)
            speed_lo = 0.15 + p * (max_lo - 0.15)     # 0.15 → max_lo
            speed_hi = 0.25 + p * (max_hi - 0.25)     # 0.25 → max_hi
            self.cur_speed_range = (speed_lo, speed_hi)
            # Phase C: target at periphery + partially outside FOV
            # Max offset = fov_radius + target_sphere_radius
            # so the sphere is still partially visible at the edge
            r_max_c = self.fov_radius + self.target_sphere_radius
            self.target_offset_range = (np.sqrt(0.75) * self.fov_radius,
                                        r_max_c)
            self.cur_pos_range = 0.5 + p * 0.3         # 0.5 → 0.8 m
            self.cur_vel_range = 0.30 + p * 0.30        # 0.30 → 0.60
            self.cur_ang_range = 0.10 + p * 0.05        # 0.10 → 0.15

        self.raw_env.stabilization_only = False
        self.raw_env.target_speed_range = self.cur_speed_range
        self.raw_env.target_offset_range = self.target_offset_range
        self.raw_env.init_pos_range = self.cur_pos_range
        self.raw_env.init_vel_range = self.cur_vel_range
        self.raw_env.init_ang_range = self.cur_ang_range

        if self.current_phase != prev_phase:
            print(f"\n  Phase {prev_phase} -> {self.current_phase}  "
                  f"(speed={self.cur_speed_range}  "
                  f"off={self.target_offset_range[0]:.3f}-{self.target_offset_range[1]:.3f}m  "
                  f"vel={self.cur_vel_range:.2f})")

        if hasattr(self.model, 'ep_info_buffer'):
            for ep in self.model.ep_info_buffer:
                if 'r' in ep:
                    self.episode_rewards.append(ep['r'])

        now = time.time()
        if now - self.last_log_time > 5.0:
            self._log()
            self.last_log_time = now

    def _on_step(self):
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones = self.locals.get('dones', [False])
        actions = self.locals.get('actions', None)

        if infos:
            info = infos[0]
            self._ep_reward += float(rewards[0])
            self._ep_steps += 1

            vt = info.get('visual_tracking', {})
            if vt.get('target_visible', False):
                self._ep_visible += 1
                if 'centering_dist' in vt:
                    self._ep_centering.append(vt['centering_dist'])
                if 'target_fraction' in vt:
                    self._ep_fractions.append(vt['target_fraction'])

            self._ep_r_stability.append(vt.get('r_stability', 0))
            self._ep_r_centering.append(vt.get('r_centering', 0))
            self._ep_r_scale.append(vt.get('r_scale', 0))
            self._ep_r_vel_damp.append(vt.get('r_vel_damp', 0))
            self._ep_r_smooth.append(vt.get('r_smooth', 0))

            if actions is not None:
                act = actions[0]
                self._ep_action_mags.append(
                    float(np.mean(np.abs(act))))
                if self._prev_action is not None:
                    self._ep_action_jerks.append(
                        float(np.mean(np.abs(act - self._prev_action))))
                self._prev_action = act.copy()

            if dones[0]:
                self._write_csv()
                self._reset_accum()

        return True

    def _write_csv(self):
        if not self._csv_writer:
            return
        self.episode_count += 1
        _m = lambda lst: round(float(np.mean(lst)), 4) if lst else 0.0
        self._csv_writer.writerow([
            self.episode_count,
            self.num_timesteps,
            round(self._ep_reward, 2),
            self._ep_steps,
            round(100 * self._ep_visible / max(self._ep_steps, 1), 1),
            _m(self._ep_centering),
            _m(self._ep_fractions),
            _m(self._ep_action_mags),
            _m(self._ep_action_jerks),
            _m(self._ep_r_stability),
            _m(self._ep_r_centering),
            _m(self._ep_r_scale),
            _m(self._ep_r_vel_damp),
            _m(self._ep_r_smooth),
            self.current_phase,
            round(self.raw_env.target_speed, 4),
            f"{self.target_offset_range[0]:.3f}-{self.target_offset_range[1]:.3f}",
            round(self.cur_pos_range, 3),
            round(self.cur_vel_range, 3),
            round(self.cur_ang_range, 3),
        ])
        self._csv_file.flush()

    def _log(self):
        elapsed = time.time() - self.start_time
        ts = self.num_timesteps
        total = self.locals.get('total_timesteps',
                                getattr(self.model, '_total_timesteps', 0))
        pct = 100 * ts / max(total, 1)
        fps = ts / max(elapsed, 1)
        mean_r = (float(np.mean(self.episode_rewards))
                  if self.episode_rewards else 0)
        print(f"  [{pct:5.1f}%] Step {ts:>7,}/{total:,} | "
              f"Ep={self.episode_count} | R={mean_r:7.2f} | "
              f"Phase={self.current_phase} "
              f"spd={self.cur_speed_range[1]:.2f} "
              f"off={self.target_offset_range[0]:.3f}-{self.target_offset_range[1]:.3f}m | "
              f"{fps:.0f} fps")

    def save_metrics(self, extras=None):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        summary = {
            'total_episodes': self.episode_count,
            'final_mean_reward': (float(np.mean(self.episode_rewards))
                                  if self.episode_rewards else 0),
            'final_phase': self.current_phase,
            'csv_path': str(self.csv_path),
        }
        if extras:
            summary.update(extras)
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
# Moving target wrapper — positions drone above target at episode start
# ══════════════════════════════════════════════════════════════════════

class MovingTargetV4Wrapper(Panda3DQuadrotorEnv):
    """Wrapper for v4 moving-target training.

    On reset:
      1. The parent sets up a lemniscate phase and target position.
      2. This wrapper repositions the drone above the target + offset.
      3. Target speed is sampled from curriculum-controlled range.
    """

    def __init__(self, *args, crash_penalty_base=10.0,
                 w_stability_bonus=0.0, w_extra_jerk=0.0,
                 w_altitude=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_start_distance = 0.0   # allow any lemniscate phase
        # FOV geometry: r_max = hover_height × (half_film_h / focal_length)
        self.fov_radius_max = self.hover_height * 12.0 / 45.0
        self.target_offset_range = (0.0, np.sqrt(0.10) * self.fov_radius_max)
        self.target_speed_range = (0.05, 0.15)  # set by curriculum
        self.crash_penalty_base = crash_penalty_base

        # ── v6 reward adjustments ──
        self.w_stability_bonus = w_stability_bonus
        self.w_extra_jerk = w_extra_jerk
        self.w_altitude = w_altitude
        self._prev_action_v6 = None

    def reset(self, seed=None, options=None):
        # Sample target speed for this episode
        self.target_speed = np.random.uniform(*self.target_speed_range)

        obs, info = super().reset(seed=seed, options=options)

        # ── Reposition drone above the target ──
        off_min, off_max = self.target_offset_range
        if True:  # always reposition for moving target
            state = self.base_env.state.copy()

            # Polar sampling: uniform in annular area [off_min, off_max]
            r = np.sqrt(np.random.uniform(off_min**2, off_max**2))
            theta = np.random.uniform(0, 2 * np.pi)
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)

            # Place drone above target + offset
            state[0] = self.target_pos[0] + dx     # x
            state[2] = self.target_pos[1] + dy     # y
            state[4] = self.target_pos[2] + self.hover_height  # z

            # Velocity perturbation (curriculum-controlled)
            vr = self.init_vel_range
            state[1] = np.random.uniform(-vr, vr)  # vx
            state[3] = np.random.uniform(-vr, vr)  # vy
            state[5] = np.random.uniform(-0.05, 0.05)  # vz small

            # Keep quaternion [6:10] and angular vel [10:13] from
            # constrained_init (already near-hover)

            # Apply state directly to the underlying physics engine
            self.base_env.state = state.copy()
            self.base_env.previous_state = state.copy()

            # Sync 3D model to new position
            self._update_visualization()
            if self.panda3d_app:
                self.panda3d_app.graphicsEngine.renderFrame()

            # Recapture camera with correct view
            if self.use_camera:
                self._capture_camera_images(force_capture=True)

            obs = self._build_observation(state.astype(np.float32))

        return obs, info

    def step(self, action):
        """Step the env with v6 reward adjustments.

        On top of the v3.1 base reward, v6 adds three components:
          1. Stability bonus:   +w_stab × r_stability
          2. Extra jerk penalty: -w_jerk × ||Δaction||²
          3. Altitude penalty:   -w_alt  × |Δz| / hover_height

        The crash penalty (if enabled) is applied last.
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # ── v6 reward adjustments ─────────────────────────────────────
        r_stab = info.get('r_stability', 0.0)

        # 1. Additive stability bonus — rewards smooth, level flight
        stab_bonus = self.w_stability_bonus * r_stab

        # 2. Extra jerk penalty — penalises violent action changes
        extra_jerk = 0.0
        if self._prev_action_v6 is not None:
            delta = float(np.linalg.norm(
                np.asarray(action) - self._prev_action_v6))
            extra_jerk = -self.w_extra_jerk * delta ** 2
        self._prev_action_v6 = np.asarray(action, dtype=np.float32).copy()

        # 3. Altitude deviation penalty — keeps drone at hover height
        drone_z = self.base_env.state[4]
        ideal_z = self.target_pos[2] + self.hover_height
        alt_error = abs(drone_z - ideal_z) / self.hover_height
        alt_penalty = -self.w_altitude * alt_error

        reward += stab_bonus + extra_jerk + alt_penalty

        info['v6_stability_bonus'] = float(stab_bonus)
        info['v6_extra_jerk'] = float(extra_jerk)
        info['v6_alt_penalty'] = float(alt_penalty)

        # ── Crash penalty (legacy, disabled by default in v4.2+) ──
        if terminated and self.crash_penalty_base > 0.0:
            max_steps = self.base_env.n
            remaining = max(0, max_steps - self._step_counter)
            crash_penalty = -max(
                self.crash_penalty_base,
                0.02 * remaining,
            )
            reward += crash_penalty
            info.setdefault('visual_tracking', {})
            info['visual_tracking']['crash_penalty'] = crash_penalty

        return obs, reward, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    default_ckpt = find_best_model()
    p = argparse.ArgumentParser(
        description="Fine-tune hover-track v4 (moving target, lemniscate)")
    p.add_argument('--base-checkpoint', type=str, default=default_ckpt,
                   help=f"Path to pre-trained checkpoint (default: {default_ckpt})")
    p.add_argument('--timesteps', type=int, default=400_000,
                   help="Total training timesteps (v4.2 default: 400k, "
                        "starting from v4.1/150k)")
    p.add_argument('--hover-height', type=float, default=1.394)
    p.add_argument('--max-ep-steps', type=int, default=3000,
                   help="Max steps per episode (3000 = 30s)")
    p.add_argument('--lemniscate-scale', type=float, default=2.0,
                   help="Lemniscate half-width in metres (default: 2.0)")
    p.add_argument('--output-dir', type=str,
                   default='./models/hover_track_v6')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--phase-b', type=float, default=0.20,
                   help="Fraction at which Phase B starts (default: 0.20)")
    p.add_argument('--phase-c', type=float, default=0.65,
                   help="Fraction at which Phase C starts (default: 0.80)")
    p.add_argument('--max-speed-c', type=float, default=0.25,
                   help="Max target speed in Phase C in m/s (default: 0.25)")
    p.add_argument('--crash-penalty-base', type=float, default=0.0,
                   help="Base crash penalty applied on early termination "
                        "(scaled by remaining steps; default: 0.0 in v4.2 — "
                        "disabled after v4.1 showed it caused short episodes)")

    # ── v6 reward adjustments ──
    p.add_argument('--w-stability-bonus', type=float, default=2.0,
                   help="Additive stability bonus weight (v6: 2.0). "
                        "Rewards smooth, level flight independently of tracking.")
    p.add_argument('--w-extra-jerk', type=float, default=1.2,
                   help="Extra jerk penalty weight (v6: 1.2). "
                        "Added on top of base r_smooth=-0.3. "
                        "Total jerk cost = -(0.3 + 1.2) × delta².")
    p.add_argument('--w-altitude', type=float, default=1.0,
                   help="Altitude deviation penalty weight (v6: 1.0). "
                        "Penalises drift from hover_height above target.")

    p.add_argument('--checkpoint-freq', type=int, default=50_000)
    p.add_argument('--record-interval', type=int, default=25,
                   help="Record one episode every N episodes (0 = disable)")
    p.add_argument('--record-fps', type=int, default=10)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Training app
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV4App(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D scene...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -8, 14)
        self.cam.lookAt(0, 0, 5)

        # FPV camera — pointing DOWN
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.1)
        self.fpv_camera.cam.setHpr(0, -90, 0)
        self.fpv_camera.buffer.setActive(1)

        # External camera (for video recordings)
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.cam.setPos(0, -6, 12)
        self.ext_camera.cam.lookAt(0, 0, 5)
        self.ext_camera.buffer.setActive(1)

        # Environment — moving target on lemniscate, v3.1 reward
        print("Creating environment (hover-track v4, moving target)...")
        self.raw_env = MovingTargetV4Wrapper(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            use_target=True,
            target_mode='moving',
            target_speed=0.10,             # overridden per episode by wrapper
            target_radius=0.25,
            lemniscate_scale=args.lemniscate_scale,
            filming_mode=True,
            enable_collisions=False,
            n=args.max_ep_steps,
            t_step=0.01,
            direct_control=1,
            centroid_obs=True,
            camera_down=True,
            hover_height=args.hover_height,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.2,
            init_vel_range=0.10,
            init_ang_range=0.03,
            reward_version='v3.1',
            crash_penalty_base=args.crash_penalty_base,
            w_stability_bonus=args.w_stability_bonus,
            w_extra_jerk=args.w_extra_jerk,
            w_altitude=args.w_altitude,
        )

        self.env = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # ── Load pre-trained model ──
        base_ckpt = Path(args.base_checkpoint)
        if not base_ckpt.exists():
            print(f"ERROR: Checkpoint not found: {base_ckpt}")
            sys.exit(1)

        print(f"Loading base checkpoint: {base_ckpt.name}")
        self.model = SAC.load(
            str(base_ckpt),
            env=self.vec_env,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

        # Empty replay buffer — old rewards are for static target
        print("Resetting replay buffer (static-target transitions incompatible)...")
        self.model.replay_buffer.reset()

        total_params = sum(p.numel() for p in self.model.policy.parameters())
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ep_duration = args.max_ep_steps * 0.01

        print("\n" + "=" * 70)
        print("  HOVER-TRACK v6 — SAC FINE-TUNE (stability-aware tracking)")
        print("=" * 70)
        print(f"  Base checkpoint: {base_ckpt}")
        print(f"  Hover height:   {args.hover_height} m")
        print(f"  Lemniscate:     scale={args.lemniscate_scale}m "
              f"(width={2*args.lemniscate_scale}m)")
        print(f"  Observation:    19-D flat (13 state + 6 centroid)")
        print(f"  Policy:         MlpPolicy [256, 128]  (preserved)")
        print(f"  Algorithm:      SAC (auto entropy)")
        print(f"  Reward:         v3.1 base + v6 adjustments")
        print(f"    ├─ Stability bonus:  w={args.w_stability_bonus}")
        print(f"    ├─ Extra jerk pen.:  w={args.w_extra_jerk} "
              f"(total jerk = -{0.3 + args.w_extra_jerk:.1f}×Δ²)")
        print(f"    └─ Altitude penalty: w={args.w_altitude}")
        print(f"  Learning rate:  {args.learning_rate}")
        print(f"  Timesteps:      {args.timesteps:,}")
        print(f"  Episode steps:  {args.max_ep_steps} ({ep_duration:.0f}s)")
        print(f"  Policy params:  {total_params:,}")
        print(f"  Phases:         A[0-{args.phase_b:.0%}] "
              f"B[{args.phase_b:.0%}-{args.phase_c:.0%}] "
              f"C[{args.phase_c:.0%}-100%]")
        print(f"  Max speed C:    {args.max_speed_c} m/s")
        print(f"  Crash penalty:  base={args.crash_penalty_base}")
        print(f"  Buffer:         EMPTY (fresh start)")
        print(f"  Output:         {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        args = self.args

        render_cb = Panda3DRenderCallback(self)

        self.curriculum_cb = CurriculumV4Callback(
            raw_env=self.raw_env,
            output_dir=str(self.output_dir),
            phase_b=args.phase_b,
            phase_c=args.phase_c,
            max_speed_c=args.max_speed_c,
        )

        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cb = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(ckpt_dir),
            name_prefix='model',
        )

        callbacks = [render_cb, self.curriculum_cb, ckpt_cb]
        self.video_cb = None
        if args.record_interval > 0:
            self.video_cb = VideoRecordCallback(
                raw_env=self.raw_env,
                ext_camera=self.ext_camera,
                curriculum_cb=self.curriculum_cb,
                output_dir=str(self.output_dir),
                record_interval=args.record_interval,
                fps=args.record_fps,
            )
            callbacks.append(self.video_cb)
            print(f"  Video recording: every {args.record_interval} episodes "
                  f"@ {args.record_fps} FPS")

        print("\nStarting training...\n")
        start = time.time()

        self.model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True,
        )

        elapsed = time.time() - start

        # Save final model
        model_path = self.output_dir / 'best_model'
        self.model.save(str(model_path))

        self.curriculum_cb.save_metrics(extras={
            'total_timesteps': args.timesteps,
            'training_time_seconds': elapsed,
            'base_checkpoint': str(args.base_checkpoint),
            'hover_height': args.hover_height,
            'lemniscate_scale': args.lemniscate_scale,
            'algorithm': 'SAC',
            'version': 'v4.2',
            'fine_tune': True,
            'observation_dim': 19,
            'net_arch': [256, 128],
            'learning_rate': args.learning_rate,
            'max_ep_steps': args.max_ep_steps,
            'episode_duration_s': args.max_ep_steps * 0.01,
            'reward_version': 'v3.1 (unchanged)',
            'target_mode': 'moving (lemniscate)',
            'curriculum_phases': {
                'A': f'0-{args.phase_b:.0%} (slow target, easy init)',
                'B': f'{args.phase_b:.0%}-{args.phase_c:.0%} (medium target)',
                'C': f'{args.phase_c:.0%}-100% '
                     f'(capped @ {args.max_speed_c} m/s, hard init)',
            },
            'max_speed_c': args.max_speed_c,
            'crash_penalty_base': args.crash_penalty_base,
            'v4_changes': [
                'Moving target (Bernoulli lemniscate trajectory)',
                'Drone repositioned above target at episode start',
                f'Progressive target speed curriculum '
                f'(0.05 → {args.max_speed_c})',
                'Phase schedule '
                f'({args.phase_b:.0%}-{args.phase_c:.0%}-100%)',
                (f'Crash penalty DISABLED (base={args.crash_penalty_base}) '
                 '— v4.2 removes v4.1 penalty that caused short episodes'
                 if args.crash_penalty_base == 0.0
                 else f'Crash penalty base={args.crash_penalty_base} '
                      '(scaled by remaining steps)'),
                'Base: v4.1/150k checkpoint (best eval: 40% survival)',
                'v3.1 stability-gated reward unchanged',
                f'Lemniscate scale: {args.lemniscate_scale}m',
            ],
            'policy_params': sum(
                p.numel() for p in self.model.policy.parameters()),
        })

        timelapse_path = None
        if self.video_cb is not None:
            print("\nCompiling training timelapse...")
            timelapse_path = self.video_cb.compile_timelapse()

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Time:       {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Episodes:   {self.curriculum_cb.episode_count}")
        print(f"  Model:      {model_path}.zip")
        print(f"  Checkpoints:{ckpt_dir}/")
        print(f"  Log:        {self.curriculum_cb.csv_path}")
        if timelapse_path:
            print(f"  Timelapse:  {timelapse_path}")
        n_recordings = (len(self.video_cb.recorder.episode_files)
                        if self.video_cb else 0)
        print(f"  Recordings: {n_recordings} episodes")
        print("=" * 70)

        self.vec_env.close()


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = HoverTrackV4App(args)
    try:
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            p = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(p))
            print(f"Model saved to {p}.zip")
        if hasattr(app, 'curriculum_cb'):
            app.curriculum_cb.save_metrics()
        if hasattr(app, 'video_cb') and app.video_cb is not None:
            print("Compiling timelapse from recorded episodes...")
            app.video_cb.compile_timelapse()


if __name__ == "__main__":
    main()
