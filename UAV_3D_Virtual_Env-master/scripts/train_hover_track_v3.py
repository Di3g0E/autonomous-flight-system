#!/usr/bin/env python
"""
Train a robust SAC hover-track agent (v3) with 4-phase curriculum.

Improvements over v2
--------------------
* Phase 0 — Pure stabilisation from perturbed conditions (no target).
  The agent learns to cancel velocity and tilt before any tracking.
* Tighter centering Gaussian (4×exp(−6d²) vs 2×exp(−3d²)).
* Centering-velocity bonus — rewards reducing centroid distance.
* Velocity-cancel reward in Phase 0 — explicit deceleration incentive.
* 30 s episodes (3000 steps) for extended recovery practice.

Phases
------
  0  [0 %, 8 %)    stabilisation only — no target
  A  [8 %, 28 %)   target offset ±0.3 m, gentle init
  B  [28 %, 65 %)  target offset ±0.6 m, moderate init
  C  [65 %, 100 %] target offset ±1.0 m, post-spiral init
                   (lateral velocity ±0.35 m/s, tilt ±0.15 rad)

Designed as a pre-training stage for v4 (moving target fine-tune).
The observation space (19-D) and action space (4-D) are identical to v2,
so the resulting model can be fine-tuned with target_mode='moving'.

Usage:
    python scripts/train_hover_track_v3.py --timesteps 1000000 --no-display
    python scripts/train_hover_track_v3.py --timesteps 1500000
"""

import argparse
import csv
import json
import math
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
# Panda3D render callback (unchanged)
# ══════════════════════════════════════════════════════════════════════

class Panda3DRenderCallback(BaseCallback):
    def __init__(self, app, verbose=0):
        super().__init__(verbose)
        self.app = app

    def _on_step(self):
        self.app.taskMgr.step()
        return True


# ══════════════════════════════════════════════════════════════════════
# Periodic video recording callback
# ══════════════════════════════════════════════════════════════════════

class VideoRecordCallback(BaseCallback):
    """Records periodic episodes to video during training.

    Captures one episode every ``record_interval`` episodes, plus one
    at every phase transition.  Each recorded episode produces an MP4
    with FPV + bird's-eye side-by-side via :class:`EpisodeRecorder`.
    At training end, all clips are compiled into a single timelapse.

    Parameters
    ----------
    raw_env : OffsetTargetV3Wrapper
        Unwrapped environment (to read ``_last_high_freq_image``).
    ext_camera : opencv_camera
        External (bird's-eye) Panda3D camera.
    curriculum_cb : CurriculumV3Callback
        Reference to the curriculum callback (for phase tracking).
    output_dir : str | Path
        Directory for video files.
    record_interval : int
        Record one episode every this many episodes.
    fps : int
        Video frame rate.
    frame_step : int
        Physics steps between captured frames (100//fps by default).
    """

    def __init__(self, raw_env, ext_camera, curriculum_cb, output_dir,
                 record_interval=25, fps=10, verbose=0):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.ext_camera = ext_camera
        self.curriculum_cb = curriculum_cb
        self.record_interval = record_interval
        self.fps = fps
        self.frame_step = max(1, 100 // fps)  # dt=0.01 → 100 steps/s

        self.recorder = EpisodeRecorder(
            output_dir=str(Path(output_dir) / 'recordings'),
            fps=fps,
            resolution=(480, 360),
        )

        # State
        self._episode_count = 0
        self._step_in_ep = 0
        self._is_recording = False
        self._last_phase = '0'
        self._force_next = False   # record next episode on phase change

    def _should_record(self):
        """Decide whether to record the upcoming episode."""
        if self._force_next:
            self._force_next = False
            return True
        return (self._episode_count % self.record_interval) == 0

    def _on_step(self):
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        info = infos[0] if infos else {}

        # Detect phase transitions → force-record next episode
        cur_phase = self.curriculum_cb.current_phase
        if cur_phase != self._last_phase:
            self._force_next = True
            self._last_phase = cur_phase

        # ── Inside a recording episode: capture frame ──
        if self._is_recording:
            self._step_in_ep += 1
            if self._step_in_ep % self.frame_step == 0:
                self._capture(info, rewards[0])

        # ── Episode boundary ──
        if dones[0]:
            if self._is_recording:
                self.recorder.end_episode()
                self._is_recording = False

            self._episode_count += 1

            # Decide whether to record the NEXT episode
            if self._should_record():
                self._is_recording = True
                self._step_in_ep = 0
                self.recorder.start_episode(self._episode_count)
                # Position external camera
                self._position_ext_camera()

        return True

    def _capture(self, info, reward):
        """Capture one video frame (FPV + bird's eye)."""
        # FPV: 32×32 RGB from env
        fpv_img = self.raw_env._last_high_freq_image

        # Bird's eye: external camera
        bird_img = None
        if self.ext_camera is not None:
            ok, rgba = self.ext_camera.get_image()
            if ok:
                bird_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

        # Build info overlay for recorder
        vt = info.get('visual_tracking', {})
        overlay = {
            'visual_tracking': vt,
            'target': info.get('target', {}),
            'Step': self._step_in_ep,
            'Timestep': self.num_timesteps,
            'Reward': round(float(reward), 2),
            'Phase': self.curriculum_cb.current_phase,
        }
        self.recorder.capture_frame(fpv_img, bird_img, overlay)

    def _position_ext_camera(self):
        """Place the external camera to frame the action."""
        if self.ext_camera is None:
            return
        drone_pos = self.raw_env.base_env.state[0:5:2]
        target_pos = self.raw_env.target_pos
        mx = (drone_pos[0] + target_pos[0]) / 2
        my = (drone_pos[1] + target_pos[1]) / 2
        mz = drone_pos[2] + 5   # viz offset
        cam_dist = 5.0
        self.ext_camera.cam.setPos(
            mx - cam_dist * 0.6,
            my - cam_dist * 0.6,
            mz + cam_dist * 0.4,
        )
        self.ext_camera.cam.lookAt(float(mx), float(my), float(mz - 0.5))

    def compile_timelapse(self):
        """Compile all recorded episodes into a single timelapse video."""
        if self.recorder.episode_files:
            return self.recorder.compile_timelapse(
                "training_timelapse.mp4", max_frames_per_ep=150)
        return None


# ══════════════════════════════════════════════════════════════════════
# 4-phase curriculum + metrics callback
# ══════════════════════════════════════════════════════════════════════

class CurriculumV3Callback(BaseCallback):
    """Four-phase curriculum: stabilisation → easy → moderate → hard.

    Phase boundaries (fraction of total timesteps)
    -----------------------------------------------
      0: [0, phase_a)       — stabilise only, no target
      A: [phase_a, phase_b) — gentle, target near centre
      B: [phase_b, phase_c) — moderate offset + wider init
      C: [phase_c, 1.0]    — post-spiral conditions
    """

    CSV_HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'visibility_pct', 'mean_centering_dist', 'mean_fraction',
        'mean_action_mag',
        'r_stability', 'r_centering', 'r_scale',
        'phase', 'target_offset', 'init_pos', 'init_vel', 'init_ang',
    ]

    def __init__(self, raw_env, output_dir,
                 phase_a=0.08, phase_b=0.28, phase_c=0.65,
                 metrics_window=50, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_a = phase_a   # end of Phase 0
        self.phase_b = phase_b   # end of Phase A
        self.phase_c = phase_c   # end of Phase B

        self.episode_count = 0
        self.episode_rewards = deque(maxlen=metrics_window)

        # CSV
        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        # Curriculum state
        self.current_phase = '0'
        self.target_offset_range = 0.0
        self.cur_pos_range = 0.3
        self.cur_vel_range = 0.15
        self.cur_ang_range = 0.05

        # Per-episode accumulators
        self._reset_accum()
        self.start_time = time.time()
        self.last_log_time = time.time()

    # -- accumulators ----------------------------------------------------

    def _reset_accum(self):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_visible = 0
        self._ep_centering = []
        self._ep_fractions = []
        self._ep_action_mags = []
        self._ep_r_stability = []
        self._ep_r_centering = []
        self._ep_r_scale = []

    # -- lifecycle -------------------------------------------------------

    def _on_training_start(self):
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADERS)

    def _on_rollout_end(self):
        # -- Curriculum progression ----------------------------------------
        total = self.locals.get(
            'total_timesteps',
            getattr(self.model, '_total_timesteps', 1))
        progress = min(self.num_timesteps / max(total, 1), 1.0)

        prev_phase = self.current_phase

        if progress < self.phase_a:
            # ── Phase 0: stabilisation only ──
            self.current_phase = '0'
            p = progress / self.phase_a                      # 0 → 1 within 0
            self.cur_pos_range = 0.3 + p * 0.2               # 0.3 → 0.5 m
            self.cur_vel_range = 0.15 + p * 0.20             # 0.15 → 0.35 m/s
            self.cur_ang_range = 0.05 + p * 0.10             # 0.05 → 0.15 rad
            self.target_offset_range = 0.0                   # no target

            # Activate stabilisation-only mode
            self.raw_env.stabilization_only = True

        elif progress < self.phase_b:
            # ── Phase A: gentle tracking ──
            self.current_phase = 'A'
            p = (progress - self.phase_a) / (self.phase_b - self.phase_a)
            self.target_offset_range = 0.1 + p * 0.2        # 0.1 → 0.3 m
            self.cur_pos_range = 0.2 + p * 0.1               # 0.2 → 0.3 m
            self.cur_vel_range = 0.10 + p * 0.05             # 0.10 → 0.15
            self.cur_ang_range = 0.05                        # fixed

            self.raw_env.stabilization_only = False

        elif progress < self.phase_c:
            # ── Phase B: moderate tracking ──
            self.current_phase = 'B'
            p = (progress - self.phase_b) / (self.phase_c - self.phase_b)
            self.target_offset_range = 0.3 + p * 0.3        # 0.3 → 0.6 m
            self.cur_pos_range = 0.3 + p * 0.2               # 0.3 → 0.5 m
            self.cur_vel_range = 0.15 + p * 0.10             # 0.15 → 0.25
            self.cur_ang_range = 0.05 + p * 0.05             # 0.05 → 0.10

            self.raw_env.stabilization_only = False

        else:
            # ── Phase C: hard / post-spiral ──
            self.current_phase = 'C'
            p = (progress - self.phase_c) / (1.0 - self.phase_c)
            self.target_offset_range = 0.6 + p * 0.4        # 0.6 → 1.0 m
            self.cur_pos_range = 0.5 + p * 0.3               # 0.5 → 0.8 m
            self.cur_vel_range = 0.25 + p * 0.10             # 0.25 → 0.35
            self.cur_ang_range = 0.10 + p * 0.05             # 0.10 → 0.15

            self.raw_env.stabilization_only = False

        # Apply to env
        self.raw_env.init_pos_range = self.cur_pos_range
        self.raw_env.init_vel_range = self.cur_vel_range
        self.raw_env.init_ang_range = self.cur_ang_range
        self.raw_env.target_offset_range = self.target_offset_range

        if self.current_phase != prev_phase:
            print(f"\n  Phase {prev_phase} -> {self.current_phase}  "
                  f"(offset={self.target_offset_range:.2f}m  "
                  f"pos={self.cur_pos_range:.2f}  "
                  f"vel={self.cur_vel_range:.2f}  "
                  f"ang={self.cur_ang_range:.2f}  "
                  f"stab_only={self.raw_env.stabilization_only})")

        # Collect ep rewards
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

            if actions is not None:
                self._ep_action_mags.append(
                    float(np.mean(np.abs(actions[0]))))

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
            _m(self._ep_r_stability),
            _m(self._ep_r_centering),
            _m(self._ep_r_scale),
            self.current_phase,
            round(self.target_offset_range, 3),
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
              f"off={self.target_offset_range:.2f}m | "
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
# Environment wrapper — applies target offset after each reset;
# hides target during Phase 0
# ══════════════════════════════════════════════════════════════════════

class OffsetTargetV3Wrapper(Panda3DQuadrotorEnv):
    """Wrapper with Phase 0 support: hides target in stabilisation mode.

    When ``self.stabilization_only`` is True (set by CurriculumV3Callback),
    the target is moved far off-screen so the camera sees nothing, and
    the environment uses the stabilisation-only reward path.

    When stabilization_only is False, this behaves like the v2
    OffsetTargetWrapper — random XY offset applied to the target.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_offset_range = 0.1   # updated by curriculum

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if self.stabilization_only:
            # Move target far away so it is invisible to the camera
            self.target_pos = np.array([50.0, 50.0, -50.0])
            self._update_target_marker_pos()

            # Recapture camera (target now invisible)
            if self.use_camera and self.panda3d_app:
                self.panda3d_app.graphicsEngine.renderFrame()
                self._capture_camera_images(force_capture=True)

            # Rebuild observation with invisible target
            state = self.base_env.state.astype(np.float32)
            obs = self._build_observation(state)
            return obs, info

        # Normal tracking: apply random XY offset
        off = self.target_offset_range
        if off > 0.01:
            dx = (np.random.rand() - 0.5) * 2 * off
            dy = (np.random.rand() - 0.5) * 2 * off
            self.target_pos[0] += dx
            self.target_pos[1] += dy
            self.target_pos = np.clip(self.target_pos, -3.0, 3.0)
            self._update_target_marker_pos()

            # Recapture camera with new target position
            if self.use_camera:
                if self.panda3d_app:
                    self.panda3d_app.graphicsEngine.renderFrame()
                self._capture_camera_images(force_capture=True)

            # Rebuild observation with updated centroid
            state = self.base_env.state.astype(np.float32)
            obs = self._build_observation(state)

        return obs, info


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train robust hover-track SAC v3 "
                    "(4-phase curriculum: stabilise → track)")
    p.add_argument('--timesteps', type=int, default=1_000_000)
    p.add_argument('--hover-height', type=float, default=1.394)
    p.add_argument('--max-ep-steps', type=int, default=3000,
                   help="Max steps per episode (3000 = 30s)")
    p.add_argument('--output-dir', type=str,
                   default='./models/hover_track_v3')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--phase-a', type=float, default=0.08,
                   help="End of Phase 0 (stabilisation)")
    p.add_argument('--phase-b', type=float, default=0.28,
                   help="End of Phase A (gentle tracking)")
    p.add_argument('--phase-c', type=float, default=0.65,
                   help="End of Phase B (moderate tracking)")
    p.add_argument('--checkpoint-freq', type=int, default=50_000)
    p.add_argument('--record-interval', type=int, default=25,
                   help="Record one episode every N episodes (0 = disable)")
    p.add_argument('--record-fps', type=int, default=10,
                   help="FPS for recorded videos")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Training app
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV3App(ShowBase):
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

        # External bird's-eye camera (for video recordings)
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.cam.setPos(0, -6, 12)
        self.ext_camera.cam.lookAt(0, 0, 5)
        self.ext_camera.buffer.setActive(1)

        # Environment — v3 wrapper with stabilisation support
        print("Creating environment (hover-track v3, Phase 0 + offset target)...")
        self.raw_env = OffsetTargetV3Wrapper(
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
            target_mode='fixed',
            target_speed=0.0,
            target_radius=0.25,
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
            init_pos_range=0.3,       # Phase 0 starts with stronger perturbations
            init_vel_range=0.15,
            init_ang_range=0.05,
            reward_version='v3',      # use _compute_v3_reward
        )

        self.env = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # SAC — same architecture as v2 for compatibility with v4 fine-tune
        print("Initializing SAC...")
        self.model = SAC(
            'MlpPolicy',
            self.vec_env,
            policy_kwargs={'net_arch': [256, 128]},
            learning_rate=3e-4,
            buffer_size=500_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            ent_coef='auto',
            train_freq=4,
            gradient_steps=4,
            verbose=1,
            seed=args.seed,
        )

        total_params = sum(p.numel() for p in self.model.policy.parameters())
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ep_duration = args.max_ep_steps * 0.01

        print("\n" + "=" * 70)
        print("  HOVER-TRACK v3 — SAC (4-phase curriculum + stabilisation)")
        print("=" * 70)
        print(f"  Hover height:   {args.hover_height} m")
        print(f"  Observation:    19-D flat (13 state + 6 centroid)")
        print(f"  Policy:         MlpPolicy [256, 128]")
        print(f"  Algorithm:      SAC (auto entropy)")
        print(f"  Timesteps:      {args.timesteps:,}")
        print(f"  Episode steps:  {args.max_ep_steps} ({ep_duration:.0f}s)")
        print(f"  Buffer size:    {500_000:,}")
        print(f"  learning_starts:{10_000}")
        print(f"  Policy params:  {total_params:,}")
        print(f"  Phases:         "
              f"0[0-{args.phase_a:.0%}] "
              f"A[{args.phase_a:.0%}-{args.phase_b:.0%}] "
              f"B[{args.phase_b:.0%}-{args.phase_c:.0%}] "
              f"C[{args.phase_c:.0%}-100%]")
        print(f"  Reward:         v3 (_compute_v3_reward)")
        print(f"  Output:         {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        args = self.args

        render_cb = Panda3DRenderCallback(self)

        self.curriculum_cb = CurriculumV3Callback(
            raw_env=self.raw_env,
            output_dir=str(self.output_dir),
            phase_a=args.phase_a,
            phase_b=args.phase_b,
            phase_c=args.phase_c,
        )

        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cb = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(ckpt_dir),
            name_prefix='model',
        )

        # Video recording callback (periodic episode snapshots)
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
        )

        elapsed = time.time() - start

        # Save final model
        model_path = self.output_dir / 'best_model'
        self.model.save(str(model_path))

        self.curriculum_cb.save_metrics(extras={
            'total_timesteps': args.timesteps,
            'training_time_seconds': elapsed,
            'hover_height': args.hover_height,
            'algorithm': 'SAC',
            'version': 'v3',
            'observation_dim': 19,
            'net_arch': [256, 128],
            'max_ep_steps': args.max_ep_steps,
            'episode_duration_s': args.max_ep_steps * 0.01,
            'reward_version': 'v3',
            'curriculum_phases': {
                '0': f'0-{args.phase_a:.0%} (stabilisation)',
                'A': f'{args.phase_a:.0%}-{args.phase_b:.0%}',
                'B': f'{args.phase_b:.0%}-{args.phase_c:.0%}',
                'C': f'{args.phase_c:.0%}-100%',
            },
            'v3_changes': [
                'Phase 0: stabilisation-only (no target)',
                'Tighter centering Gaussian (4.0*exp(-6*d^2))',
                'Centering velocity bonus',
                'Velocity-cancel reward in Phase 0',
                '30s episodes (3000 steps)',
            ],
            'policy_params': sum(
                p.numel() for p in self.model.policy.parameters()),
        })

        # Compile timelapse from recorded episodes
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
    app = HoverTrackV3App(args)
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
