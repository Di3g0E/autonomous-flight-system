#!/usr/bin/env python
"""
Fine-tune hover-track v3.1 from v3 checkpoint (900k).

Loads a pre-trained v3 SAC model and fine-tunes it with the v3.1
stability-gated reward.  Only Phases B and C are used (the model
already masters easy/medium from v3 training).

Key changes over v3
--------------------
* Multiplicative reward: R_stability gates R_tracking.
* Action smoothness penalty (−0.3 × Δa²) for gradual corrections.
* Velocity damping bonus (+0.5) during tracking phases.
* Lower learning rate (1e-4) to preserve learned behaviours.
* Empty replay buffer (old v3 rewards are incompatible).

Phases (2-phase curriculum)
---------------------------
  B  [0 %, 50 %)   offset 0.3→0.6 m, moderate init
  C  [50 %, 100 %]  offset 0.6→1.0 m, hard init

Usage:
    python scripts/train_hover_track_v3_1.py
    python scripts/train_hover_track_v3_1.py --timesteps 300000
    python scripts/train_hover_track_v3_1.py --base-checkpoint ./models/hover_track_v3/checkpoints/model_900000_steps.zip
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
    """Records periodic episodes to video during fine-tuning."""

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
        self._last_phase = 'B'
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
        cam_dist = 5.0
        self.ext_camera.cam.setPos(
            mx - cam_dist * 0.6,
            my - cam_dist * 0.6,
            mz + cam_dist * 0.4,
        )
        self.ext_camera.cam.lookAt(float(mx), float(my), float(mz - 0.5))

    def compile_timelapse(self):
        if self.recorder.episode_files:
            return self.recorder.compile_timelapse(
                "finetune_timelapse.mp4", max_frames_per_ep=150)
        return None


# ══════════════════════════════════════════════════════════════════════
# 2-phase curriculum (B + C only) + metrics callback
# ══════════════════════════════════════════════════════════════════════

class CurriculumV31Callback(BaseCallback):
    """Two-phase curriculum for v3.1 fine-tune: moderate → hard.

    Phase boundaries (fraction of total timesteps)
    -----------------------------------------------
      B: [0, phase_c)    — moderate offset (0.3→0.6 m)
      C: [phase_c, 1.0]  — hard offset (0.6→1.0 m)
    """

    CSV_HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'visibility_pct', 'mean_centering_dist', 'mean_fraction',
        'mean_action_mag', 'mean_action_jerk',
        'r_stability', 'r_centering', 'r_scale',
        'r_vel_damp', 'r_smooth',
        'phase', 'target_offset', 'init_pos', 'init_vel', 'init_ang',
    ]

    def __init__(self, raw_env, output_dir,
                 phase_c=0.50, metrics_window=50, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_c = phase_c  # start of Phase C

        self.episode_count = 0
        self.episode_rewards = deque(maxlen=metrics_window)

        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        # Start in Phase B
        self.current_phase = 'B'
        self.target_offset_range = 0.3
        self.cur_pos_range = 0.3
        self.cur_vel_range = 0.15
        self.cur_ang_range = 0.05

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

        if progress < self.phase_c:
            # ── Phase B: moderate tracking ──
            self.current_phase = 'B'
            p = progress / self.phase_c
            self.target_offset_range = 0.3 + p * 0.3      # 0.3 → 0.6 m
            self.cur_pos_range = 0.3 + p * 0.2             # 0.3 → 0.5 m
            self.cur_vel_range = 0.15 + p * 0.10           # 0.15 → 0.25
            self.cur_ang_range = 0.05 + p * 0.05           # 0.05 → 0.10
        else:
            # ── Phase C: hard / post-spiral ──
            self.current_phase = 'C'
            p = (progress - self.phase_c) / (1.0 - self.phase_c)
            self.target_offset_range = 0.6 + p * 0.4      # 0.6 → 1.0 m
            self.cur_pos_range = 0.5 + p * 0.3             # 0.5 → 0.8 m
            self.cur_vel_range = 0.25 + p * 0.10           # 0.25 → 0.35
            self.cur_ang_range = 0.10 + p * 0.05           # 0.10 → 0.15

        self.raw_env.stabilization_only = False

        self.raw_env.init_pos_range = self.cur_pos_range
        self.raw_env.init_vel_range = self.cur_vel_range
        self.raw_env.init_ang_range = self.cur_ang_range
        self.raw_env.target_offset_range = self.target_offset_range

        if self.current_phase != prev_phase:
            print(f"\n  Phase {prev_phase} -> {self.current_phase}  "
                  f"(offset={self.target_offset_range:.2f}m  "
                  f"pos={self.cur_pos_range:.2f}  "
                  f"vel={self.cur_vel_range:.2f}  "
                  f"ang={self.cur_ang_range:.2f})")

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
# Environment wrapper (reuse v3 wrapper — identical offset logic)
# ══════════════════════════════════════════════════════════════════════

class OffsetTargetV31Wrapper(Panda3DQuadrotorEnv):
    """Offset target wrapper for v3.1 (no Phase 0 — always tracking)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_offset_range = 0.3   # start at Phase B

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        off = self.target_offset_range
        if off > 0.01:
            dx = (np.random.rand() - 0.5) * 2 * off
            dy = (np.random.rand() - 0.5) * 2 * off
            self.target_pos[0] += dx
            self.target_pos[1] += dy
            self.target_pos = np.clip(self.target_pos, -3.0, 3.0)
            self._update_target_marker_pos()

            if self.use_camera:
                if self.panda3d_app:
                    self.panda3d_app.graphicsEngine.renderFrame()
                self._capture_camera_images(force_capture=True)

            state = self.base_env.state.astype(np.float32)
            obs = self._build_observation(state)

        return obs, info


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune hover-track v3.1 (stability-gated reward)")
    p.add_argument('--base-checkpoint', type=str,
                   default='./models/hover_track_v3/checkpoints/'
                           'model_900000_steps.zip',
                   help="Path to pre-trained v3 checkpoint")
    p.add_argument('--timesteps', type=int, default=500_000)
    p.add_argument('--hover-height', type=float, default=1.394)
    p.add_argument('--max-ep-steps', type=int, default=3000,
                   help="Max steps per episode (3000 = 30s)")
    p.add_argument('--output-dir', type=str,
                   default='./models/hover_track_v3_1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--learning-rate', type=float, default=1e-4,
                   help="Fine-tune LR (lower than training, default: 1e-4)")
    p.add_argument('--phase-c', type=float, default=0.50,
                   help="Fraction at which Phase C starts (default: 0.50)")
    p.add_argument('--checkpoint-freq', type=int, default=50_000)
    p.add_argument('--record-interval', type=int, default=25,
                   help="Record one episode every N episodes (0 = disable)")
    p.add_argument('--record-fps', type=int, default=10)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Fine-tune app
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV31App(ShowBase):
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

        # Environment — v3.1 reward, no stabilization-only mode
        print("Creating environment (hover-track v3.1, fine-tune)...")
        self.raw_env = OffsetTargetV31Wrapper(
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
            init_pos_range=0.3,
            init_vel_range=0.15,
            init_ang_range=0.05,
            reward_version='v3.1',      # ← NEW: stability-gated reward
        )

        self.env = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # ── Load pre-trained model and update environment ──
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

        # Empty replay buffer — old v3 rewards are incompatible
        print("Resetting replay buffer (v3 rewards incompatible with v3.1)...")
        self.model.replay_buffer.reset()

        total_params = sum(p.numel() for p in self.model.policy.parameters())
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ep_duration = args.max_ep_steps * 0.01

        print("\n" + "=" * 70)
        print("  HOVER-TRACK v3.1 — SAC FINE-TUNE (stability-gated reward)")
        print("=" * 70)
        print(f"  Base checkpoint: {base_ckpt.name}")
        print(f"  Hover height:   {args.hover_height} m")
        print(f"  Observation:    19-D flat (13 state + 6 centroid)")
        print(f"  Policy:         MlpPolicy [256, 128]  (preserved)")
        print(f"  Algorithm:      SAC (auto entropy)")
        print(f"  Learning rate:  {args.learning_rate}")
        print(f"  Timesteps:      {args.timesteps:,}")
        print(f"  Episode steps:  {args.max_ep_steps} ({ep_duration:.0f}s)")
        print(f"  Policy params:  {total_params:,}")
        print(f"  Phases:         B[0-{args.phase_c:.0%}] "
              f"C[{args.phase_c:.0%}-100%]")
        print(f"  Reward:         v3.1 (stability-gated + smoothness)")
        print(f"  Buffer:         EMPTY (fresh start)")
        print(f"  Output:         {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        args = self.args

        render_cb = Panda3DRenderCallback(self)

        self.curriculum_cb = CurriculumV31Callback(
            raw_env=self.raw_env,
            output_dir=str(self.output_dir),
            phase_c=args.phase_c,
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

        print("\nStarting fine-tune...\n")
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
            'algorithm': 'SAC',
            'version': 'v3.1',
            'fine_tune': True,
            'observation_dim': 19,
            'net_arch': [256, 128],
            'learning_rate': args.learning_rate,
            'max_ep_steps': args.max_ep_steps,
            'episode_duration_s': args.max_ep_steps * 0.01,
            'reward_version': 'v3.1',
            'curriculum_phases': {
                'B': f'0-{args.phase_c:.0%} (moderate tracking)',
                'C': f'{args.phase_c:.0%}-100% (hard tracking)',
            },
            'v3_1_changes': [
                'Multiplicative coupling: R_stability × (R_tracking + 0.5)',
                'Action smoothness penalty: -0.3 × Δa²',
                'Velocity damping: 0.5 × exp(-4v²)',
                'Centering Gaussian unchanged: 4.0 × exp(-6d²)',
                'Fine-tuned from v3 900k checkpoint',
                'Empty replay buffer (fresh experience)',
            ],
            'policy_params': sum(
                p.numel() for p in self.model.policy.parameters()),
        })

        timelapse_path = None
        if self.video_cb is not None:
            print("\nCompiling fine-tune timelapse...")
            timelapse_path = self.video_cb.compile_timelapse()

        print("\n" + "=" * 70)
        print("  FINE-TUNE COMPLETE")
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
    app = HoverTrackV31App(args)
    try:
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nFine-tune interrupted.")
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
