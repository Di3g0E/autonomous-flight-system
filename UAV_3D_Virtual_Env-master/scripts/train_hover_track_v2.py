#!/usr/bin/env python
"""
Train a robust SAC hover-track agent (v2) with curriculum learning.

Improvements over v1
--------------------
* Target offset randomisation — the target is NOT always centred in the
  image.  Offset grows across phases so the agent learns to re-centre.
* Post-spiral initialisation — in later phases the drone starts with
  lateral velocity and tilt, simulating recovery after a spiral search.
* Longer episodes (1500 steps = 15 s) for recovery practice.
* Larger replay buffer and network for the richer data distribution.
* Checkpoints every 50 k steps.

Phases
------
  A  [0 %, 30 %)   target offset ±0.3 m, gentle init
  B  [30 %, 70 %)  target offset ±0.6 m, moderate init
  C  [70 %, 100 %] target offset ±1.0 m, post-spiral init
                   (lateral velocity ±0.35 m/s, tilt ±0.15 rad)

The original model in models/hover_track/ is never touched.
Output goes to models/hover_track_v2/.

Usage:
    python scripts/train_hover_track_v2.py --timesteps 500000 --no-display
    python scripts/train_hover_track_v2.py --timesteps 300000
"""

import argparse
import csv
import json
import math
import os
import sys
import time
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


# ══════════════════════════════════════════════════════════════════════
# Panda3D render callback (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════

class Panda3DRenderCallback(BaseCallback):
    def __init__(self, app, verbose=0):
        super().__init__(verbose)
        self.app = app

    def _on_step(self):
        self.app.taskMgr.step()
        return True


# ══════════════════════════════════════════════════════════════════════
# Curriculum + metrics callback
# ══════════════════════════════════════════════════════════════════════

class CurriculumCallback(BaseCallback):
    """Three-phase curriculum that progressively increases difficulty.

    Controls two things on every rollout boundary:
      1.  Target XY offset range (applied after each env.reset).
      2.  Init condition ranges (position, velocity, angular).

    Phase boundaries
    ----------------
      A: [0, phase_b)      — gentle, target near centre
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
                 phase_b=0.30, phase_c=0.70,
                 metrics_window=50, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env          # unwrapped Panda3DQuadrotorEnv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_b = phase_b
        self.phase_c = phase_c

        self.episode_count = 0
        self.episode_rewards = deque(maxlen=metrics_window)

        # CSV
        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        # Curriculum state
        self.current_phase = 'A'
        self.target_offset_range = 0.3
        self.cur_pos_range = 0.2
        self.cur_vel_range = 0.10
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

        if progress < self.phase_b:
            self.current_phase = 'A'
            p = progress / self.phase_b                   # 0 → 1 within A
            self.target_offset_range = 0.1 + p * 0.2     # 0.1 → 0.3 m
            self.cur_pos_range = 0.2 + p * 0.1            # 0.2 → 0.3 m
            self.cur_vel_range = 0.10 + p * 0.05          # 0.10 → 0.15
            self.cur_ang_range = 0.05                     # fixed

        elif progress < self.phase_c:
            self.current_phase = 'B'
            p = (progress - self.phase_b) / (self.phase_c - self.phase_b)
            self.target_offset_range = 0.3 + p * 0.3     # 0.3 → 0.6 m
            self.cur_pos_range = 0.3 + p * 0.2            # 0.3 → 0.5 m
            self.cur_vel_range = 0.15 + p * 0.10          # 0.15 → 0.25
            self.cur_ang_range = 0.05 + p * 0.05          # 0.05 → 0.10

        else:
            self.current_phase = 'C'
            p = (progress - self.phase_c) / (1.0 - self.phase_c)
            self.target_offset_range = 0.6 + p * 0.4     # 0.6 → 1.0 m
            self.cur_pos_range = 0.5 + p * 0.3            # 0.5 → 0.8 m
            self.cur_vel_range = 0.25 + p * 0.10          # 0.25 → 0.35
            self.cur_ang_range = 0.10 + p * 0.05          # 0.10 → 0.15

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
                  f"ang={self.cur_ang_range:.2f})")

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
# Environment wrapper — applies target offset after each reset
# ══════════════════════════════════════════════════════════════════════

class OffsetTargetWrapper(Panda3DQuadrotorEnv):
    """Thin wrapper that shifts the target by a random XY offset after reset.

    The curriculum callback sets ``target_offset_range`` dynamically.
    On each reset this wrapper adds a random offset to the target
    position so the target is NOT always centred in the image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_offset_range = 0.1   # updated by curriculum

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Apply random XY offset to target
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
        description="Train robust hover-track SAC v2 (curriculum)")
    p.add_argument('--timesteps', type=int, default=500_000)
    p.add_argument('--hover-height', type=float, default=1.394)
    p.add_argument('--max-ep-steps', type=int, default=1500,
                   help="Max steps per episode (1500 = 15s)")
    p.add_argument('--output-dir', type=str,
                   default='./models/hover_track_v2')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--phase-b', type=float, default=0.30)
    p.add_argument('--phase-c', type=float, default=0.70)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Training app
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV2App(ShowBase):
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

        # Environment — with offset wrapper
        print("Creating environment (hover-track v2, offset target)...")
        self.raw_env = OffsetTargetWrapper(
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
            init_pos_range=0.2,
            init_vel_range=0.10,
            init_ang_range=0.05,
        )

        self.env = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # SAC — larger network for richer distribution
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

        print("\n" + "=" * 70)
        print("  HOVER-TRACK v2 — ROBUST SAC (curriculum + offset target)")
        print("=" * 70)
        print(f"  Hover height:   {args.hover_height} m")
        print(f"  Observation:    19-D flat (13 state + 6 centroid)")
        print(f"  Policy:         MlpPolicy [256, 128]")
        print(f"  Algorithm:      SAC (auto entropy)")
        print(f"  Timesteps:      {args.timesteps:,}")
        print(f"  Episode steps:  {args.max_ep_steps} "
              f"({args.max_ep_steps * 0.01:.0f}s)")
        print(f"  Buffer size:    {500_000:,}")
        print(f"  learning_starts:{10_000}")
        print(f"  Policy params:  {total_params:,}")
        print(f"  Phases:         A[0-{args.phase_b:.0%}] "
              f"B[{args.phase_b:.0%}-{args.phase_c:.0%}] "
              f"C[{args.phase_c:.0%}-100%]")
        print(f"  Output:         {args.output_dir}")
        print(f"  Original model: models/hover_track/ (untouched)")
        print("=" * 70)

    def run_training(self):
        args = self.args

        render_cb = Panda3DRenderCallback(self)

        self.curriculum_cb = CurriculumCallback(
            raw_env=self.raw_env,
            output_dir=str(self.output_dir),
            phase_b=args.phase_b,
            phase_c=args.phase_c,
        )

        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=str(ckpt_dir),
            name_prefix='model',
        )

        print("\nStarting training...\n")
        start = time.time()

        self.model.learn(
            total_timesteps=args.timesteps,
            callback=[render_cb, self.curriculum_cb, ckpt_cb],
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
            'version': 'v2',
            'observation_dim': 19,
            'net_arch': [256, 128],
            'curriculum_phases': {
                'A': f'0-{args.phase_b:.0%}',
                'B': f'{args.phase_b:.0%}-{args.phase_c:.0%}',
                'C': f'{args.phase_c:.0%}-100%',
            },
            'policy_params': sum(
                p.numel() for p in self.model.policy.parameters()),
        })

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Time:       {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Episodes:   {self.curriculum_cb.episode_count}")
        print(f"  Model:      {model_path}.zip")
        print(f"  Checkpoints:{ckpt_dir}/")
        print(f"  Log:        {self.curriculum_cb.csv_path}")
        print("=" * 70)

        self.vec_env.close()


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = HoverTrackV2App(args)
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


if __name__ == "__main__":
    main()
