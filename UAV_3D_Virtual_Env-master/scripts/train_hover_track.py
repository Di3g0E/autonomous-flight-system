#!/usr/bin/env python
"""
Train a SAC agent to hover above a magenta sphere, keeping it centered
in a downward-facing camera at the calibrated distance (1.394 m).

Observation : 19-D flat vector (MlpPolicy)
  [0:13]  drone state (position, velocity, quaternion, angular velocity)
  [13:15] centroid_x, centroid_y  (normalised image coords [-1,1])
  [15]    fraction                (target pixel fraction [0,1])
  [16]    visible                 (0/1)
  [17:19] delta_cx, delta_cy      (centroid change since last step)

Action : 4-D continuous  (direct motor throttle [-1,1])

Reward (3 components, range [-1, +4]):
  R_stability  0 → +1.0   low angular velocity × low tilt
  R_centering  0 → +2.0   target centroid close to image centre
  R_scale      0 → +1.0   target fraction near ideal (0.25)
  R_invisible  -1.0        per step without seeing the target

Usage:
    python scripts/train_hover_track.py --timesteps 200000
    python scripts/train_hover_track.py --timesteps 500000 --no-display
"""

import argparse
import csv
import json
import os
import sys
import time
import numpy as np
from collections import deque
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from panda3d.core import Filename, loadPrcFile, loadPrcFileData

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from direct.showbase.ShowBase import ShowBase
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


# ──────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────

class Panda3DRenderCallback(BaseCallback):
    """Advance Panda3D's task manager on every SB3 training step."""
    def __init__(self, app, verbose=0):
        super().__init__(verbose)
        self.app = app

    def _on_step(self):
        self.app.taskMgr.step()
        return True


class HoverTrackCallback(BaseCallback):
    """Metrics logging + monitoring for hover-tracking training."""

    CSV_HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'visibility_pct', 'mean_centering_dist', 'mean_fraction',
        'mean_action_mag',
        'r_stability', 'r_centering', 'r_scale',
    ]

    def __init__(self, env, output_dir, metrics_window=50, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.episode_count = 0
        self.episode_rewards = deque(maxlen=metrics_window)

        # CSV
        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        # Per-episode accumulators
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
        self._ep_r_stability = []
        self._ep_r_centering = []
        self._ep_r_scale = []

    def _on_training_start(self):
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADERS)

    def _on_rollout_end(self):
        # Collect ep rewards from SB3 buffer
        if hasattr(self.model, 'ep_info_buffer'):
            for ep in self.model.ep_info_buffer:
                if 'r' in ep:
                    self.episode_rewards.append(ep['r'])

        # Log every 5s
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
                self._ep_action_mags.append(float(np.mean(np.abs(actions[0]))))

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
        ])
        self._csv_file.flush()

        # Action magnitude alert
        mean_act = _m(self._ep_action_mags)
        if mean_act > 0.3:
            print(f"  [!] mean(|action|)={mean_act:.2f} > 0.3 — "
                  f"agent may be too aggressive")

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
            'csv_path': str(self.csv_path),
        }
        if extras:
            summary.update(extras)
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train hover-tracking agent (SAC, centroid obs)")
    p.add_argument('--timesteps', type=int, default=200_000)
    p.add_argument('--hover-height', type=float, default=1.394,
                   help="Calibrated distance drone→sphere (m)")
    p.add_argument('--max-ep-steps', type=int, default=500,
                   help="Max steps per episode (500 = 5s)")
    p.add_argument('--output-dir', type=str,
                   default='./models/hover_track')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Training app
# ──────────────────────────────────────────────────────────────────────

class HoverTrackApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D scene...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # Aerial camera for visual feedback
        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -8, 14)
        self.cam.lookAt(0, 0, 5)

        # FPV camera — pointing straight DOWN
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.1)   # centered, below drone
        self.fpv_camera.cam.setHpr(0, -90, 0)     # pitch -90° = looking down
        self.fpv_camera.buffer.setActive(1)

        # Environment — centroid obs mode (19-D flat, no CNN)
        print("Creating environment (hover-track, centroid obs)...")
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            # Camera: needed internally for HSV detection, not in obs
            use_camera=True,
            use_depth=False,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            # Target
            use_target=True,
            target_mode='fixed',
            target_speed=0.0,
            target_radius=0.25,
            # Filming mode (no geometric reward from base env)
            filming_mode=True,
            enable_collisions=False,
            n=args.max_ep_steps,
            t_step=0.01,
            direct_control=1,
            # v3: centroid observation + downward camera + hover reward
            centroid_obs=True,
            camera_down=True,
            hover_height=args.hover_height,
            use_new_reward=True,
            # Constrained init: near-hover, drone at ~origin
            constrained_init=True,
            init_pos_range=0.2,
            init_vel_range=0.1,
            init_ang_range=0.05,
        )
        self.env = Monitor(self.env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # SAC — MlpPolicy on 19-D flat obs
        print("Initializing SAC...")
        self.model = SAC(
            'MlpPolicy',
            self.vec_env,
            policy_kwargs={'net_arch': [128, 64]},
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=5_000,
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
        print("  HOVER-TRACK — SAC TRAINING (centroid obs, camera down)")
        print("=" * 70)
        print(f"  Hover height:   {args.hover_height} m (calibrated)")
        print(f"  Observation:    19-D flat (13 state + 4 centroid + 2 delta)")
        print(f"  Policy:         MlpPolicy [128, 64]")
        print(f"  Algorithm:      SAC (auto entropy)")
        print(f"  Timesteps:      {args.timesteps:,}")
        print(f"  Episode steps:  {args.max_ep_steps} ({args.max_ep_steps * 0.01:.0f}s)")
        print(f"  Buffer size:    {300_000:,} (~21 MB)")
        print(f"  learning_starts:{5_000}")
        print(f"  gamma:          0.995 (horizon ~200 steps)")
        print(f"  train_freq:     4, gradient_steps: 4")
        print(f"  Policy params:  {total_params:,}")
        print(f"  Reward range:   [-1.0, +4.0]")
        print(f"  Output:         {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        args = self.args

        render_cb = Panda3DRenderCallback(self)
        self.metrics_cb = HoverTrackCallback(
            env=self.env,
            output_dir=str(self.output_dir),
        )

        print("\nStarting training...\n")
        start = time.time()

        self.model.learn(
            total_timesteps=args.timesteps,
            callback=[render_cb, self.metrics_cb],
            progress_bar=True,
        )

        elapsed = time.time() - start

        # Save
        model_path = self.output_dir / 'best_model'
        self.model.save(str(model_path))

        self.metrics_cb.save_metrics(extras={
            'total_timesteps': args.timesteps,
            'training_time_seconds': elapsed,
            'hover_height': args.hover_height,
            'algorithm': 'SAC',
            'observation_dim': 19,
            'policy_params': sum(
                p.numel() for p in self.model.policy.parameters()),
        })

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Time:      {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Episodes:  {self.metrics_cb.episode_count}")
        print(f"  Model:     {model_path}.zip")
        print("=" * 70)

        self.vec_env.close()


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = HoverTrackApp(args)
    try:
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            p = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(p))
            print(f"Model saved to {p}.zip")
        if hasattr(app, 'metrics_cb'):
            app.metrics_cb.save_metrics()


if __name__ == "__main__":
    main()
