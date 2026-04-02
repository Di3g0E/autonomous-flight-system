#!/usr/bin/env python
"""
Train a PPO agent to follow an Archimedes spiral trajectory.

The spiral serves as a search pattern for when the drone loses sight
of the visual target.  Arm spacing is designed for a 0.5 m vision
radius, ensuring complete ground coverage during the search.

Architecture: state-only MLP (18-D observation → 4-D motor commands).
Lightweight "binary" policy: one half of the binary decision system
(target visible → tracking policy, target lost → THIS spiral policy).

Curriculum:
  Phase A  [0 , 40 %)   slow spiral (ω×0.3→0.7), tight init
  Phase B  [40 %, 100 %] full-speed spiral (ω×0.7→1.0), wider init

Usage:
    python scripts/train_spiral_follow.py --timesteps 500000
    python scripts/train_spiral_follow.py --timesteps 1000000 --no-display
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.envs.spiral_follow_env import SpiralFollowEnv


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


class SpiralFollowCallback(BaseCallback):
    """
    Metrics, 2-phase curriculum, and periodic recording for spiral training.

    Phases (by training progress):
      A  [0 , phase_b)   slow spiral: ω_scale ramps 0.3 → 0.7
      B  [phase_b, 1.0]  full spiral: ω_scale ramps 0.7 → 1.0
    """

    CSV_HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'mean_pos_error', 'mean_alt_error',
        'omega_scale', 'phase',
        'r_tracking', 'r_velocity', 'r_altitude',
        'r_stability', 'r_progress', 'r_off_track',
    ]

    def __init__(self, spiral_env, output_dir, *,
                 phase_b=0.40,
                 metrics_window=50, verbose=1):
        super().__init__(verbose)
        self.spiral_env = spiral_env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.episode_count = 0
        self.episode_rewards = deque(maxlen=metrics_window)

        # CSV
        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        # Accumulators
        self._reset_accum()

        # Curriculum
        self.phase_b = phase_b
        self._prev_phase = 'A'
        self.current_phase = 'A'
        self.omega_scale = 0.3

        # Domain randomisation level (ratchets up)
        self._dr_level = 0.0

        self._chunk = 0
        self.start_time = time.time()
        self.last_log_time = time.time()

    def _reset_accum(self):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_pos_errors = []
        self._ep_alt_errors = []
        self._ep_r_tracking = []
        self._ep_r_velocity = []
        self._ep_r_altitude = []
        self._ep_r_stability = []
        self._ep_r_progress = []
        self._ep_r_off_track = []

    # ── lifecycle ─────────────────────────────────────────────────────

    def _on_training_start(self):
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADERS)

    def _on_rollout_end(self):
        self._chunk += 1

        # ── Curriculum ────────────────────────────────────────────────
        total = self.locals.get(
            'total_timesteps',
            getattr(self.model, '_total_timesteps', 1))
        progress = min(self.num_timesteps / max(total, 1), 1.0)

        if progress < self.phase_b:
            self.current_phase = 'A'
            local_p = progress / self.phase_b
            self.omega_scale = 0.3 + local_p * (0.7 - 0.3)
        else:
            self.current_phase = 'B'
            local_p = (progress - self.phase_b) / (1.0 - self.phase_b)
            self.omega_scale = 0.7 + local_p * (1.0 - 0.7)

        # Apply omega scale to spiral env
        self.spiral_env.omega_scale = self.omega_scale

        # Phase transition log
        if self.current_phase != self._prev_phase:
            print(f"\n  Phase transition {self._prev_phase} -> "
                  f"{self.current_phase}  |  omega_scale={self.omega_scale:.2f}")
            self._prev_phase = self.current_phase

        # ── Domain randomisation tied to performance ──────────────────
        if self.episode_rewards:
            mean_r = float(np.mean(self.episode_rewards))
            max_steps = 2000
            normalised = mean_r / max(max_steps, 1)
            new_level = np.clip((normalised - 0.5) / 2.5, 0.0, 1.0)
            self._dr_level = max(self._dr_level, new_level)

        dr = self._dr_level
        raw = self.spiral_env.env  # PandaD3DQuadrotorEnv
        raw.init_pos_range = 0.1 + dr * 0.4        # 0.1 → 0.5
        raw.init_vel_range = 0.05 + dr * 0.20       # 0.05 → 0.25
        raw.init_ang_range = 0.03 + dr * 0.07        # 0.03 → 0.10

        # Collect ep rewards from SB3 buffer
        if hasattr(self.model, 'ep_info_buffer'):
            for ep in self.model.ep_info_buffer:
                if 'r' in ep:
                    self.episode_rewards.append(ep['r'])

        # Log every 5 s
        now = time.time()
        if now - self.last_log_time > 5.0:
            self._log()
            self.last_log_time = now

    def _on_step(self):
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones = self.locals.get('dones', [False])

        if infos:
            info = infos[0]
            self._ep_reward += float(rewards[0])
            self._ep_steps += 1

            sp = info.get('spiral', {})
            if sp:
                self._ep_pos_errors.append(sp.get('pos_error', 0))
                self._ep_alt_errors.append(sp.get('alt_error', 0))
                self._ep_r_tracking.append(sp.get('r_tracking', 0))
                self._ep_r_velocity.append(sp.get('r_velocity', 0))
                self._ep_r_altitude.append(sp.get('r_altitude', 0))
                self._ep_r_stability.append(sp.get('r_stability', 0))
                self._ep_r_progress.append(sp.get('r_progress', 0))
                self._ep_r_off_track.append(sp.get('r_off_track', 0))

            if dones[0]:
                self._write_csv()
                self._reset_accum()

        return True

    # ── CSV ────────────────────────────────────────────────────────────

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
            _m(self._ep_pos_errors),
            _m(self._ep_alt_errors),
            round(self.omega_scale, 3),
            self.current_phase,
            _m(self._ep_r_tracking),
            _m(self._ep_r_velocity),
            _m(self._ep_r_altitude),
            _m(self._ep_r_stability),
            _m(self._ep_r_progress),
            _m(self._ep_r_off_track),
        ])
        self._csv_file.flush()

    # ── Logging ────────────────────────────────────────────────────────

    def _log(self):
        elapsed = time.time() - self.start_time
        ts = self.num_timesteps
        total = self.locals.get(
            'total_timesteps',
            getattr(self.model, '_total_timesteps', 0))
        pct = 100 * ts / max(total, 1)
        fps = ts / max(elapsed, 1)
        mean_r = (float(np.mean(self.episode_rewards))
                  if self.episode_rewards else 0)

        print(f"  [{pct:5.1f}%] Step {ts:>7,}/{total:,} | "
              f"Ep={self.episode_count} | R={mean_r:7.1f} | "
              f"Phase={self.current_phase} ω_s={self.omega_scale:.2f} | "
              f"DR={self._dr_level:.2f} | {fps:.0f} fps")

    # ── Save ──────────────────────────────────────────────────────────

    def save_metrics(self, extras=None):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        summary = {
            'total_episodes': self.episode_count,
            'final_mean_reward': (float(np.mean(self.episode_rewards))
                                  if self.episode_rewards else 0),
            'final_omega_scale': self.omega_scale,
            'final_phase': self.current_phase,
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
        description="Train spiral trajectory follower (state-only MLP)")
    p.add_argument('--timesteps', type=int, default=500_000)
    p.add_argument('--omega', type=float, default=1.8,
                   help="Base angular rate (rad/s)")
    p.add_argument('--r-growth', type=float, default=0.12,
                   help="Radial expansion (m/s)")
    p.add_argument('--hover-height', type=float, default=1.39,
                   help="Target altitude (m)")
    p.add_argument('--vision-radius', type=float, default=0.5,
                   help="Drone detection radius (m)")
    p.add_argument('--phase-b', type=float, default=0.40,
                   help="Progress fraction when phase B starts")
    p.add_argument('--n-steps', type=int, default=2048,
                   help="PPO rollout buffer length")
    p.add_argument('--max-ep-steps', type=int, default=2000,
                   help="Max steps per episode (2000 = 20 s)")
    p.add_argument('--output-dir', type=str,
                   default='./models/spiral_follow')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--load-model', type=str, default=None,
                   help="Path to .zip to resume training")
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Training app
# ──────────────────────────────────────────────────────────────────────

class SpiralFollowApp(ShowBase):
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

        # Environment — no camera, no target (spiral is state-only)
        print("Creating environment (spiral follow)...")
        self.base_env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=False,
            use_depth=False,
            use_target=False,
            enable_collisions=False,
            n=args.max_ep_steps,
            t_step=0.01,
            direct_control=1,
            filming_mode=True,
            # v2 init for constrained near-hover starts
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.1,
            init_vel_range=0.05,
            init_ang_range=0.03,
        )

        # Wrap with spiral reference tracking
        self.spiral_env = SpiralFollowEnv(
            self.base_env,
            omega=args.omega,
            r_growth=args.r_growth,
            hover_height=args.hover_height,
            vision_radius=args.vision_radius,
        )

        self.env = Monitor(self.spiral_env)
        self.vec_env = DummyVecEnv([lambda: self.env])
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=False,
            norm_reward=True,
            gamma=0.99,
            clip_reward=10.0,
        )

        # PPO with MLP policy (state-only, 18-D input)
        if args.load_model and os.path.exists(args.load_model):
            print(f"Resuming from {args.load_model}")
            self.model = PPO.load(args.load_model, env=self.vec_env,
                                  device='auto')
        else:
            print("Training from scratch (MLP policy)")
            self.model = PPO(
                'MlpPolicy', self.vec_env,
                policy_kwargs={
                    'net_arch': dict(pi=[64, 32], vf=[64, 32]),
                },
                learning_rate=3e-4,
                n_steps=args.n_steps,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                seed=args.seed,
            )

        total_params = sum(p.numel() for p in self.model.policy.parameters())
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compute arm spacing for display
        arm_spacing = args.r_growth * 2 * np.pi / args.omega
        coverage_ratio = arm_spacing / (2 * args.vision_radius)

        print("\n" + "=" * 70)
        print("  SPIRAL FOLLOW — TRAJECTORY FOLLOWING TRAINING")
        print("=" * 70)
        print(f"  ω base:       {args.omega} rad/s")
        print(f"  r_growth:     {args.r_growth} m/s")
        print(f"  Hover height: {args.hover_height} m")
        print(f"  Vision radius:{args.vision_radius} m")
        print(f"  Arm spacing:  {arm_spacing:.3f} m "
              f"({coverage_ratio:.0%} of vision diameter)")
        print(f"  Phases:       A[0-{args.phase_b:.0%}] "
              f"B[{args.phase_b:.0%}-100%]")
        print(f"  Timesteps:    {args.timesteps:,}")
        print(f"  n_steps:      {args.n_steps}")
        print(f"  Episode max:  {args.max_ep_steps} "
              f"({args.max_ep_steps * 0.01:.0f} s)")
        print(f"  Params:       {total_params:,}")
        print(f"  Observation:  18-D (13 state + 5 spiral ref)")
        print(f"  Output:       {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        args = self.args

        render_cb = Panda3DRenderCallback(self)
        self.metrics_cb = SpiralFollowCallback(
            spiral_env=self.spiral_env,
            output_dir=str(self.output_dir),
            phase_b=args.phase_b,
        )

        print("\nStarting training...\n")
        start = time.time()

        self.model.learn(
            total_timesteps=args.timesteps,
            callback=[render_cb, self.metrics_cb],
            progress_bar=True,
        )

        elapsed = time.time() - start

        # Save model + VecNormalize stats
        model_path = self.output_dir / 'best_model'
        self.model.save(str(model_path))
        vecnorm_path = self.output_dir / 'vecnormalize.pkl'
        self.vec_env.save(str(vecnorm_path))

        self.metrics_cb.save_metrics(extras={
            'total_timesteps': args.timesteps,
            'training_time_seconds': elapsed,
            'omega': args.omega,
            'r_growth': args.r_growth,
            'hover_height': args.hover_height,
            'vision_radius': args.vision_radius,
            'arm_spacing': args.r_growth * 2 * np.pi / args.omega,
            'phase_b': args.phase_b,
            'n_steps': args.n_steps,
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
    app = SpiralFollowApp(args)
    try:
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            p = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(p))
            print(f"Model saved to {p}.zip")
        if hasattr(app, 'vec_env'):
            vn = Path(args.output_dir) / 'vecnormalize.pkl'
            app.vec_env.save(str(vn))
            print(f"VecNormalize saved to {vn}")
        if hasattr(app, 'metrics_cb'):
            app.metrics_cb.save_metrics()


if __name__ == "__main__":
    main()
