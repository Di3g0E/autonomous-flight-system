#!/usr/bin/env python
"""
Train a PPO agent to follow a magenta sphere moving in a lemniscate (∞).

The agent observes ONLY its FPV camera (32×32) and its own motion sensors
(13D state: position, velocity, quaternion, angular rates).
No target position is provided — the agent must learn to detect, follow,
and maintain the target at the desired fraction of the camera frame.

Reward:
  - Target occupies ideal_fraction ± tolerance → positive exponential
    (max = max_visual_reward at exact ideal_fraction)
  - Outside tolerance → negative exponential penalty
  - Target not visible → small fixed penalty (-5)

Usage:
    python scripts/train_lemniscate_follower.py --timesteps 1000000
    python scripts/train_lemniscate_follower.py --timesteps 500000 --scale 5.0 --ideal-fraction 0.25
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from panda3d.core import Filename, loadPrcFile, loadPrcFileData
from direct.showbase.ShowBase import ShowBase

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.agents.feature_extractors import StateCameraExtractor
from src.utils.episode_recorder import EpisodeRecorder


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


class LemniscateMetricsCallback(BaseCallback):
    """
    Track training metrics, handle speed curriculum, and periodic recording.
    """

    CSV_HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'mean_distance', 'visibility_pct',
        'mean_fraction', 'mean_fraction_error',
        'target_speed',
    ]

    def __init__(self, env, output_dir,
                 initial_speed=0.05, max_speed=0.3,
                 video_recorder=None, record_interval=50,
                 metrics_window=50, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from collections import deque
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=metrics_window)

        # CSV
        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        # Per-episode accumulators
        self._reset_accum()

        # Speed curriculum
        self.initial_speed = initial_speed
        self.max_speed = max_speed
        self.current_speed = initial_speed

        # Recording
        self.video_recorder = video_recorder
        self.record_interval = record_interval
        self._chunk = 0

        self.start_time = time.time()
        self.last_log_time = time.time()

    def _reset_accum(self):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_distances = []
        self._ep_visible = 0
        self._ep_fractions = []
        self._ep_errors = []

    def _on_training_start(self):
        import csv
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADERS)

    def _on_rollout_end(self):
        self._chunk += 1
        raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env

        # Speed curriculum: linear with training progress
        total = self.locals.get('total_timesteps',
                                getattr(self.model, '_total_timesteps', 1))
        progress = min(self.num_timesteps / max(total, 1), 1.0)
        self.current_speed = (self.initial_speed +
                              progress * (self.max_speed - self.initial_speed))
        raw.target_speed = self.current_speed

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

        # Record
        if (self.video_recorder and
                self._chunk % self.record_interval == 0):
            self._record()

    def _on_step(self):
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones = self.locals.get('dones', [False])

        if infos:
            info = infos[0]
            self._ep_reward += float(rewards[0])
            self._ep_steps += 1

            t = info.get('target', {})
            if 'distance_to_target' in t:
                self._ep_distances.append(t['distance_to_target'])

            vt = info.get('visual_tracking', {})
            if vt.get('target_visible', False):
                self._ep_visible += 1
                self._ep_fractions.append(vt.get('target_fraction', 0))
                self._ep_errors.append(vt.get('fraction_error', 0))

            if dones[0]:
                self._write_csv()
                self._reset_accum()

        return True

    def _write_csv(self):
        if not self._csv_writer:
            return
        self.episode_count += 1
        d = self._ep_distances
        self._csv_writer.writerow([
            self.episode_count,
            self.num_timesteps,
            round(self._ep_reward, 2),
            self._ep_steps,
            round(float(np.mean(d)), 3) if d else 0,
            round(100 * self._ep_visible / max(self._ep_steps, 1), 1),
            round(float(np.mean(self._ep_fractions)), 4) if self._ep_fractions else 0,
            round(float(np.mean(self._ep_errors)), 4) if self._ep_errors else 0,
            round(self.current_speed, 3),
        ])
        self._csv_file.flush()

    def _log(self):
        elapsed = time.time() - self.start_time
        ts = self.num_timesteps
        total = self.locals.get('total_timesteps',
                                getattr(self.model, '_total_timesteps', 0))
        pct = 100 * ts / max(total, 1)
        fps = ts / max(elapsed, 1)
        mean_r = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0
        print(f"  [{pct:5.1f}%] Step {ts:>7,}/{total:,} | "
              f"Ep={self.episode_count} | R={mean_r:7.1f} | "
              f"Speed={self.current_speed:.2f} | {fps:.0f} fps")

    def _record(self):
        print(f"\n  Recording eval episode (chunk {self._chunk})...")
        self.video_recorder.start_episode(self._chunk)
        obs, info = self.env.reset()
        done = False
        step = 0
        total_r = 0
        while not done and step < 1000:
            obs_t = {k: v[np.newaxis, ...] for k, v in obs.items()}
            action, _ = self.model.predict(obs_t, deterministic=True)
            obs, r, term, trunc, info = self.env.step(action.squeeze())
            done = term or trunc
            total_r += r
            step += 1
            raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            fpv = raw._last_high_freq_image
            bird = None
            if hasattr(raw, '_bird_camera') and raw._bird_camera:
                ok, rgba = raw._bird_camera.get_image()
                if ok and rgba is not None:
                    bird = rgba[:, :, :3]
            t_info = info.get('target', {})
            overlay = {
                'Chunk': self._chunk, 'Step': f"{step}/1000",
                'Timestep': f"{self.num_timesteps:,}",
                'Reward': round(total_r, 1),
                'Distance': t_info.get('distance_to_target', 0),
                'target': t_info,
                'visual_tracking': info.get('visual_tracking', {}),
            }
            self.video_recorder.capture_frame(fpv, bird, info=overlay)
        self.video_recorder.end_episode()
        print(f"  Done: {step} steps, R={total_r:.1f}")
        self.env.reset()

    def save_metrics(self, extras=None):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        summary = {
            'total_episodes': self.episode_count,
            'final_mean_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'final_speed': self.current_speed,
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
        description="Train lemniscate follower (vision-only)")
    p.add_argument('--timesteps', type=int, default=1_000_000)
    p.add_argument('--scale', type=float, default=5.0,
                   help="Lemniscate half-width (m)")
    p.add_argument('--ideal-fraction', type=float, default=0.25,
                   help="Target fraction of image the sphere should occupy")
    p.add_argument('--fraction-tolerance', type=float, default=0.05,
                   help="±tolerance around ideal for positive reward")
    p.add_argument('--max-reward', type=float, default=1000.0,
                   help="Max reward per step at ideal fraction")
    p.add_argument('--initial-speed', type=float, default=0.05)
    p.add_argument('--max-speed', type=float, default=0.3)
    p.add_argument('--n-steps', type=int, default=2048)
    p.add_argument('--max-ep-steps', type=int, default=1000)
    p.add_argument('--output-dir', type=str,
                   default='./models/lemniscate_follower')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--record', action='store_true', default=False)
    p.add_argument('--record-interval', type=int, default=50)
    p.add_argument('--load-model', type=str, default=None)
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Training app
# ──────────────────────────────────────────────────────────────────────

class LemniscateTrainerApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D scene...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # FPV camera
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Bird camera (for recording)
        self.bird_camera = None
        if args.record:
            self.bird_camera = opencv_camera(self, 'bird_cam', 1)
            self.bird_camera.cam.reparentTo(self.render)
            self.bird_camera.cam.setPos(0, -12, 16)
            self.bird_camera.cam.lookAt(0, 0, 5)
            self.bird_camera.buffer.setActive(1)

        # Environment
        print("Creating environment...")
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='moving',
            target_range=3.0,
            target_speed=args.initial_speed,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            enable_collisions=False,
            n=args.max_ep_steps,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            filming_mode=True,
            lemniscate_scale=args.scale,
            ideal_fraction=args.ideal_fraction,
            fraction_tolerance=args.fraction_tolerance,
            max_visual_reward=args.max_reward,
            min_start_distance=3.0,
        )
        self.env = Monitor(self.env)
        if self.bird_camera:
            self.env.unwrapped._bird_camera = self.bird_camera
        self.vec_env = DummyVecEnv([lambda: self.env])

        # PPO
        print("Initializing PPO...")
        policy_kwargs = {
            'features_extractor_class': StateCameraExtractor,
            'features_extractor_kwargs': {
                'features_dim': 128,
                'camera_key': 'camera_high_freq',
            },
            'net_arch': dict(pi=[128, 64], vf=[128, 64]),
        }

        if args.load_model and os.path.exists(args.load_model):
            print(f"Resuming from {args.load_model}")
            self.model = PPO.load(args.load_model, env=self.vec_env,
                                  device='auto')
        else:
            self.model = PPO(
                'MultiInputPolicy', self.vec_env,
                policy_kwargs=policy_kwargs,
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

        self.video_recorder = None
        if args.record:
            rec_dir = self.output_dir / 'recordings'
            self.video_recorder = EpisodeRecorder(
                output_dir=str(rec_dir), fps=30, resolution=(640, 360))

        print("\n" + "=" * 70)
        print("  LEMNISCATE FOLLOWER — VISION-ONLY TRAINING")
        print("=" * 70)
        print(f"  Scale:          {args.scale}m (width={2*args.scale:.0f}m)")
        print(f"  Ideal fraction: {args.ideal_fraction*100:.0f}% ± {args.fraction_tolerance*100:.0f}%")
        print(f"  Max reward:     {args.max_reward}")
        print(f"  Speed:          {args.initial_speed} → {args.max_speed} (curriculum)")
        print(f"  Timesteps:      {args.timesteps:,}")
        print(f"  Episode steps:  {args.max_ep_steps}")
        print(f"  Policy params:  {total_params:,}")
        print(f"  Output:         {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        args = self.args

        render_cb = Panda3DRenderCallback(self)
        self.metrics_cb = LemniscateMetricsCallback(
            env=self.env,
            output_dir=str(self.output_dir),
            initial_speed=args.initial_speed,
            max_speed=args.max_speed,
            video_recorder=self.video_recorder,
            record_interval=args.record_interval,
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
            'lemniscate_scale': args.scale,
            'ideal_fraction': args.ideal_fraction,
            'fraction_tolerance': args.fraction_tolerance,
            'max_visual_reward': args.max_reward,
            'initial_speed': args.initial_speed,
            'max_speed': args.max_speed,
            'policy_params': sum(
                p.numel() for p in self.model.policy.parameters()),
        })

        if self.video_recorder and self.video_recorder.episode_files:
            self.video_recorder.compile_timelapse(
                "training_timelapse.mp4", max_frames_per_ep=150)

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
        loadPrcFileData('', 'win-size 320 240')
        loadPrcFileData('', 'undecorated true')
    app = LemniscateTrainerApp(args)
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