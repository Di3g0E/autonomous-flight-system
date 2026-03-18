#!/usr/bin/env python
"""
Train a goal-conditioned PPO controller using vision-based target following.

The drone learns to navigate toward a visible target marker (orange sphere)
using only its FPV camera and state vector (no target position in obs).

IMPORTANT: This script launches a full Panda3D application with the 3D city 
scene so the camera captures REAL rendered images. A custom SB3 callback
advances Panda3D's task manager on every training step to keep the renderer
in sync.

Video recording:
    Use --record to periodically record evaluation episodes showing the 
    drone's progress. At the end, a timelapse video is compiled.

Usage:
    python scripts/train_goal_controller.py --timesteps 500000 --target-mode fixed
    python scripts/train_goal_controller.py --timesteps 500000 --record --record-interval 50
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# IMPORTANT: Import torch BEFORE Panda3D to avoid DLL conflicts on Windows
import torch

# Panda3D imports
from panda3d.core import Filename, loadPrcFile, loadPrcFileData
from direct.showbase.ShowBase import ShowBase

# Load Panda3D config
prc_path = os.path.join(project_root, 'config', 'conf.prc')
loadPrcFile(Filename.fromOsSpecific(prc_path))

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

import cv2


# ──────────────────────────────────────────────────────────────────────
# Custom SB3 Callbacks
# ──────────────────────────────────────────────────────────────────────

class Panda3DRenderCallback(BaseCallback):
    """
    Advance Panda3D's task manager on every SB3 training step.
    
    This inverts the control: instead of Panda3D calling model.learn()
    inside its task loop, SB3 drives the main loop and this callback
    keeps the 3D renderer in sync.
    """

    def __init__(self, app, verbose=0):
        super().__init__(verbose)
        self.app = app

    def _on_step(self):
        # Advance one Panda3D task-manager tick (processes window events,
        # keeps the 3D window responsive, etc.)
        self.app.taskMgr.step()
        return True


class GoalMetricsCallback(BaseCallback):
    """
    Track goal-specific metrics during training.

    Logs episode rewards, distances to target, visual tracking quality,
    and handles curriculum learning. Writes per-episode CSV for TFG analysis.
    """

    CSV_HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'mean_distance', 'min_distance', 'final_distance', 'std_distance',
        'mean_centering', 'mean_scale',
        'visibility_pct', 'proximity_violations',
        'mean_target_fraction', 'max_target_fraction',
    ]

    def __init__(self, env, output_dir, curriculum=True,
                 initial_range=1.0, max_range=3.0,
                 initial_speed=0.05, max_speed=0.3,
                 video_recorder=None, record_interval=50,
                 metrics_window=50, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage (fixed-size ring buffers to avoid RAM growth)
        self.episode_count = 0
        from collections import deque
        self.episode_rewards = deque(maxlen=metrics_window)
        self.episode_distances = deque(maxlen=metrics_window)

        # Per-episode CSV
        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        # Per-episode accumulators (reset after each episode)
        self._reset_episode_accumulators()

        # Curriculum
        self.curriculum = curriculum
        self.current_range = initial_range
        self.max_range = max_range

        # Speed curriculum
        self.initial_speed = initial_speed
        self.max_speed = max_speed
        self.current_speed = initial_speed

        # Recording
        self.video_recorder = video_recorder
        self.record_interval = record_interval
        self._chunk_counter = 0  # incremented on each rollout

        self.start_time = time.time()
        self.last_log_time = time.time()

    def _reset_episode_accumulators(self):
        """Reset per-episode metric accumulators."""
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_distances = []
        self._ep_centering_scores = []
        self._ep_scale_scores = []
        self._ep_visible_steps = 0
        self._ep_proximity_violations = 0
        self._ep_target_fractions = []

    # ── called once at training start ──
    def _on_training_start(self):
        self._chunk_counter = 0
        # Open CSV file
        import csv
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADERS)

    # ── called at the end of each rollout (n_steps block) ──
    def _on_rollout_end(self):
        self._chunk_counter += 1

        # Collect episode rewards from SB3's internal buffer (for console logging)
        if hasattr(self.model, 'ep_info_buffer'):
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])

        # Speed curriculum: increase target speed linearly with training progress
        raw_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        if self.curriculum:
            total = self.locals.get(
                'total_timesteps',
                self.model._total_timesteps if hasattr(self.model, '_total_timesteps') else 1
            )
            progress = min(self.num_timesteps / max(total, 1), 1.0)
            self.current_speed = self.initial_speed + progress * (self.max_speed - self.initial_speed)
            raw_env.target_speed = self.current_speed

        # Target metrics snapshot for console logging
        if hasattr(raw_env, '_goal_reward'):
            info = raw_env._goal_reward()
            if info:
                self.episode_distances.append(info['distance_to_target'])

        # Log every 5 seconds
        now = time.time()
        if now - self.last_log_time > 5.0:
            self._log_progress()
            self.last_log_time = now

        # Periodic recording
        if (self.video_recorder and
                self._chunk_counter % self.record_interval == 0):
            self._record_eval_episode()

    # ── per-step hook: collect per-step metrics for CSV ──
    def _on_step(self):
        # Access info from the latest step (SB3 VecEnv format)
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones = self.locals.get('dones', [False])

        if infos:
            info = infos[0]
            self._ep_reward += float(rewards[0])
            self._ep_steps += 1

            # Target distance
            target_info = info.get('target', {})
            if 'distance_to_target' in target_info:
                self._ep_distances.append(target_info['distance_to_target'])

            # Visual tracking quality
            vis_info = info.get('visual_tracking', {})
            if vis_info:
                if vis_info.get('target_visible', False):
                    self._ep_visible_steps += 1
                    if 'centering_reward' in vis_info:
                        self._ep_centering_scores.append(vis_info['centering_reward'])
                    if 'scale_reward' in vis_info:
                        self._ep_scale_scores.append(vis_info['scale_reward'])
                    frac = vis_info.get('target_fraction', 0.0)
                    self._ep_target_fractions.append(frac)
                    if frac > 0.20:
                        self._ep_proximity_violations += 1

            # Episode ended → write CSV row
            if dones[0]:
                self._write_episode_csv()
                self._reset_episode_accumulators()

        return True

    def _write_episode_csv(self):
        """Write one row to training_log.csv with this episode's metrics."""
        if self._csv_writer is None:
            return

        self.episode_count += 1
        d = self._ep_distances
        row = [
            self.episode_count,
            self.num_timesteps,
            round(self._ep_reward, 2),
            self._ep_steps,
            # Distance metrics
            round(float(np.mean(d)), 3) if d else 0.0,
            round(float(np.min(d)), 3) if d else 0.0,
            round(float(d[-1]), 3) if d else 0.0,
            round(float(np.std(d)), 3) if d else 0.0,
            # Visual quality
            round(float(np.mean(self._ep_centering_scores)), 3) if self._ep_centering_scores else 0.0,
            round(float(np.mean(self._ep_scale_scores)), 3) if self._ep_scale_scores else 0.0,
            # Safety
            round(100.0 * self._ep_visible_steps / max(self._ep_steps, 1), 1),
            self._ep_proximity_violations,
            # Target fraction
            round(float(np.mean(self._ep_target_fractions)), 4) if self._ep_target_fractions else 0.0,
            round(float(np.max(self._ep_target_fractions)), 4) if self._ep_target_fractions else 0.0,
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    # ── helpers ──
    def _log_progress(self):
        elapsed = time.time() - self.start_time
        ts = self.num_timesteps
        total = self.locals.get('total_timesteps',
                                self.model._total_timesteps if hasattr(self.model, '_total_timesteps') else 0)
        pct = 100.0 * ts / max(total, 1)
        fps = ts / max(elapsed, 1)

        mean_r = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0
        mean_dist = float(np.mean(self.episode_distances)) if self.episode_distances else 0

        print(
            f"  [{pct:5.1f}%] Step {ts:>7,}/{total:,} | "
            f"Ep={self.episode_count} | R={mean_r:7.1f} | "
            f"Dist={mean_dist:.2f}m | "
            f"Range={self.current_range:.1f}m | Speed={self.current_speed:.2f} | {fps:.0f} fps"
        )

    def _record_eval_episode(self):
        """Run one evaluation episode with recording enabled."""
        print(f"\n  📹 Recording evaluation episode (chunk {self._chunk_counter})...")

        self.video_recorder.start_episode(self._chunk_counter)

        obs, info = self.env.reset()
        target_info = info.get('target', {})

        done = False
        total_reward = 0
        step = 0

        while not done and step < 1000:
            obs_tensor = {}
            for key, val in obs.items():
                obs_tensor[key] = val[np.newaxis, ...]
            action, _ = self.model.predict(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action.squeeze())
            done = terminated or truncated
            total_reward += reward
            step += 1

            target_info = info.get('target', {})

            # Capture frames (access unwrapped env through Monitor)
            raw_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            fpv_img = raw_env._last_high_freq_image
            bird_img = None
            if hasattr(raw_env, '_bird_camera') and raw_env._bird_camera is not None:
                success, bird_rgba = raw_env._bird_camera.get_image()
                if success and bird_rgba is not None:
                    bird_img = bird_rgba[:, :, :3]

            # Build overlay: top-level keys for display + sub-dicts for detailed overlays
            overlay = {
                'Chunk': self._chunk_counter,
                'Step': f"{step}/1000",
                'Timestep': f"{self.num_timesteps:,}",
                'Reward': round(total_reward, 1),
                'Distance': target_info.get('distance_to_target', 0),
                'target': target_info,
                'visual_tracking': info.get('visual_tracking', {}),
            }
            self.video_recorder.capture_frame(fpv_img, bird_img, info=overlay)

        self.video_recorder.end_episode()

        final_dist = target_info.get('distance_to_target', 0)
        arrived = target_info.get('arrived', False)
        print(
            f"  📹 Done: {step} steps, R={total_reward:.1f}, "
            f"dist={final_dist:.2f}m, arrived={'✓' if arrived else '✗'}"
        )
        # Reset env for the next training chunk
        self.env.reset()

    def save_metrics(self, summary_extras=None):
        """Save metrics history, final summary, and close CSV."""
        # Close CSV file
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        # Save summary (all per-episode data is in training_log.csv)
        metrics_path = self.output_dir / 'training_summary.json'
        summary = {
            'total_episodes': self.episode_count,
            'final_mean_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'final_mean_distance': float(np.mean(self.episode_distances)) if self.episode_distances else 0,
            'final_target_range': self.current_range,
            'final_target_speed': self.current_speed,
            'csv_path': str(self.csv_path),
        }
        if summary_extras:
            summary.update(summary_extras)

        with open(metrics_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return metrics_path


# ──────────────────────────────────────────────────────────────────────
# Main Panda3D training application
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train goal-conditioned PPO controller")
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps')
    parser.add_argument('--target-mode', type=str, default='moving',
                        choices=['fixed', 'waypoints', 'moving'],
                        help='Target mode')
    parser.add_argument('--target-range', type=float, default=3.0,
                        help='Max target distance from origin')
    parser.add_argument('--output-dir', type=str, default='./models/goal_controller',
                        help='Output directory for models')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='PPO rollout buffer size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--curriculum', action='store_true', default=True,
                        help='Use curriculum learning')
    parser.add_argument('--initial-target-speed', type=float, default=0.05,
                        help='Initial target speed for curriculum (m/s)')
    parser.add_argument('--max-target-speed', type=float, default=0.3,
                        help='Maximum target speed for curriculum (m/s)')
    parser.add_argument('--metrics-window', type=int, default=50,
                        help='Max episodes kept in RAM for rolling stats (default: 50)')
    # Recording options
    parser.add_argument('--record', action='store_true', default=False,
                        help='Record evaluation episodes as video')
    parser.add_argument('--record-interval', type=int, default=50,
                        help='Record an evaluation episode every N rollouts')
    parser.add_argument('--record-fps', type=int, default=30,
                        help='Video FPS for recordings')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to a saved model .zip to resume training')
    # Filming mode
    parser.add_argument('--filming-mode', action='store_true', default=True,
                        help='Enable follow/film mode (navigation by vision only)')
    parser.add_argument('--no-filming-mode', dest='filming_mode', action='store_false',
                        help='Disable filming mode (use reach mode)')
    return parser.parse_args()


class GoalTrainerApp(ShowBase):
    """
    Panda3D application that integrates RL training with live rendering.
    
    Architecture change vs. previous version:
    - Previously: Panda3D task loop called model.learn(chunk) → broken episode tracking.
    - Now: model.learn() is the main driver; a Panda3DRenderCallback calls
      taskMgr.step() on every SB3 step → correct ep_info_buffer.
    """

    def __init__(self, args):
        ShowBase.__init__(self)

        self.args = args
        self.mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)
        ).getFullpath()

        # Load 3D world and drone model
        print("Loading 3D scene...")
        world_setup(self, self.render, self.mydir)
        quad_setup(self, self.render, self.mydir)
        camera_control(self, self.render)

        # Create FPV camera attached to drone
        print("Setting up FPV camera...")
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Bird's-eye camera for recording
        self.bird_camera = None
        if args.record:
            print("Setting up bird's-eye camera for recording...")
            self.bird_camera = opencv_camera(self, 'bird_cam', 1)
            self.bird_camera.cam.reparentTo(self.render)
            self.bird_camera.cam.setPos(0, -8, 12)
            self.bird_camera.cam.lookAt(0, 0, 5)
            self.bird_camera.buffer.setActive(1)

        # Create environment with REAL Panda3D integration
        print("Creating goal-conditioned environment...")
        initial_range = 1.0 if args.curriculum else args.target_range

        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode=args.target_mode,
            target_range=initial_range,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            enable_collisions=False,
            n=1000,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            target_speed=args.initial_target_speed,
            filming_mode=args.filming_mode,
        )
        
        # Wrap environment with Monitor to track episode metrics
        self.env = Monitor(self.env)
        
        # Store bird camera on unwrapped env (recording callback accesses via .unwrapped)
        self.env.unwrapped._bird_camera = self.bird_camera

        # Wrap in DummyVecEnv for SB3 (n_envs=1)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # Create PPO model
        print("Initializing PPO with StateCameraExtractor...")
        policy_kwargs = {
            'features_extractor_class': StateCameraExtractor,
            'features_extractor_kwargs': {
                'features_dim': 128,
                'camera_key': 'camera_high_freq',
            },
            'net_arch': dict(pi=[128, 64], vf=[128, 64]),
        }

        if args.load_model and os.path.exists(args.load_model):
            print(f"Loading existing model from {args.load_model}...")
            self.model = PPO.load(
                args.load_model,
                env=self.vec_env,
                device='auto'
            )
        else:
            self.model = PPO(
                'MultiInputPolicy',
                self.vec_env,
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

        # Output dir
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Video recorder
        self.video_recorder = None
        if args.record:
            rec_dir = self.output_dir / 'recordings'
            self.video_recorder = EpisodeRecorder(
                output_dir=str(rec_dir),
                fps=args.record_fps,
                resolution=(640, 360),
            )
            print(f"Recording enabled: 1 episode every {args.record_interval} rollouts → {rec_dir}")

        # Print config
        mode_label = "FOLLOW/FILM MODE" if args.filming_mode else "REACH MODE"
        print("\n" + "=" * 70)
        print(f"GOAL-CONDITIONED PPO TRAINING — {mode_label}")
        print("=" * 70)
        print(f"  Filming mode:   {args.filming_mode}")
        print(f"  Target mode:    {args.target_mode}")
        print(f"  Target range:   {initial_range}m → {args.target_range}m (curriculum)")
        print(f"  Target speed:   {args.initial_target_speed} → {args.max_target_speed} (curriculum)")
        print(f"  Camera:         32×32 FPV")
        print(f"  Timesteps:      {args.timesteps:,}")
        print(f"  N_steps/rollout:{args.n_steps}")
        print(f"  Policy params:  {total_params:,}")
        print(f"  Recording:      {'ON (interval={})'.format(args.record_interval) if args.record else 'OFF'}")
        print(f"  Output:         {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        """
        Main entry point — drives training from model.learn().
        
        model.learn() is the main loop; two callbacks keep Panda3D
        in sync and collect metrics.
        """
        args = self.args

        # Callbacks
        render_cb = Panda3DRenderCallback(self)
        self.metrics_cb = GoalMetricsCallback(
            env=self.env,
            output_dir=str(self.output_dir),
            curriculum=args.curriculum,
            initial_range=1.0 if args.curriculum else args.target_range,
            max_range=args.target_range,
            initial_speed=args.initial_target_speed,
            max_speed=args.max_target_speed,
            video_recorder=self.video_recorder,
            record_interval=args.record_interval,
            metrics_window=args.metrics_window,
        )

        print("\nStarting training...\n")
        start_time = time.time()

        self.model.learn(
            total_timesteps=args.timesteps,
            callback=[render_cb, self.metrics_cb],
            progress_bar=True,
        )

        elapsed = time.time() - start_time

        # ── Finalize ──
        model_path = self.output_dir / 'best_model'
        self.model.save(str(model_path))

        self.metrics_cb.save_metrics(summary_extras={
            'total_timesteps': args.timesteps,
            'training_time_seconds': elapsed,
            'target_mode': args.target_mode,
            'filming_mode': args.filming_mode,
            'initial_target_speed': args.initial_target_speed,
            'max_target_speed': args.max_target_speed,
            'final_target_speed': self.metrics_cb.current_speed,
            'policy_params': sum(p.numel() for p in self.model.policy.parameters()),
            'recorded_episodes': len(self.video_recorder.episode_files) if self.video_recorder else 0,
        })

        # Compile timelapse
        if self.video_recorder and self.video_recorder.episode_files:
            print("\nCompiling training timelapse...")
            tl = self.video_recorder.compile_timelapse(
                output_name="training_timelapse.mp4",
                max_frames_per_ep=150,
            )
            if tl:
                print(f"  Timelapse: {tl}")

        # Generate TFG training plots from CSV
        csv_path = self.metrics_cb.csv_path
        print("\nGenerating training plots for TFG...")
        try:
            from scripts.generate_training_plots import generate_all_plots
            plots_dir = self.output_dir / 'plots'
            generate_all_plots(csv_path, plots_dir)
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Time:          {elapsed:.0f}s ({elapsed / 60:.1f} min)")
        print(f"  Timesteps:     {args.timesteps:,}")
        print(f"  Episodes:      {self.metrics_cb.episode_count}")
        print(f"  Model:         {model_path}.zip")
        print("=" * 70)

        self.vec_env.close()


def main():
    args = parse_args()
    app = GoalTrainerApp(args)
    try:
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            emergency_path = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(emergency_path))
            print(f"Model saved to {emergency_path}.zip")
        if hasattr(app, 'video_recorder') and app.video_recorder:
            print("Compiling partial timelapse...")
            app.video_recorder.compile_timelapse("partial_timelapse.mp4")
        
        if hasattr(app, 'metrics_cb'):
            print("Saving training metrics...")
            app.metrics_cb.save_metrics()
            
            # Generate TFG plots on interrupt
            print("Generating training plots for TFG...")
            try:
                from scripts.generate_training_plots import generate_all_plots
                csv_path = app.metrics_cb.csv_path
                plots_dir = Path(args.output_dir) / 'plots'
                generate_all_plots(csv_path, plots_dir)
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    main()
