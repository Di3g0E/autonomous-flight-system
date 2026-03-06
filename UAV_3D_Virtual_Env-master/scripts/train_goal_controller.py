#!/usr/bin/env python
"""
Train a goal-conditioned PPO controller using vision-based target following.

The drone learns to navigate toward a visible target marker (orange sphere)
using only its FPV camera and state vector (no target position in obs).

IMPORTANT: This script launches a full Panda3D application with the 3D city 
scene so the camera captures REAL rendered images. Training runs inside the 
Panda3D task loop with n_envs=1.

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
from direct.task import Task

# Load Panda3D config
loadPrcFile(os.path.join(project_root, 'config', 'conf.prc'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.agents.feature_extractors import StateCameraExtractor
from src.utils.episode_recorder import EpisodeRecorder

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Train goal-conditioned PPO controller")
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps')
    parser.add_argument('--target-mode', type=str, default='fixed',
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
    # Recording options
    parser.add_argument('--record', action='store_true', default=False,
                        help='Record evaluation episodes as video')
    parser.add_argument('--record-interval', type=int, default=50,
                        help='Record an evaluation episode every N training chunks')
    parser.add_argument('--record-fps', type=int, default=30,
                        help='Video FPS for recordings')
    return parser.parse_args()


class GoalTrainerApp(ShowBase):
    """
    Panda3D application that integrates RL training with live rendering.
    
    The training loop runs inside Panda3D's task system so that camera
    images are real rendered frames (not black placeholders).
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
        
        # Create bird's-eye camera for recording (separate offscreen buffer)
        self.bird_camera = None
        if args.record:
            print("Setting up bird's-eye camera for recording...")
            self.bird_camera = opencv_camera(self, 'bird_cam', 1)
            # Position above and behind the drone area, looking down
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
            camera_high_freq_size=(64, 64),
            camera_low_freq_size=(64, 64),
            enable_collisions=False,
            n=1000,
            t_step=0.01,
            direct_control=1
        )
        
        # Wrap in DummyVecEnv for SB3 (n_envs=1)
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Create PPO model
        print("Initializing PPO with StateCameraExtractor...")
        policy_kwargs = {
            'features_extractor_class': StateCameraExtractor,
            'features_extractor_kwargs': {
                'features_dim': 128,
                'camera_key': 'camera_high_freq'
            },
            'net_arch': dict(pi=[128, 64], vf=[128, 64])
        }
        
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
            verbose=0,
            seed=args.seed
        )
        
        total_params = sum(p.numel() for p in self.model.policy.parameters())
        
        # Training state
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.total_timesteps = args.timesteps
        self.current_timestep = 0
        self.chunk_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_distances = []
        self.episode_arrivals = []
        self.metrics_history = []
        
        # Curriculum
        self.curriculum = args.curriculum
        self.current_range = initial_range
        self.max_range = args.target_range
        
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # Video recorder
        self.video_recorder = None
        if args.record:
            rec_dir = self.output_dir / 'recordings'
            self.video_recorder = EpisodeRecorder(
                output_dir=str(rec_dir),
                fps=args.record_fps,
                resolution=(640, 360)
            )
            print(f"Recording enabled: 1 episode every {args.record_interval} chunks → {rec_dir}")
        
        # Print config
        print("\n" + "=" * 70)
        print("GOAL-CONDITIONED PPO TRAINING (Panda3D Live Rendering)")
        print("=" * 70)
        print(f"  Target mode:    {args.target_mode}")
        print(f"  Target range:   {initial_range}m → {args.target_range}m (curriculum)")
        print(f"  Timesteps:      {args.timesteps:,}")
        print(f"  N_steps/rollout:{args.n_steps}")
        print(f"  Policy params:  {total_params:,}")
        print(f"  Recording:      {'ON (interval={})'.format(args.record_interval) if args.record else 'OFF'}")
        print(f"  Output:         {args.output_dir}")
        print("=" * 70)
        print("\nStarting training...\n")
        
        # Start training loop
        self.taskMgr.add(self.training_task, 'training_task')
    
    def training_task(self, task):
        """Main training loop running inside Panda3D's task system."""
        
        if self.current_timestep >= self.total_timesteps:
            self._finalize()
            return Task.done
        
        # Record an evaluation episode before training chunk (if recording)
        if (self.video_recorder and 
            self.chunk_count % self.args.record_interval == 0):
            self._record_eval_episode()
        
        # Train one chunk
        chunk_steps = min(self.args.n_steps, self.total_timesteps - self.current_timestep)
        
        self.graphicsEngine.renderFrame()
        
        self.model.learn(
            total_timesteps=chunk_steps,
            reset_num_timesteps=False if self.current_timestep > 0 else True,
            progress_bar=False
        )
        
        self.current_timestep += chunk_steps
        self.chunk_count += 1
        
        # Extract episode metrics from SB3
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])
                    self.episode_count += 1
        
        # Target metrics
        if hasattr(self.env, '_goal_reward'):
            info = self.env._goal_reward()
            if info:
                self.episode_distances.append(info['distance_to_target'])
                self.episode_arrivals.append(1.0 if info['arrived'] else 0.0)
        
        # Log progress
        now = time.time()
        if now - self.last_log_time > 5.0:
            self._log_progress()
            self.last_log_time = now
        
        return Task.cont
    
    def _record_eval_episode(self):
        """Run one evaluation episode with recording enabled."""
        print(f"\n  📹 Recording evaluation episode (chunk {self.chunk_count})...")
        
        self.video_recorder.start_episode(self.chunk_count)
        
        obs, info = self.env.reset()
        target_info = info.get('target', {})
        
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < 1000:
            # Get action from current policy (deterministic for evaluation)
            obs_tensor = {}
            for key, val in obs.items():
                obs_tensor[key] = val[np.newaxis, ...]  # Add batch dim
            
            action, _ = self.model.predict(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action.squeeze())
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            target_info = info.get('target', {})
            
            # Capture frames
            fpv_img = self.env._last_high_freq_image  # Already captured in step()
            
            # Capture bird's-eye view
            bird_img = None
            if self.bird_camera is not None:
                success, bird_rgba = self.bird_camera.get_image()
                if success and bird_rgba is not None:
                    bird_img = bird_rgba[:, :, :3]  # RGBA → RGB
            
            # Build info overlay
            overlay = {
                'Chunk': self.chunk_count,
                'Step': f"{step}/1000",
                'Timestep': f"{self.current_timestep:,}",
                'Reward': total_reward,
                'Distance': target_info.get('distance_to_target', 0),
                'Range': f"{self.current_range:.1f}m"
            }
            
            self.video_recorder.capture_frame(fpv_img, bird_img, info=overlay)
        
        self.video_recorder.end_episode()
        
        final_dist = target_info.get('distance_to_target', 0)
        arrived = target_info.get('arrived', False)
        print(f"  📹 Done: {step} steps, R={total_reward:.1f}, "
              f"dist={final_dist:.2f}m, arrived={'✓' if arrived else '✗'}")
        
        # Reset env for the next training chunk
        self.env.reset()
    
    def _log_progress(self):
        """Log training metrics."""
        elapsed = time.time() - self.start_time
        pct = 100.0 * self.current_timestep / self.total_timesteps
        fps = self.current_timestep / max(elapsed, 1)
        
        recent_n = min(50, max(1, len(self.episode_rewards)))
        mean_r = float(np.mean(self.episode_rewards[-recent_n:])) if self.episode_rewards else 0
        
        recent_dist_n = min(50, max(1, len(self.episode_distances)))
        mean_dist = float(np.mean(self.episode_distances[-recent_dist_n:])) if self.episode_distances else 0
        mean_arrival = float(np.mean(self.episode_arrivals[-recent_dist_n:])) if self.episode_arrivals else 0
        
        metrics = {
            'timestep': self.current_timestep,
            'episodes': self.episode_count,
            'mean_reward': mean_r,
            'mean_distance': mean_dist,
            'arrival_rate': mean_arrival,
            'target_range': self.current_range,
            'fps': fps
        }
        self.metrics_history.append(metrics)
        
        print(f"  [{pct:5.1f}%] Step {self.current_timestep:>7,}/{self.total_timesteps:,} | "
              f"Ep={self.episode_count} | R={mean_r:7.1f} | "
              f"Dist={mean_dist:.2f}m | Arr={mean_arrival:.0%} | "
              f"Range={self.current_range:.1f}m | {fps:.0f} fps")
        
        # Curriculum
        if self.curriculum and mean_arrival > 0.5 and self.current_range < self.max_range:
            self.current_range = min(self.current_range + 0.5, self.max_range)
            self.env.target_range = self.current_range
            print(f"  → Curriculum: target range → {self.current_range:.1f}m")
    
    def _finalize(self):
        """Save model, metrics, and compile recording timelapse."""
        elapsed = time.time() - self.start_time
        
        # Save model
        model_path = self.output_dir / 'best_model'
        self.model.save(str(model_path))
        
        # Save metrics
        metrics_path = self.output_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Save summary
        summary = {
            'total_timesteps': self.current_timestep,
            'total_episodes': self.episode_count,
            'training_time_seconds': elapsed,
            'final_mean_reward': float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0,
            'final_mean_distance': float(np.mean(self.episode_distances[-50:])) if self.episode_distances else 0,
            'final_arrival_rate': float(np.mean(self.episode_arrivals[-50:])) if self.episode_arrivals else 0,
            'final_target_range': self.current_range,
            'target_mode': self.args.target_mode,
            'policy_params': sum(p.numel() for p in self.model.policy.parameters()),
            'recorded_episodes': len(self.video_recorder.episode_files) if self.video_recorder else 0
        }
        
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Compile timelapse video
        if self.video_recorder and self.video_recorder.episode_files:
            print("\nCompiling training timelapse...")
            timelapse_path = self.video_recorder.compile_timelapse(
                output_name="training_timelapse.mp4",
                max_frames_per_ep=150
            )
            if timelapse_path:
                print(f"  Timelapse: {timelapse_path}")
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Time:          {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"  Timesteps:     {self.current_timestep:,}")
        print(f"  Episodes:      {self.episode_count}")
        print(f"  Final reward:  {summary['final_mean_reward']:.1f}")
        print(f"  Final dist:    {summary['final_mean_distance']:.2f}m")
        print(f"  Arrival rate:  {summary['final_arrival_rate']:.0%}")
        print(f"  Model:         {model_path}.zip")
        if self.video_recorder:
            print(f"  Videos:        {self.output_dir / 'recordings'}")
        print("=" * 70)
        
        self.vec_env.close()
        self.userExit()


def main():
    args = parse_args()
    app = GoalTrainerApp(args)
    try:
        app.run()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            emergency_path = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(emergency_path))
            print(f"Model saved to {emergency_path}.zip")
        if hasattr(app, 'video_recorder') and app.video_recorder:
            print("Compiling partial timelapse...")
            app.video_recorder.compile_timelapse("partial_timelapse.mp4")


if __name__ == "__main__":
    main()
