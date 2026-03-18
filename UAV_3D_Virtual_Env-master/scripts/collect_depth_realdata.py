#!/usr/bin/env python
"""
Collect REAL RGB-Depth dataset using Panda3D live rendering.

This script launches the full Panda3D simulation with the city scene,
flies the drone through the environment, and captures paired RGB-Depth
data from the depth buffer.

Unlike collect_depth_dataset.py (headless mode), this script produces
REAL depth maps with actual obstacle distances.

Usage:
    python scripts/collect_depth_realdata.py --num-samples 5000 --output-dir ./data/depth_real
"""

import sys
import os
import argparse
import json
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
from panda3d.core import Filename, loadPrcFile, LPoint3f
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

# Load Panda3D config
loadPrcFile(os.path.join(project_root, 'config', 'conf.prc'))

# Project imports
from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.quadrotor_env import quad
from src.dataset.depth_dataset_collector import DepthDatasetCollector

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect real RGB-Depth dataset from Panda3D simulation"
    )
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of RGB-Depth pairs to collect')
    parser.add_argument('--output-dir', type=str, default='./data/depth_real',
                        help='Output directory for dataset')
    parser.add_argument('--image-size', type=int, nargs=2, default=[32, 32],
                        help='Output image size (width height)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--depth-metric', action='store_true',
                        help='Use metric depth (meters) instead of normalized [0,1]')
    parser.add_argument('--random-actions', action='store_true', default=True,
                        help='Use random actions for diverse viewpoints')
    parser.add_argument('--episode-steps', type=int, default=200,
                        help='Steps per episode before reset')
    parser.add_argument('--capture-interval', type=int, default=5,
                        help='Capture every N physics steps (higher = more diverse)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


class DepthCollectorApp(ShowBase):
    """
    Panda3D application for collecting real RGB-Depth datasets.
    
    Launches the 3D city scene, attaches a depth-enabled camera to the drone,
    and captures paired RGB-Depth frames as the drone executes random actions.
    """
    
    def __init__(self, args):
        ShowBase.__init__(self)
        
        self.args = args
        self.mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)
        ).getFullpath()
        
        # Load 3D world and drone model
        render = self.render
        world_setup(self, render, self.mydir)
        quad_setup(self, render, self.mydir)
        
        # Camera control for main viewport
        camera_control(self, self.render)
        
        # Create depth-enabled camera attached to drone
        self.depth_cam = opencv_camera(self, 'depth_cam', 1)  # frame_interval=1
        # Parent camera to drone model so it moves with the drone
        self.depth_cam.cam.reparentTo(self.quad_model)
        # Position camera at front of drone, looking forward
        self.depth_cam.cam.setPos(0, 0.3, -0.05)
        self.depth_cam.cam.setHpr(0, 0, 0)
        # Ensure buffer is always active for collection
        self.depth_cam.buffer.setActive(1)
        
        # Create physics environment
        np.random.seed(args.seed)
        self.physics_env = quad(
            t_step=0.01,
            n=args.episode_steps,
            euler=0,
            direct_control=1,
            T=1
        )
        self.physics_env.reset(seed=args.seed)
        
        # Dataset collector
        self.collector = DepthDatasetCollector(
            save_dir=args.output_dir,
            validation_split=args.val_split,
            test_split=args.test_split,
            max_samples_per_file=1000,
            camera_type='high_freq'
        )
        
        # Collection state
        self.target_samples = args.num_samples
        self.samples_collected = 0
        self.step_count = 0
        self.episode_count = 0
        self.capture_interval = args.capture_interval
        self.image_size = tuple(args.image_size)
        
        # Stats
        self.start_time = time.time()
        self.last_print_time = time.time()
        
        # Print configuration
        print("\n" + "=" * 70)
        print("REAL DEPTH DATA COLLECTION")
        print("=" * 70)
        print(f"Target samples: {self.target_samples}")
        print(f"Image size: {self.image_size}")
        print(f"Capture interval: every {self.capture_interval} steps")
        print(f"Depth format: {'metric (meters)' if args.depth_metric else 'normalized [0,1]'}")
        print(f"Output: {args.output_dir}")
        print("=" * 70)
        print("\nStarting collection... (close window or Ctrl+C to stop early)\n")
        
        # Start the collection task
        self.taskMgr.add(self.collection_task, 'collection_task')
    
    def collection_task(self, task):
        """Main collection loop running as a Panda3D task."""
        
        # Check if we're done
        if self.samples_collected >= self.target_samples:
            self._finalize()
            return Task.done
        
        # Execute physics step
        if self.args.random_actions:
            action = self.physics_env.action_space.sample()
        else:
            action = np.array([0.25, 0.25, 0.25, 0.25])  # hover
        
        obs, reward, terminated, truncated, info = self.physics_env.step(action)
        self.step_count += 1
        
        # Update drone 3D model position from physics
        state = obs  # quadrotor_env returns state directly as obs
        pos = state[0:3]   # x, y, z
        self.quad_model.setPos(pos[0], pos[1], pos[2])
        
        # Use quaternion for rotation if available
        if len(state) >= 10:
            # state[6:10] = quaternion (q0, q1, q2, q3)
            quat = state[6:10]
            from panda3d.core import Quat as PandaQuat
            pq = PandaQuat(quat[0], quat[1], quat[2], quat[3])
            self.quad_model.setQuat(pq)
        
        # Ensure Panda3D renders the scene
        self.graphicsEngine.renderFrame()
        
        # Capture at the configured interval
        if self.step_count % self.capture_interval == 0:
            self._capture_and_store(obs, action, reward)
        
        # Reset episode if needed
        if terminated or truncated:
            self.physics_env.reset()
            self.episode_count += 1
        
        # Print progress periodically
        now = time.time()
        if now - self.last_print_time > 2.0:  # Every 2 seconds
            elapsed = now - self.start_time
            rate = self.samples_collected / max(elapsed, 0.001)
            remaining = (self.target_samples - self.samples_collected) / max(rate, 0.001)
            pct = 100.0 * self.samples_collected / self.target_samples
            print(f"  [{pct:5.1f}%] Samples: {self.samples_collected}/{self.target_samples} "
                  f"| Episodes: {self.episode_count} "
                  f"| Rate: {rate:.1f} samples/s "
                  f"| ETA: {remaining:.0f}s")
            self.last_print_time = now
        
        return Task.cont
    
    def _capture_and_store(self, state, action, reward):
        """Capture RGB + Depth from the camera and store in dataset."""
        # Capture RGB
        success_rgb, img_rgba = self.depth_cam.get_image()
        if not success_rgb or img_rgba is None:
            return
        
        # Convert RGBA to RGB and resize
        img_rgb = img_rgba[:, :, :3]
        img_rgb = cv2.resize(img_rgb, self.image_size, interpolation=cv2.INTER_AREA)
        img_rgb = img_rgb.astype(np.uint8)
        
        # Capture Depth
        success_depth, depth = self.depth_cam.get_depth(
            normalize=not self.args.depth_metric,
            metric=self.args.depth_metric
        )
        if not success_depth or depth is None:
            return
        
        # Resize depth to match target size
        depth_2d = depth[:, :, 0]  # Remove channel dim for resize
        depth_resized = cv2.resize(depth_2d, self.image_size, interpolation=cv2.INTER_NEAREST)
        depth_resized = np.expand_dims(depth_resized, axis=-1).astype(np.float32)
        
        # Validate: check depth is not all zeros
        if np.max(depth_resized) == 0 and np.min(depth_resized) == 0:
            # Skip empty frames (can happen during initialization)
            return
        
        # Store sample
        metadata = {
            'episode': self.episode_count,
            'step': self.step_count,
            'action': action.tolist() if hasattr(action, 'tolist') else list(action),
            'reward': float(reward),
            'state': state.tolist() if hasattr(state, 'tolist') else list(state),
            'depth_min': float(np.min(depth_resized)),
            'depth_max': float(np.max(depth_resized)),
            'depth_mean': float(np.mean(depth_resized))
        }
        
        self.collector.add_sample(img_rgb, depth_resized, metadata)
        self.samples_collected += 1
    
    def _finalize(self):
        """Finalize dataset and print summary."""
        elapsed = time.time() - self.start_time
        
        print(f"\nFinalizing dataset...")
        summary = self.collector.finalize()
        
        # Save extra collection info
        collection_info = {
            'collection_mode': 'panda3d_live',
            'total_samples': self.samples_collected,
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'elapsed_seconds': elapsed,
            'capture_interval': self.capture_interval,
            'image_size': list(self.image_size),
            'depth_format': 'metric' if self.args.depth_metric else 'normalized',
            'scene': 'city.egg',
            'seed': self.args.seed
        }
        
        info_path = Path(self.args.output_dir) / 'collection_info.json'
        with open(info_path, 'w') as f:
            json.dump(collection_info, f, indent=2)
        
        print("\n" + "=" * 70)
        print("COLLECTION COMPLETE!")
        print("=" * 70)
        print(f"  Samples:  {self.samples_collected}")
        print(f"  Episodes: {self.episode_count}")
        print(f"  Time:     {elapsed:.1f}s ({self.samples_collected/max(elapsed,1):.1f} samples/s)")
        print(f"  Saved to: {self.args.output_dir}")
        print("=" * 70)
        
        # Exit the Panda3D application
        self.userExit()


def main():
    args = parse_args()
    app = DepthCollectorApp(args)
    try:
        app.run()
    except (KeyboardInterrupt, SystemExit):
        print("\nCollection interrupted. Finalizing...")
        if app.samples_collected > 0:
            app._finalize()
        else:
            print("No samples collected. Nothing to save.")


if __name__ == "__main__":
    main()
