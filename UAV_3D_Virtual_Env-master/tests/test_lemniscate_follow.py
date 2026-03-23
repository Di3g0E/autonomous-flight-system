#!/usr/bin/env python
"""
Test the trained model following the lemniscate (∞) trajectory.

Runs 5 episodes where the orange sphere traces the figure-8 and the drone
(controlled by the best PPO model) tries to follow it. Each episode is
recorded as a side-by-side video (FPV + aerial perspective). The episode
ends immediately if the drone leaves the bounding box (terminated=True).

Usage:
    python tests/test_lemniscate_follow.py
    python tests/test_lemniscate_follow.py --model-path ./models/goal_controller/best_model.zip
"""

import argparse
import json
import os
import sys
import csv
import numpy as np
import cv2
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
from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.utils.episode_recorder import EpisodeRecorder


def parse_args():
    p = argparse.ArgumentParser(
        description="Test drone following lemniscate trajectory")
    p.add_argument('--model-path', type=str,
                   default='./models/goal_controller/best_model.zip')
    p.add_argument('--output-dir', type=str,
                   default='./experiments/lemniscate_follow')
    p.add_argument('--max-steps', type=int, default=2000,
                   help="Max steps per episode (cut earlier if drone leaves map)")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true',
                   help="Run without the Panda3D visualization window")
    return p.parse_args()


LEMNISCATE_SCALE = 5.0
TARGET_SPEED = 0.3
N_EPISODES = 3


class LemniscateFollowApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # Override main window camera: aerial perspective instead of down-facing
        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -12, 16)
        self.cam.lookAt(0, 0, 5)

        # FPV camera (on the drone)
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Aerial perspective camera
        self.bird_camera = opencv_camera(self, 'bird_cam', 1)
        self.bird_camera.cam.reparentTo(self.render)
        self.bird_camera.cam.setPos(0, -12, 16)
        self.bird_camera.cam.lookAt(0, 0, 5)
        self.bird_camera.buffer.setActive(1)

        # Environment with lemniscate trajectory
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='moving',
            target_range=3.0,
            target_speed=TARGET_SPEED,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            enable_collisions=False,
            n=args.max_steps,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            filming_mode=True,
            lemniscate_scale=LEMNISCATE_SCALE,
        )
        self.env._bird_camera = self.bird_camera

        # Load model
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model not found at {args.model_path}")
            sys.exit(1)
        self.model = PPO.load(args.model_path, env=None)
        print(f"Model loaded: {args.model_path}")

        # Recorder
        self.out_dir = Path(args.output_dir)
        self.video_dir = self.out_dir / 'videos'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.ep_recorder = EpisodeRecorder(
            output_dir=str(self.video_dir), fps=30, resolution=(640, 360))

    # ----------------------------------------------------------------- #
    def run_tests(self):
        np.random.seed(self.args.seed)

        print(f"\n{'='*60}")
        print(f"  Lemniscate follow test")
        print(f"  Scale: {LEMNISCATE_SCALE}m   Speed: {TARGET_SPEED}")
        print(f"  Episodes: {N_EPISODES}   Max steps: {self.args.max_steps}")
        print(f"{'='*60}\n")

        telemetry_path = self.out_dir / 'telemetry.csv'
        results = []

        with open(telemetry_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'episode', 'step', 'drone_x', 'drone_y', 'drone_z',
                'target_x', 'target_y', 'dist_to_target',
                'target_visible', 'centering_reward', 'scale_reward',
                'target_fraction', 'reward', 'terminated_reason'
            ])

            for ep in range(N_EPISODES):
                print(f"  Episode {ep+1}/{N_EPISODES}...")
                obs, info = self.env.reset()
                self.ep_recorder.start_episode(ep + 1)

                step = 0
                ep_rewards = []
                ep_distances = []
                ep_centering = []
                end_reason = "max_steps"

                while step < self.args.max_steps:
                    # Model prediction
                    action, _ = self.model.predict(obs, deterministic=True)

                    # Step
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    step += 1

                    # Panda3D tick
                    self.taskMgr.step()

                    # Extract data
                    target_info = info.get('target', {})
                    vt_info = info.get('visual_tracking', {})
                    drone_pos = self.env.base_env.state[0:5:2]
                    target_pos = self.env.target_pos

                    dist = target_info.get('distance_to_target', 0)
                    visible = vt_info.get('target_visible', False)
                    centering = vt_info.get('centering_reward', 0.0)
                    scale_r = vt_info.get('scale_reward', 0.0)
                    frac = vt_info.get('target_fraction', 0.0)

                    ep_rewards.append(reward)
                    ep_distances.append(dist)
                    if visible:
                        ep_centering.append(centering)

                    # Telemetry row
                    writer.writerow([
                        ep + 1, step,
                        round(drone_pos[0], 3), round(drone_pos[1], 3),
                        round(drone_pos[2], 3),
                        round(target_pos[0], 3), round(target_pos[1], 3),
                        round(dist, 3), visible,
                        round(centering, 3), round(scale_r, 3),
                        round(frac, 4), round(reward, 3), ''
                    ])

                    # Video frame
                    fpv_img = self.env._last_high_freq_image
                    bird_img = None
                    ok, bird_rgba = self.bird_camera.get_image()
                    if ok and bird_rgba is not None:
                        bird_img = bird_rgba[:, :, :3]

                    overlay = {
                        'Chunk': f"Ep {ep+1}/{N_EPISODES}",
                        'Step': f"{step}/{self.args.max_steps}",
                        'Timestep': f"scale={LEMNISCATE_SCALE} speed={TARGET_SPEED}",
                        'Reward': round(reward, 2),
                        'Distance': round(dist, 2),
                        'target': target_info,
                        'visual_tracking': vt_info,
                    }
                    self.ep_recorder.capture_frame(fpv_img, bird_img, info=overlay)

                    # Stop if drone left the map
                    if terminated:
                        end_reason = "out_of_bounds"
                        # Write the termination reason on the last row
                        writer.writerow([
                            ep + 1, step,
                            round(drone_pos[0], 3), round(drone_pos[1], 3),
                            round(drone_pos[2], 3),
                            round(target_pos[0], 3), round(target_pos[1], 3),
                            round(dist, 3), visible,
                            round(centering, 3), round(scale_r, 3),
                            round(frac, 4), round(reward, 3),
                            'OUT_OF_BOUNDS'
                        ])
                        break
                    if truncated:
                        end_reason = "truncated"
                        break

                self.ep_recorder.end_episode()

                mean_dist = float(np.mean(ep_distances)) if ep_distances else 0
                mean_cent = float(np.mean(ep_centering)) if ep_centering else 0
                total_reward = float(np.sum(ep_rewards))

                results.append({
                    'episode': ep + 1,
                    'steps': step,
                    'end_reason': end_reason,
                    'total_reward': round(total_reward, 1),
                    'mean_distance': round(mean_dist, 3),
                    'mean_centering': round(mean_cent, 3),
                    'final_distance': round(dist, 3),
                })

                print(f"    {end_reason} at step {step}  |  "
                      f"mean_dist={mean_dist:.2f}m  reward={total_reward:.0f}")

        # Summary
        summary = {
            'config': {
                'lemniscate_scale': LEMNISCATE_SCALE,
                'target_speed': TARGET_SPEED,
                'max_steps': self.args.max_steps,
                'model': self.args.model_path,
            },
            'mean_distance': round(float(np.mean(
                [r['mean_distance'] for r in results])), 3),
            'mean_steps': round(float(np.mean(
                [r['steps'] for r in results])), 1),
            'episodes': results,
        }
        with open(self.out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Done! Results in {self.out_dir}/")
        print(f"  Videos:    {self.video_dir}/")
        print(f"  Telemetry: {telemetry_path}")
        print(f"  Summary:   {self.out_dir / 'summary.json'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 320 240')
        loadPrcFileData('', 'undecorated true')
    app = LemniscateFollowApp(args)
    app.run_tests()
    app.userExit()      # Shut down Panda3D engine cleanly
    sys.exit(0)         # Force-terminate any lingering threads