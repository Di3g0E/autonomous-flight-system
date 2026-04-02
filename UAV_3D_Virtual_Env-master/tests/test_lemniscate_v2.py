#!/usr/bin/env python
"""
Evaluate a trained lemniscate-v2 model.

Runs N_EPISODES evaluation episodes on the lemniscate trajectory and records:
  - Side-by-side video (FPV + aerial perspective)
  - Per-step telemetry CSV with all v2 reward components
  - JSON summary with aggregate metrics

Usage:
    python tests/test_lemniscate_v2.py
    python tests/test_lemniscate_v2.py --model-path ./models/lemniscate_v2/best_model.zip
    python tests/test_lemniscate_v2.py --target-speed 0.15 --episodes 5
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
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

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.utils.episode_recorder import EpisodeRecorder


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate lemniscate-v2 trained model")
    p.add_argument('--model-path', type=str,
                   default='./models/lemniscate_v2/best_model.zip')
    p.add_argument('--output-dir', type=str,
                   default='./experiments/lemniscate_v2')
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--target-speed', type=float, default=0.15,
                   help="Target speed for evaluation (m/s)")
    p.add_argument('--scale', type=float, default=5.0)
    p.add_argument('--target-mode', type=str, default='moving',
                   choices=['fixed', 'moving'],
                   help="'fixed' to test static tracking, 'moving' for lemniscate")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


class LemniscateV2TestApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # Aerial main camera
        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -12, 16)
        self.cam.lookAt(0, 0, 5)

        # FPV camera
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Bird camera
        self.bird_camera = opencv_camera(self, 'bird_cam', 1)
        self.bird_camera.cam.reparentTo(self.render)
        self.bird_camera.cam.setPos(0, -12, 16)
        self.bird_camera.cam.lookAt(0, 0, 5)
        self.bird_camera.buffer.setActive(1)

        # Environment with v2 reward
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode=args.target_mode,
            target_range=3.0,
            target_speed=args.target_speed,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            enable_collisions=False,
            n=args.max_steps,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            filming_mode=True,
            lemniscate_scale=args.scale,
            ideal_fraction=0.25,
            # v2 flags
            use_new_reward=True,
            initial_target_distance=2.0,
            constrained_init=True,
            init_pos_range=0.5,
            init_vel_range=0.25,
            init_ang_range=0.1,
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

    def run_tests(self):
        args = self.args
        np.random.seed(args.seed)
        n_eps = args.episodes

        print(f"\n{'='*60}")
        print(f"  Lemniscate v2 evaluation")
        print(f"  Scale: {args.scale}m   Speed: {args.target_speed}")
        print(f"  Mode: {args.target_mode}   Episodes: {n_eps}")
        print(f"  Max steps: {args.max_steps}")
        print(f"{'='*60}\n")

        telemetry_path = self.out_dir / 'telemetry.csv'
        results = []

        telem_headers = [
            'episode', 'step',
            'drone_x', 'drone_y', 'drone_z',
            'target_x', 'target_y', 'dist_to_target',
            'target_visible', 'target_fraction',
            'r_survival', 'r_stability', 'r_centering',
            'r_scale', 'r_discovery', 'r_not_visible',
            'reward', 'end_reason',
        ]

        with open(telemetry_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(telem_headers)

            for ep in range(n_eps):
                print(f"  Episode {ep+1}/{n_eps}...")
                obs, info = self.env.reset()
                self.ep_recorder.start_episode(ep + 1)

                step = 0
                ep_rewards = []
                ep_distances = []
                ep_centering = []
                ep_stability = []
                ep_visible_steps = 0
                end_reason = "max_steps"

                while step < args.max_steps:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    step += 1
                    self.taskMgr.step()

                    # Extract data
                    target_info = info.get('target', {})
                    vt = info.get('visual_tracking', {})
                    drone_pos = self.env.base_env.state[0:5:2]
                    target_pos = self.env.target_pos

                    dist = target_info.get('distance_to_target', 0)
                    visible = vt.get('target_visible', False)
                    frac = vt.get('target_fraction', 0.0)

                    # v2 components
                    rs = vt.get('r_survival', 0)
                    rst = vt.get('r_stability', 0)
                    rc = vt.get('r_centering', 0)
                    rsc = vt.get('r_scale', 0)
                    rd = vt.get('r_discovery', 0)
                    rnv = vt.get('r_not_visible', 0)

                    ep_rewards.append(reward)
                    ep_distances.append(dist)
                    ep_stability.append(rst)
                    if visible:
                        ep_visible_steps += 1
                        ep_centering.append(rc)

                    # Telemetry
                    writer.writerow([
                        ep + 1, step,
                        round(drone_pos[0], 3), round(drone_pos[1], 3),
                        round(drone_pos[2], 3),
                        round(target_pos[0], 3), round(target_pos[1], 3),
                        round(dist, 3), visible, round(frac, 4),
                        round(rs, 4), round(rst, 4), round(rc, 4),
                        round(rsc, 4), round(rd, 4), round(rnv, 4),
                        round(reward, 4), '',
                    ])

                    # Video
                    fpv_img = self.env._last_high_freq_image
                    bird_img = None
                    ok, bird_rgba = self.bird_camera.get_image()
                    if ok and bird_rgba is not None:
                        bird_img = bird_rgba[:, :, :3]

                    overlay = {
                        'Chunk': f"Ep {ep+1}/{n_eps}",
                        'Step': f"{step}/{args.max_steps}",
                        'Timestep': f"scale={args.scale} speed={args.target_speed}",
                        'Reward': round(reward, 3),
                        'Distance': round(dist, 2),
                        'target': target_info,
                        'visual_tracking': vt,
                    }
                    self.ep_recorder.capture_frame(fpv_img, bird_img, info=overlay)

                    if terminated:
                        end_reason = "out_of_bounds"
                        break
                    if truncated:
                        end_reason = "truncated"
                        break

                self.ep_recorder.end_episode()

                total_reward = float(np.sum(ep_rewards))
                mean_dist = float(np.mean(ep_distances)) if ep_distances else 0
                mean_cent = float(np.mean(ep_centering)) if ep_centering else 0
                mean_stab = float(np.mean(ep_stability)) if ep_stability else 0
                vis_pct = 100.0 * ep_visible_steps / max(step, 1)

                results.append({
                    'episode': ep + 1,
                    'steps': step,
                    'end_reason': end_reason,
                    'total_reward': round(total_reward, 1),
                    'mean_distance': round(mean_dist, 3),
                    'mean_centering': round(mean_cent, 3),
                    'mean_stability': round(mean_stab, 3),
                    'visibility_pct': round(vis_pct, 1),
                    'final_distance': round(dist, 3),
                })

                print(f"    {end_reason} at step {step}  |  "
                      f"dist={mean_dist:.2f}m  vis={vis_pct:.0f}%  "
                      f"R={total_reward:.0f}  stab={mean_stab:.2f}")

        # Summary
        summary = {
            'config': {
                'lemniscate_scale': args.scale,
                'target_speed': args.target_speed,
                'target_mode': args.target_mode,
                'max_steps': args.max_steps,
                'model': args.model_path,
                'n_episodes': n_eps,
            },
            'aggregate': {
                'mean_reward': round(float(np.mean(
                    [r['total_reward'] for r in results])), 1),
                'mean_distance': round(float(np.mean(
                    [r['mean_distance'] for r in results])), 3),
                'mean_visibility_pct': round(float(np.mean(
                    [r['visibility_pct'] for r in results])), 1),
                'mean_stability': round(float(np.mean(
                    [r['mean_stability'] for r in results])), 3),
                'mean_steps': round(float(np.mean(
                    [r['steps'] for r in results])), 1),
            },
            'episodes': results,
        }
        with open(self.out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        agg = summary['aggregate']
        print(f"  AGGREGATE RESULTS")
        print(f"  Mean reward:     {agg['mean_reward']}")
        print(f"  Mean distance:   {agg['mean_distance']}m")
        print(f"  Mean visibility: {agg['mean_visibility_pct']}%")
        print(f"  Mean stability:  {agg['mean_stability']}")
        print(f"  Mean steps:      {agg['mean_steps']}")
        print(f"{'='*60}")
        print(f"  Videos:    {self.video_dir}/")
        print(f"  Telemetry: {telemetry_path}")
        print(f"  Summary:   {self.out_dir / 'summary.json'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 320 240')
        loadPrcFileData('', 'undecorated true')
    app = LemniscateV2TestApp(args)
    app.run_tests()
    app.userExit()
    sys.exit(0)