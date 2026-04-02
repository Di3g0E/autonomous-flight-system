#!/usr/bin/env python
"""
Evaluate the trained hover-tracking model with optional spiral search.

Runs N episodes where the drone must keep the magenta sphere centred
in the downward-facing camera.  When the sphere disappears for K
consecutive steps, the pre-trained spiral model takes over until the
sphere is re-acquired, then control blends back to the tracking policy.

Outputs
-------
  experiments/hover_track/videos/episode_*.mp4   (FPV + aerial side-by-side)
  experiments/hover_track/telemetry.csv           (per-step data)
  experiments/hover_track/summary.json            (aggregate metrics)

Usage:
    python tests/test_hover_track.py
    python tests/test_hover_track.py --target-mode moving --target-speed 0.1
    python tests/test_hover_track.py --no-spiral   (disable spiral fallback)
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
from stable_baselines3 import SAC

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.utils.episode_recorder import EpisodeRecorder
from src.agents.spiral_search_controller import SpiralSearchController


# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Test hover-tracking model with spiral fallback")
    p.add_argument('--model-path', type=str,
                   default='./models/hover_track/best_model.zip')
    p.add_argument('--spiral-model', type=str,
                   default='./models/spiral_follow/best_model.zip')
    p.add_argument('--output-dir', type=str,
                   default='./experiments/hover_track')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=2000,
                   help="Max steps per episode (2000 = 20s)")
    p.add_argument('--hover-height', type=float, default=1.394)
    p.add_argument('--target-mode', type=str, default='fixed',
                   choices=['fixed', 'moving'],
                   help="Target behaviour during test")
    p.add_argument('--target-speed', type=float, default=0.0,
                   help="Speed for moving target (m/s)")
    p.add_argument('--lemniscate-scale', type=float, default=2.5,
                   help="Half-width for lemniscate trajectory")
    p.add_argument('--no-spiral', action='store_true',
                   help="Disable spiral fallback (pure RL)")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────

class HoverTrackTestApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # Aerial camera for main window
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

        # Bird camera for recording
        self.bird_camera = opencv_camera(self, 'bird_cam', 1)
        self.bird_camera.cam.reparentTo(self.render)
        self.bird_camera.cam.setPos(0, -8, 14)
        self.bird_camera.cam.lookAt(0, 0, 5)
        self.bird_camera.buffer.setActive(1)

        # Environment
        self.env = Panda3DQuadrotorEnv(
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
            target_mode=args.target_mode,
            target_speed=args.target_speed,
            target_radius=0.25,
            filming_mode=True,
            enable_collisions=False,
            n=args.max_steps,
            t_step=0.01,
            direct_control=1,
            lemniscate_scale=args.lemniscate_scale,
            centroid_obs=True,
            camera_down=True,
            hover_height=args.hover_height,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.2,
            init_vel_range=0.1,
            init_ang_range=0.05,
        )
        self.env._bird_camera = self.bird_camera

        # Load tracking model
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model not found: {args.model_path}")
            sys.exit(1)
        self.model = SAC.load(args.model_path, env=None)
        print(f"Tracking model: {args.model_path}")

        # Spiral search controller
        self.spiral_ctrl = None
        if not args.no_spiral and os.path.exists(args.spiral_model):
            self.spiral_ctrl = SpiralSearchController(
                spiral_model_path=args.spiral_model,
                hover_height=args.hover_height,
            )
            print(f"Spiral model:   {args.spiral_model}")
        elif not args.no_spiral:
            print(f"WARNING: Spiral model not found: {args.spiral_model}")
            print("         Running without spiral fallback.")

        # Recorder
        self.out_dir = Path(args.output_dir)
        self.video_dir = self.out_dir / 'videos'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.ep_recorder = EpisodeRecorder(
            output_dir=str(self.video_dir), fps=30, resolution=(640, 360))

    # ────────────────────────────────────────────────────────────────

    def run_tests(self):
        args = self.args
        np.random.seed(args.seed)

        print(f"\n{'='*60}")
        print(f"  Hover-Track Evaluation")
        print(f"  Target mode: {args.target_mode}  "
              f"Speed: {args.target_speed}  "
              f"Height: {args.hover_height}m")
        print(f"  Episodes: {args.episodes}  "
              f"Max steps: {args.max_steps}")
        print(f"  Spiral: {'ON' if self.spiral_ctrl else 'OFF'}")
        print(f"{'='*60}\n")

        telemetry_path = self.out_dir / 'telemetry.csv'
        results = []

        with open(telemetry_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'episode', 'step', 'drone_x', 'drone_y', 'drone_z',
                'target_x', 'target_y', 'target_z',
                'centroid_x', 'centroid_y', 'fraction', 'visible',
                'r_stability', 'r_centering', 'r_scale',
                'reward', 'action_mag', 'controller_state',
            ])

            for ep in range(args.episodes):
                print(f"  Episode {ep+1}/{args.episodes}...")
                obs, info = self.env.reset()
                if self.spiral_ctrl:
                    self.spiral_ctrl.reset()

                self.ep_recorder.start_episode(ep + 1)

                step = 0
                ep_rewards = []
                ep_centering = []
                ep_action_mags = []
                end_reason = "max_steps"

                while step < args.max_steps:
                    # Determine action
                    vt = info.get('visual_tracking', {})
                    target_visible = vt.get('target_visible', False)
                    ctrl_state = 'track'

                    if self.spiral_ctrl:
                        action = self.spiral_ctrl.get_action(
                            obs, target_visible, self.model, self.env)
                        ctrl_state = self.spiral_ctrl.current_state
                    else:
                        action, _ = self.model.predict(obs, deterministic=True)

                    obs, reward, terminated, truncated, info = self.env.step(action)
                    step += 1
                    self.taskMgr.step()

                    # Extract data
                    vt = info.get('visual_tracking', {})
                    drone_pos = self.env.base_env.state[0:5:2]
                    target_pos = self.env.target_pos
                    action_mag = float(np.mean(np.abs(action)))

                    ep_rewards.append(reward)
                    ep_action_mags.append(action_mag)
                    if vt.get('target_visible', False) and 'centering_dist' in vt:
                        ep_centering.append(vt['centering_dist'])

                    # Telemetry row
                    writer.writerow([
                        ep + 1, step,
                        round(drone_pos[0], 3), round(drone_pos[1], 3),
                        round(drone_pos[2], 3),
                        round(target_pos[0], 3), round(target_pos[1], 3),
                        round(target_pos[2], 3),
                        round(obs[13], 3), round(obs[14], 3),
                        round(obs[15], 4), round(obs[16], 0),
                        round(vt.get('r_stability', 0), 3),
                        round(vt.get('r_centering', 0), 3),
                        round(vt.get('r_scale', 0), 3),
                        round(reward, 3), round(action_mag, 3),
                        ctrl_state,
                    ])

                    # Video frame
                    fpv_img = self.env._last_high_freq_image
                    bird_img = None
                    ok, bird_rgba = self.bird_camera.get_image()
                    if ok and bird_rgba is not None:
                        bird_img = bird_rgba[:, :, :3]

                    overlay = {
                        'Chunk': f"Ep {ep+1}/{args.episodes}",
                        'Step': f"{step}/{args.max_steps}",
                        'Timestep': f"ctrl={ctrl_state}",
                        'Reward': round(reward, 2),
                        'Distance': round(np.linalg.norm(
                            drone_pos - target_pos), 2),
                        'target': info.get('target', {}),
                        'visual_tracking': vt,
                    }
                    self.ep_recorder.capture_frame(fpv_img, bird_img,
                                                    info=overlay)

                    if terminated:
                        end_reason = "out_of_bounds"
                        break
                    if truncated:
                        end_reason = "truncated"
                        break

                self.ep_recorder.end_episode()

                mean_cent = (float(np.mean(ep_centering))
                             if ep_centering else -1)
                mean_act = float(np.mean(ep_action_mags))
                total_reward = float(np.sum(ep_rewards))

                results.append({
                    'episode': ep + 1,
                    'steps': step,
                    'end_reason': end_reason,
                    'total_reward': round(total_reward, 1),
                    'mean_centering_dist': round(mean_cent, 3),
                    'mean_action_mag': round(mean_act, 3),
                    'visibility_pct': round(
                        100 * sum(1 for r in ep_rewards if r > 0) / max(step, 1), 1),
                })

                flag = " [!ACT]" if mean_act > 0.3 else ""
                print(f"    {end_reason:15s} step={step:4d}  "
                      f"R={total_reward:7.1f}  cent={mean_cent:.3f}  "
                      f"|a|={mean_act:.3f}{flag}")

        # Summary
        summary = {
            'config': {
                'hover_height': args.hover_height,
                'target_mode': args.target_mode,
                'target_speed': args.target_speed,
                'max_steps': args.max_steps,
                'model': args.model_path,
                'spiral': args.spiral_model if self.spiral_ctrl else None,
            },
            'mean_reward': round(float(np.mean(
                [r['total_reward'] for r in results])), 1),
            'mean_steps': round(float(np.mean(
                [r['steps'] for r in results])), 1),
            'mean_centering_dist': round(float(np.mean(
                [r['mean_centering_dist'] for r in results
                 if r['mean_centering_dist'] >= 0])), 3),
            'mean_action_mag': round(float(np.mean(
                [r['mean_action_mag'] for r in results])), 3),
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
    app = HoverTrackTestApp(args)
    app.run_tests()
    app.userExit()
    sys.exit(0)