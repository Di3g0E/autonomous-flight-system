#!/usr/bin/env python
"""
Generate 10 recorded test episodes for documentation and analysis.
Automatically loads the best model and captures FPV + Bird's-eye views with telemetry.
"""

import argparse
import json
import os
import sys
import time
import csv
import numpy as np
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# IMPORTANT: Import torch BEFORE Panda3D
import torch
from panda3d.core import Filename, loadPrcFile
from direct.showbase.ShowBase import ShowBase

# Load Panda3D config
prc_path = os.path.join(project_root, 'config', 'conf.prc')
loadPrcFile(Filename.fromOsSpecific(prc_path))

from stable_baselines3 import PPO
from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.utils.episode_recorder import EpisodeRecorder

def parse_args():
    p = argparse.ArgumentParser(description="Record 10 test episodes")
    p.add_argument('--model-path', type=str, 
                   default='./models/goal_controller/best_model.zip')
    p.add_argument('--output-dir', type=str,
                   default='./experiments/recorded_tests')
    p.add_argument('--target-range', type=float, default=3.0)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()

class RecordApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        self.mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)
        ).getFullpath()

        print("Loading 3D world and drone...")
        world_setup(self, self.render, self.mydir)
        quad_setup(self, self.render, self.mydir)
        camera_control(self, self.render)

        # FPV Camera
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Bird's-eye Camera
        self.bird_camera = opencv_camera(self, 'bird_cam', 1)
        self.bird_camera.cam.reparentTo(self.render)
        self.bird_camera.cam.setPos(0, -8, 12)
        self.bird_camera.cam.lookAt(0, 0, 5)
        self.bird_camera.buffer.setActive(1)

        print("Creating environment and loading model...")
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='moving',
            target_range=args.target_range,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            enable_collisions=False,
            n=args.max_steps,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            filming_mode=True, # Follow mode
        )
        self.env.unwrapped._bird_camera = self.bird_camera

        if not os.path.exists(args.model_path):
            print(f"ERROR: Model not found at {args.model_path}")
            sys.exit(1)
            
        self.model = PPO.load(args.model_path, env=None)
        
        # Setup recorder
        self.out_dir = Path(args.output_dir)
        self.video_dir = self.out_dir / 'videos'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.ep_recorder = EpisodeRecorder(
            output_dir=str(self.video_dir),
            fps=30,
            resolution=(640, 360)
        )

    def run_tests(self):
        print(f"\nStarting 10 recorded tests. Output: {self.args.output_dir}")
        np.random.seed(self.args.seed)
        
        telemetry_path = self.out_dir / 'telemetry.csv'
        telemetry_headers = [
            'episode', 'step', 'x', 'y', 'z', 'dist_to_target', 
            'target_visible', 'centering_reward', 'scale_reward', 'target_fraction'
        ]
        
        results = []
        
        with open(telemetry_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(telemetry_headers)
            
            for ep in range(10):
                print(f"  Recording Episode {ep+1}/10...")
                obs, info = self.env.reset()
                self.ep_recorder.start_episode(ep + 1)
                
                done = False
                step = 0
                ep_distances = []
                ep_centering = []
                
                while not done and step < self.args.max_steps:
                    # Model prediction
                    obs_tensor = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
                    action, _ = self.model.predict(obs_tensor, deterministic=True)
                    
                    # Environment step
                    obs, reward, terminated, truncated, info = self.env.step(action.squeeze())
                    done = terminated or truncated
                    step += 1
                    
                    # Panda3D tick
                    self.taskMgr.step()
                    
                    # Extract info
                    target_info = info.get('target', {})
                    vt_info = info.get('visual_tracking', {})
                    state = info.get('state', [0]*13) # [x, vx, y, vy, z, vz, ...]
                    
                    dist = target_info.get('distance_to_target', 0)
                    ep_distances.append(dist)
                    
                    visible = vt_info.get('target_visible', False)
                    centering = vt_info.get('centering_reward', 0.0)
                    if visible:
                        ep_centering.append(centering)
                    
                    # Log telemetry
                    writer.writerow([
                        ep + 1, step, 
                        round(state[0], 3), round(state[2], 3), round(state[4], 3),
                        round(dist, 3), visible, 
                        round(centering, 3), 
                        round(vt_info.get('scale_reward', 0.0), 3),
                        round(vt_info.get('target_fraction', 0.0), 4)
                    ])
                    
                    # Capture video frame
                    fpv_img = self.env._last_high_freq_image
                    bird_img = None
                    if self.bird_camera:
                        success, bird_rgba = self.bird_camera.get_image()
                        if success and bird_rgba is not None:
                            bird_img = bird_rgba[:, :, :3]
                            
                    overlay = {
                        'Chunk': ep + 1,
                        'Step': f"{step}/{self.args.max_steps}",
                        'Timestep': "EVAL",
                        'Reward': round(reward, 2),
                        'Distance': round(dist, 2),
                        'target': target_info,
                        'visual_tracking': vt_info
                    }
                    self.ep_recorder.capture_frame(fpv_img, bird_img, info=overlay)
                
                self.ep_recorder.end_episode()
                
                results.append({
                    'episode': ep + 1,
                    'steps': step,
                    'mean_dist': float(np.mean(ep_distances)),
                    'mean_centering': float(np.mean(ep_centering)) if ep_centering else 0.0,
                    'final_dist': float(dist)
                })
                print(f"    Done. Steps: {step}, Mean Dist: {np.mean(ep_distances):.2f}m")

        # Save summary
        summary = {
            'mean_distance': float(np.mean([r['mean_dist'] for r in results])),
            'mean_centering': float(np.mean([r['mean_centering'] for r in results])),
            'episodes': results
        }
        with open(self.out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\nAll 10 tests recorded successfully!")
        print(f"Videos: {self.video_dir}")
        print(f"Telemetry: {telemetry_path}")
        print(f"Summary: {self.out_dir / 'summary.json'}")

if __name__ == "__main__":
    args = parse_args()
    app = RecordApp(args)
    app.run_tests()
