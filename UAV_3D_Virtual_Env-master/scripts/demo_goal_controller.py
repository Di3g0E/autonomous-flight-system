#!/usr/bin/env python
"""
Demo: Visualize a trained goal-conditioned controller in Panda3D.

Loads a trained PPO model and shows the drone following target waypoints
in the 3D city scene.

Usage:
    python scripts/demo_goal_controller.py --model-path ./models/goal_controller/best_model.zip
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo: Goal Controller Visualization")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model .zip file')
    parser.add_argument('--target-mode', type=str, default='waypoints',
                        choices=['fixed', 'waypoints', 'moving'],
                        help='Target mode for demo')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--record', action='store_true',
                        help='Record trajectories to file')
    return parser.parse_args()


def run_demo(args):
    """Run the Panda3D visualization demo."""
    from panda3d.core import loadPrcFileData
    from direct.showbase.ShowBase import ShowBase
    from stable_baselines3 import PPO
    
    from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
    from src.setup.world_setup import world_setup
    from src.setup.quad_setup import quad_setup
    from src.vision.img_2_cv import opencv_camera
    
    # Configure Panda3D
    loadPrcFileData('', 'window-title Goal Controller Demo')
    loadPrcFileData('', 'win-size 1280 720')
    
    print("\n" + "=" * 60)
    print("GOAL CONTROLLER DEMO")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Mode:  {args.target_mode}")
    print("=" * 60)
    
    # Initialize Panda3D
    app = ShowBase()
    mydir = str(Path(__file__).parent.parent)
    world_setup(app, app.render, mydir)
    quad_setup(app, app.render, mydir)
    
    # Setup FPV camera on drone
    fpv_camera = opencv_camera(app, "fpv_cam", frame_interval=1)
    fpv_camera.cam.reparentTo(app.quad_model)
    fpv_camera.cam.setPos(0, 0.5, 0)
    fpv_camera.cam.setHpr(0, 0, 0)
    
    # Create environment with Panda3D integration
    env = Panda3DQuadrotorEnv(
        panda3d_app=app,
        quad_model=app.quad_model,
        render_node=app.render,
        use_camera=True,
        use_target=True,
        target_mode=args.target_mode,
        target_range=3.0,
        camera_high_freq_obj=fpv_camera,
        camera_high_freq_size=(64, 64),
        camera_low_freq_size=(64, 64),
        enable_collisions=False,
        n=1000,
        t_step=0.01,
        direct_control=1
    )
    
    # Load trained model
    print(f"\nLoading model from {args.model_path}...")
    model = PPO.load(args.model_path, env=env)
    print("Model loaded successfully!")
    
    trajectories = []
    
    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        obs, info = env.reset()
        target = info.get('target', {})
        print(f"Target at: [{target.get('target_pos', [0,0,0])}]")
        
        episode_trajectory = {
            'positions': [],
            'targets': [],
            'distances': []
        }
        
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # Record trajectory
            if 'target' in info:
                drone_pos = env.base_env.state[0:5:2]
                episode_trajectory['positions'].append(drone_pos.tolist())
                episode_trajectory['targets'].append(info['target']['target_pos'].tolist())
                episode_trajectory['distances'].append(info['target']['distance_to_target'])
            
            # Update Panda3D render
            app.taskMgr.step()
        
        final_dist = episode_trajectory['distances'][-1] if episode_trajectory['distances'] else 0
        print(f"  Steps: {step}, Reward: {total_reward:.1f}, Final dist: {final_dist:.2f}m")
        trajectories.append(episode_trajectory)
    
    # Save trajectories
    if args.record and trajectories:
        import json
        traj_path = Path(args.model_path).parent / 'demo_trajectories.json'
        with open(traj_path, 'w') as f:
            json.dump(trajectories, f, indent=2)
        print(f"\nTrajectories saved to {traj_path}")
    
    print("\nDemo complete!")
    env.close()


def main():
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
