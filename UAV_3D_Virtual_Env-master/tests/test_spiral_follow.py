#!/usr/bin/env python
"""
Evaluate a trained spiral-follow model.

Runs N evaluation episodes on the Archimedes spiral reference trajectory
and records:
  - Side-by-side video (aerial perspective)
  - Per-step telemetry CSV with all reward components
  - Spiral trajectory plot (reference vs actual)
  - Altitude profile plot
  - Tracking error plot
  - JSON summary with aggregate metrics

Usage:
    python tests/test_spiral_follow.py
    python tests/test_spiral_follow.py --model-path ./models/spiral_follow/best_model.zip
    python tests/test_spiral_follow.py --episodes 10 --max-steps 3000
"""

import argparse
import csv
import json
import math
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
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.envs.spiral_follow_env import SpiralFollowEnv


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate spiral-follow trained model")
    p.add_argument('--model-path', type=str,
                   default='./models/spiral_follow/best_model.zip')
    p.add_argument('--output-dir', type=str,
                   default='./experiments/spiral_follow')
    p.add_argument('--max-steps', type=int, default=3000,
                   help="Max steps per episode (3000 = 30 s)")
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--omega', type=float, default=1.8)
    p.add_argument('--r-growth', type=float, default=0.12)
    p.add_argument('--hover-height', type=float, default=1.39)
    p.add_argument('--vision-radius', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Spiral visualisation in Panda3D
# ──────────────────────────────────────────────────────────────────────

def draw_spiral_3d(app, center_x, center_y, hover_height,
                   omega, r_growth, max_tilt, dt, n_points):
    """Draw the reference spiral as a green line in the Panda3D scene."""
    from panda3d.core import LineSegs, Vec4

    ls = LineSegs()
    ls.setColor(Vec4(0.0, 1.0, 0.2, 0.8))
    ls.setThickness(2)

    theta = 0.0
    for i in range(n_points):
        t = i * dt
        r = r_growth * t + 0.05

        a_budget = 0.70 * 9.82 * math.sin(max_tilt)
        w_max = math.sqrt(a_budget / max(r, 0.05))
        w = min(omega, w_max)
        theta += w * dt

        x = center_x + r * math.cos(theta)
        y = center_y + r * math.sin(theta)
        z = hover_height + 5  # Panda3D z-offset

        if i == 0:
            ls.moveTo(x, y, z)
        else:
            ls.drawTo(x, y, z)

    node = ls.create()
    np_node = app.render.attachNewNode(node)
    return np_node


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────

def plot_results(trajectories, out_dir, args):
    """Generate evaluation plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)

    # ── 1. Spiral trajectory (reference vs actual) ────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for ep_data in trajectories:
        # Reference spiral
        ref_x = [d['x_ref'] for d in ep_data]
        ref_y = [d['y_ref'] for d in ep_data]
        ax.plot(ref_x, ref_y, 'g-', alpha=0.3, linewidth=1)

        # Actual trajectory
        drone_x = [d['drone_x'] for d in ep_data]
        drone_y = [d['drone_y'] for d in ep_data]
        ax.plot(drone_x, drone_y, 'b-', alpha=0.7, linewidth=1)

    # Vision radius circle at origin
    theta_c = np.linspace(0, 2 * np.pi, 100)
    ax.plot(args.vision_radius * np.cos(theta_c),
            args.vision_radius * np.sin(theta_c),
            'r--', alpha=0.4, label=f'Vision radius ({args.vision_radius} m)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Spiral Trajectory: Reference (green) vs Actual (blue)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'spiral_trajectory.png', dpi=150)
    plt.close(fig)
    print(f"  Plot: {out_dir / 'spiral_trajectory.png'}")

    # ── 2. Altitude profile ───────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for i, ep_data in enumerate(trajectories):
        times = [d['step'] * 0.01 for d in ep_data]
        altitudes = [d['drone_z'] for d in ep_data]
        ax.plot(times, altitudes, alpha=0.7, label=f'Ep {i+1}')

    ax.axhline(y=args.hover_height, color='r', linestyle='--',
               alpha=0.5, label=f'Target ({args.hover_height} m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'altitude_profile.png', dpi=150)
    plt.close(fig)
    print(f"  Plot: {out_dir / 'altitude_profile.png'}")

    # ── 3. Tracking error ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i, ep_data in enumerate(trajectories):
        times = [d['step'] * 0.01 for d in ep_data]
        pos_err = [d['pos_error'] for d in ep_data]
        alt_err = [d['alt_error'] for d in ep_data]
        axes[0].plot(times, pos_err, alpha=0.7, label=f'Ep {i+1}')
        axes[1].plot(times, alt_err, alpha=0.7, label=f'Ep {i+1}')

    axes[0].axhline(y=args.vision_radius, color='r', linestyle='--',
                     alpha=0.5, label=f'Vision radius ({args.vision_radius} m)')
    axes[0].set_ylabel('Position Error (m)')
    axes[0].set_title('Tracking Error over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Altitude Error (m)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'tracking_error.png', dpi=150)
    plt.close(fig)
    print(f"  Plot: {out_dir / 'tracking_error.png'}")


# ──────────────────────────────────────────────────────────────────────
# Test Application
# ──────────────────────────────────────────────────────────────────────

class SpiralFollowTestApp(ShowBase):
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
        self.cam.setPos(0, -10, 16)
        self.cam.lookAt(0, 0, 5)

        # Environment (no camera, no target)
        self.base_env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=False,
            use_depth=False,
            use_target=False,
            enable_collisions=False,
            n=args.max_steps,
            t_step=0.01,
            direct_control=1,
            filming_mode=True,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.3,
            init_vel_range=0.15,
            init_ang_range=0.05,
        )

        # Spiral wrapper
        self.spiral_env = SpiralFollowEnv(
            self.base_env,
            omega=args.omega,
            r_growth=args.r_growth,
            hover_height=args.hover_height,
            vision_radius=args.vision_radius,
        )
        self.spiral_env.omega_scale = 1.0  # full speed for evaluation

        # Load model
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model not found at {args.model_path}")
            sys.exit(1)
        self.model = PPO.load(args.model_path, env=None)
        print(f"Model loaded: {args.model_path}")
        total_params = sum(p.numel() for p in self.model.policy.parameters())
        print(f"Policy params: {total_params:,}")

        # Output directory
        self.out_dir = Path(args.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Spiral 3D line (drawn after first reset)
        self._spiral_line_node = None

    def run_tests(self):
        args = self.args
        np.random.seed(args.seed)
        n_eps = args.episodes

        arm_spacing = args.r_growth * 2 * np.pi / args.omega

        print(f"\n{'='*60}")
        print(f"  SPIRAL FOLLOW EVALUATION")
        print(f"  ω={args.omega} rad/s  r_growth={args.r_growth} m/s")
        print(f"  Arm spacing: {arm_spacing:.3f} m  "
              f"Vision radius: {args.vision_radius} m")
        print(f"  Hover height: {args.hover_height} m")
        print(f"  Episodes: {n_eps}  Max steps: {args.max_steps}")
        print(f"{'='*60}\n")

        telemetry_path = self.out_dir / 'telemetry.csv'
        results = []
        all_trajectories = []

        telem_headers = [
            'episode', 'step',
            'drone_x', 'drone_y', 'drone_z',
            'x_ref', 'y_ref',
            'pos_error', 'alt_error',
            'r_tracking', 'r_velocity', 'r_altitude',
            'r_stability', 'r_progress', 'r_off_track',
            'reward', 'spiral_radius', 'omega',
        ]

        with open(telemetry_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(telem_headers)

            for ep in range(n_eps):
                print(f"  Episode {ep+1}/{n_eps}...")
                obs, info = self.spiral_env.reset()

                # Draw spiral reference in 3D (first episode only)
                if ep == 0 and self._spiral_line_node is None:
                    center_x = self.spiral_env._center_x
                    center_y = self.spiral_env._center_y
                    self._spiral_line_node = draw_spiral_3d(
                        self, center_x, center_y,
                        args.hover_height, args.omega, args.r_growth,
                        0.25, 0.01, args.max_steps)

                step = 0
                ep_rewards = []
                ep_pos_errors = []
                ep_alt_errors = []
                ep_trajectory = []
                end_reason = "max_steps"

                while step < args.max_steps:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = \
                        self.spiral_env.step(action)
                    step += 1
                    self.taskMgr.step()

                    sp = info.get('spiral', {})
                    drone_state = self.base_env.base_env.state
                    drone_x = float(drone_state[0])
                    drone_y = float(drone_state[2])
                    drone_z = float(drone_state[4])

                    pos_err = sp.get('pos_error', 0)
                    alt_err = sp.get('alt_error', 0)

                    ep_rewards.append(reward)
                    ep_pos_errors.append(pos_err)
                    ep_alt_errors.append(alt_err)

                    # Per-step data for plots
                    ep_trajectory.append({
                        'step': step,
                        'drone_x': drone_x,
                        'drone_y': drone_y,
                        'drone_z': drone_z,
                        'x_ref': sp.get('x_ref', 0),
                        'y_ref': sp.get('y_ref', 0),
                        'pos_error': pos_err,
                        'alt_error': alt_err,
                    })

                    # CSV
                    writer.writerow([
                        ep + 1, step,
                        round(drone_x, 4), round(drone_y, 4),
                        round(drone_z, 4),
                        round(sp.get('x_ref', 0), 4),
                        round(sp.get('y_ref', 0), 4),
                        round(pos_err, 4),
                        round(alt_err, 4),
                        round(sp.get('r_tracking', 0), 4),
                        round(sp.get('r_velocity', 0), 4),
                        round(sp.get('r_altitude', 0), 4),
                        round(sp.get('r_stability', 0), 4),
                        round(sp.get('r_progress', 0), 4),
                        round(sp.get('r_off_track', 0), 4),
                        round(reward, 4),
                        round(sp.get('spiral_radius', 0), 4),
                        round(sp.get('omega', 0), 4),
                    ])

                    if terminated:
                        end_reason = "out_of_bounds"
                        break
                    if truncated:
                        end_reason = "truncated"
                        break

                all_trajectories.append(ep_trajectory)

                total_reward = float(np.sum(ep_rewards))
                mean_pos_err = float(np.mean(ep_pos_errors))
                mean_alt_err = float(np.mean(ep_alt_errors))
                within_vision = sum(
                    1 for e in ep_pos_errors if e <= args.vision_radius)
                vision_pct = 100.0 * within_vision / max(step, 1)

                results.append({
                    'episode': ep + 1,
                    'steps': step,
                    'end_reason': end_reason,
                    'total_reward': round(total_reward, 1),
                    'mean_pos_error': round(mean_pos_err, 4),
                    'mean_alt_error': round(mean_alt_err, 4),
                    'within_vision_pct': round(vision_pct, 1),
                    'max_pos_error': round(float(max(ep_pos_errors)), 4),
                })

                print(f"    {end_reason} at step {step}  |  "
                      f"pos_err={mean_pos_err:.3f}m  "
                      f"alt_err={mean_alt_err:.3f}m  "
                      f"in_vision={vision_pct:.0f}%  "
                      f"R={total_reward:.0f}")

        # Generate plots
        print(f"\n  Generating plots...")
        plot_results(all_trajectories, self.out_dir, args)

        # Summary
        summary = {
            'config': {
                'omega': args.omega,
                'r_growth': args.r_growth,
                'hover_height': args.hover_height,
                'vision_radius': args.vision_radius,
                'arm_spacing': round(arm_spacing, 4),
                'max_steps': args.max_steps,
                'model': args.model_path,
                'n_episodes': n_eps,
            },
            'aggregate': {
                'mean_reward': round(float(np.mean(
                    [r['total_reward'] for r in results])), 1),
                'mean_pos_error': round(float(np.mean(
                    [r['mean_pos_error'] for r in results])), 4),
                'mean_alt_error': round(float(np.mean(
                    [r['mean_alt_error'] for r in results])), 4),
                'mean_within_vision_pct': round(float(np.mean(
                    [r['within_vision_pct'] for r in results])), 1),
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
        print(f"  Mean reward:        {agg['mean_reward']}")
        print(f"  Mean pos error:     {agg['mean_pos_error']} m")
        print(f"  Mean alt error:     {agg['mean_alt_error']} m")
        print(f"  Within vision:      {agg['mean_within_vision_pct']}%")
        print(f"  Mean steps:         {agg['mean_steps']}")
        print(f"{'='*60}")
        print(f"  Telemetry: {telemetry_path}")
        print(f"  Plots:     {self.out_dir}/")
        print(f"  Summary:   {self.out_dir / 'summary.json'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = SpiralFollowTestApp(args)
    app.run_tests()
    app.userExit()
    sys.exit(0)
