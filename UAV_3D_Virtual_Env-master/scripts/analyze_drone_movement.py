#!/usr/bin/env python
"""
Analyze drone movement during goal-conditioned flight.

Loads a trained model and runs evaluation episodes, recording the full
state trajectory on every timestep. Generates publication-quality plots for
the TFG thesis document:

  1. 3D trajectory (start → target)
  2. Temporal evolution of x, y, z, yaw
  3. Motor action heatmap over time
  4. Target visibility timeline
  5. Target centering quality over time

Usage:
    python scripts/analyze_drone_movement.py \\
        --model-path ./models/goal_controller/best_model.zip \\
        --episodes 10 \\
        --output-dir ./experiments/goal_tracking/analysis
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # before Panda3D
from panda3d.core import Filename, loadPrcFile
from direct.showbase.ShowBase import ShowBase

loadPrcFile(os.path.join(project_root, 'config', 'conf.prc'))

from stable_baselines3 import PPO
from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_args():
    p = argparse.ArgumentParser(description="Analyze drone movement")
    p.add_argument('--model-path', type=str, required=True,
                   help='Path to trained model .zip')
    p.add_argument('--episodes', type=int, default=10,
                   help='Number of evaluation episodes')
    p.add_argument('--output-dir', type=str,
                   default='./experiments/goal_tracking/analysis',
                   help='Directory to save plots')
    p.add_argument('--target-range', type=float, default=3.0)
    p.add_argument('--max-steps', type=int, default=1000)
    return p.parse_args()


class AnalysisApp(ShowBase):
    """Panda3D app for running evaluation with full state logging."""

    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        self.mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)
        ).getFullpath()

        world_setup(self, self.render, self.mydir)
        quad_setup(self, self.render, self.mydir)
        camera_control(self, self.render)

        # FPV camera
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Environment
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='fixed',
            target_range=args.target_range,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            enable_collisions=False,
            n=args.max_steps,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
        )

        # Load model
        print(f"Loading model from {args.model_path}...")
        self.model = PPO.load(args.model_path, env=None)

    def run_analysis(self):
        args = self.args
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_episodes = []

        for ep in range(args.episodes):
            obs, info = self.env.reset()
            target_pos = info['target']['target_pos'].copy()

            episode_data = {
                'target_pos': target_pos.tolist(),
                'timesteps': [],
            }

            done = False
            step = 0
            total_reward = 0

            while not done and step < args.max_steps:
                # Predict action
                obs_tensor = {k: v[np.newaxis, ...] for k, v in obs.items()}
                action, _ = self.model.predict(obs_tensor, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action.squeeze())
                done = terminated or truncated
                total_reward += reward
                step += 1

                # Advance Panda3D
                self.taskMgr.step()

                # Log state
                state = self.env.base_env.state
                ang = self.env.base_env.ang
                target_info = info.get('target', {})
                vis_info = info.get('visual_tracking', {})

                episode_data['timesteps'].append({
                    'step': step,
                    'time': step * 0.01,
                    'pos': [float(state[0]), float(state[2]), float(state[4])],
                    'vel': [float(state[1]), float(state[3]), float(state[5])],
                    'euler': [float(ang[0]), float(ang[1]), float(ang[2])],
                    'action': action.squeeze().tolist(),
                    'reward': float(reward),
                    'distance': float(target_info.get('distance_to_target', 0)),
                    'arrived': bool(target_info.get('arrived', False)),
                    'target_visible': bool(vis_info.get('target_visible', False)),
                    'target_pixels': int(vis_info.get('target_pixels', 0)),
                    'centering_reward': float(vis_info.get('centering_reward', 0)),
                    'target_center': vis_info.get('target_center', None),
                })

            episode_data['total_reward'] = float(total_reward)
            episode_data['total_steps'] = step
            episode_data['arrived'] = episode_data['timesteps'][-1]['arrived'] if episode_data['timesteps'] else False
            all_episodes.append(episode_data)

            status = '✓ ARRIVED' if episode_data['arrived'] else '✗'
            print(f"  Episode {ep+1}/{args.episodes}: {step} steps | "
                  f"R={total_reward:.1f} | dist={episode_data['timesteps'][-1]['distance']:.2f}m | {status}")

        # Save raw data
        data_path = output_dir / 'movement_data.json'
        with open(data_path, 'w') as f:
            json.dump(all_episodes, f, indent=2)
        print(f"\nRaw data saved to {data_path}")

        # Generate plots
        self._plot_trajectories(all_episodes, output_dir)
        self._plot_temporal(all_episodes, output_dir)
        self._plot_actions(all_episodes, output_dir)
        self._plot_visibility(all_episodes, output_dir)
        self._plot_centering(all_episodes, output_dir)

        print(f"\nAll plots saved to {output_dir}")

    # ── Plot functions ──

    def _plot_trajectories(self, episodes, out_dir):
        """3D trajectory plot with start and target markers."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, ep in enumerate(episodes):
            ts = ep['timesteps']
            xs = [t['pos'][0] for t in ts]
            ys = [t['pos'][1] for t in ts]
            zs = [t['pos'][2] for t in ts]
            color = 'green' if ep['arrived'] else 'steelblue'
            alpha = 0.8 if ep['arrived'] else 0.4
            ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=0.8)
            ax.scatter(xs[0], ys[0], zs[0], c='blue', s=30, marker='o', zorder=5)

            tp = ep['target_pos']
            ax.scatter(tp[0], tp[1], tp[2], c='orange', s=80, marker='*', zorder=5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Trayectorias 3D del Dron')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)
        plt.tight_layout()
        plt.savefig(out_dir / 'trajectories_3d.png', dpi=150)
        plt.close()
        print("  📊 trajectories_3d.png")

    def _plot_temporal(self, episodes, out_dir):
        """Temporal evolution of position and yaw for the first episode."""
        ep = episodes[0]
        ts = ep['timesteps']
        t = [s['time'] for s in ts]
        x = [s['pos'][0] for s in ts]
        y = [s['pos'][1] for s in ts]
        z = [s['pos'][2] for s in ts]
        yaw = [np.degrees(s['euler'][2]) for s in ts]
        dist = [s['distance'] for s in ts]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(t, x, label='X', color='#e74c3c')
        axes[0].plot(t, y, label='Y', color='#2ecc71')
        axes[0].plot(t, z, label='Z', color='#3498db')
        axes[0].axhline(ep['target_pos'][0], ls='--', color='#e74c3c', alpha=0.3, label='Target X')
        axes[0].axhline(ep['target_pos'][1], ls='--', color='#2ecc71', alpha=0.3, label='Target Y')
        axes[0].set_ylabel('Posición (m)')
        axes[0].legend(ncol=3, fontsize=8)
        axes[0].set_title('Evolución Temporal del Estado del Dron (Episodio 1)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, yaw, color='#9b59b6', label='Yaw')
        axes[1].set_ylabel('Yaw (°)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, dist, color='#e67e22', label='Distancia al Target')
        axes[2].axhline(0.3, ls='--', color='green', alpha=0.5, label='Umbral llegada')
        axes[2].set_ylabel('Distancia (m)')
        axes[2].set_xlabel('Tiempo (s)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'temporal_evolution.png', dpi=150)
        plt.close()
        print("  📊 temporal_evolution.png")

    def _plot_actions(self, episodes, out_dir):
        """Motor action heatmap for the first episode."""
        ep = episodes[0]
        ts = ep['timesteps']
        actions = np.array([s['action'] for s in ts])  # (steps, 4)
        time_arr = np.array([s['time'] for s in ts])

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(actions.T, aspect='auto', cmap='RdBu_r',
                        vmin=-1, vmax=1,
                        extent=[time_arr[0], time_arr[-1], 3.5, -0.5])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4'])
        ax.set_xlabel('Tiempo (s)')
        ax.set_title('Acciones de los Motores (Episodio 1)')
        plt.colorbar(im, ax=ax, label='Fuerza normalizada [-1, 1]')
        plt.tight_layout()
        plt.savefig(out_dir / 'action_heatmap.png', dpi=150)
        plt.close()
        print("  📊 action_heatmap.png")

    def _plot_visibility(self, episodes, out_dir):
        """Target visibility timeline for the first episode."""
        ep = episodes[0]
        ts = ep['timesteps']
        t = [s['time'] for s in ts]
        vis = [1 if s['target_visible'] else 0 for s in ts]
        pixels = [s['target_pixels'] for s in ts]

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        axes[0].fill_between(t, vis, alpha=0.4, color='green', step='post')
        axes[0].plot(t, vis, color='green', drawstyle='steps-post', linewidth=0.5)
        axes[0].set_ylabel('Visible')
        axes[0].set_yticks([0, 1])
        axes[0].set_yticklabels(['No', 'Sí'])
        axes[0].set_title('Visibilidad del Objetivo en la Cámara FPV (Episodio 1)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, pixels, color='orange', linewidth=0.8)
        axes[1].set_ylabel('Píxeles Naranja')
        axes[1].set_xlabel('Tiempo (s)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'visibility_timeline.png', dpi=150)
        plt.close()
        print("  📊 visibility_timeline.png")

    def _plot_centering(self, episodes, out_dir):
        """Target centering quality over time for the first episode."""
        ep = episodes[0]
        ts = ep['timesteps']
        t = [s['time'] for s in ts]
        centering = [s['centering_reward'] for s in ts]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, centering, color='#2ecc71', linewidth=0.8, label='Centering Reward')
        ax.axhline(2.0, ls='--', color='green', alpha=0.3, label='Máximo (2.0)')
        ax.axhline(0, ls='-', color='gray', alpha=0.3)
        ax.fill_between(t, centering, alpha=0.15, color='#2ecc71')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Recompensa de Centrado')
        ax.set_title('Calidad de Centrado del Objetivo (Episodio 1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'centering_quality.png', dpi=150)
        plt.close()
        print("  📊 centering_quality.png")


def main():
    args = parse_args()
    app = AnalysisApp(args)
    app.run_analysis()
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
