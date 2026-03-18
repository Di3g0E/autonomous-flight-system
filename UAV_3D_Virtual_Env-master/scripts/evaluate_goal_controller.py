#!/usr/bin/env python
"""
Evaluate a trained goal-conditioned controller over N episodes.

Runs the model in the full Panda3D environment and collects quantitative
metrics suitable for the TFG thesis:

  - Success Rate:     % of episodes where the drone reached the target
  - Collision Rate:   % of episodes ended by crash / bounding box violation
  - Mean Distance:    average final distance to target
  - Search Time:      average time (s) until target first seen in camera
  - Tracking Quality: % of steps with target centred (<25% of image radius)

Outputs:
  - evaluation_results.json   (raw data)
  - evaluation_summary.json   (aggregate metrics)
  - evaluation_table.tex      (LaTeX table for thesis)

Usage:
    python scripts/evaluate_goal_controller.py \\
        --model-path ./models/goal_controller/best_model.zip \\
        --episodes 100 \\
        --output-dir ./experiments/goal_tracking/evaluation
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

import torch
from panda3d.core import Filename, loadPrcFile
from direct.showbase.ShowBase import ShowBase

loadPrcFile(os.path.join(project_root, 'config', 'conf.prc'))

from stable_baselines3 import PPO
from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate goal controller")
    p.add_argument('--model-path', type=str, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--output-dir', type=str,
                   default='./experiments/goal_tracking/evaluation')
    p.add_argument('--target-range', type=float, default=3.0)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


class EvalApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        self.mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)
        ).getFullpath()

        world_setup(self, self.render, self.mydir)
        quad_setup(self, self.render, self.mydir)
        camera_control(self, self.render)

        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

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

        print(f"Loading model from {args.model_path}...")
        self.model = PPO.load(args.model_path, env=None)

    def run_evaluation(self):
        args = self.args
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(args.seed)

        results = []

        for ep in range(args.episodes):
            obs, info = self.env.reset()
            target_pos = info['target']['target_pos'].copy()

            done = False
            step = 0
            total_reward = 0
            first_seen_step = None
            centered_steps = 0
            total_visible_steps = 0

            while not done and step < args.max_steps:
                obs_tensor = {k: v[np.newaxis, ...] for k, v in obs.items()}
                action, _ = self.model.predict(obs_tensor, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action.squeeze())
                done = terminated or truncated
                total_reward += reward
                step += 1

                self.taskMgr.step()

                vis = info.get('visual_tracking', {})
                target_info = info.get('target', {})

                if vis.get('target_visible', False):
                    total_visible_steps += 1
                    if first_seen_step is None:
                        first_seen_step = step

                    # Check centering quality (<25% of max distance from center)
                    tc = vis.get('target_center', None)
                    if tc is not None:
                        cx, cy = tc
                        dist = np.sqrt((cx - 16)**2 + (cy - 16)**2)
                        if dist < 16 * 0.25:  # within 25% of half-width
                            centered_steps += 1

            final_dist = info.get('target', {}).get('distance_to_target', 999)
            arrived = info.get('target', {}).get('arrived', False)

            ep_result = {
                'episode': ep,
                'steps': step,
                'total_reward': float(total_reward),
                'final_distance': float(final_dist),
                'arrived': arrived,
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'search_time': (first_seen_step * 0.01) if first_seen_step else None,
                'visible_fraction': total_visible_steps / max(step, 1),
                'centered_fraction': centered_steps / max(total_visible_steps, 1),
                'target_pos': target_pos.tolist(),
            }
            results.append(ep_result)

            status = '✓' if arrived else '✗'
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [{ep+1:3d}/{args.episodes}] {status} dist={final_dist:.2f}m "
                      f"R={total_reward:.1f} steps={step}")

        # Aggregate
        n = len(results)
        success_rate = sum(1 for r in results if r['arrived']) / n
        collision_rate = sum(1 for r in results if r['terminated'] and not r['arrived']) / n
        timeout_rate = sum(1 for r in results if r['truncated']) / n
        mean_dist = np.mean([r['final_distance'] for r in results])
        std_dist = np.std([r['final_distance'] for r in results])
        mean_reward = np.mean([r['total_reward'] for r in results])

        search_times = [r['search_time'] for r in results if r['search_time'] is not None]
        mean_search = np.mean(search_times) if search_times else None

        visible_fracs = [r['visible_fraction'] for r in results]
        mean_visible = np.mean(visible_fracs)

        centered_fracs = [r['centered_fraction'] for r in results]
        mean_centered = np.mean(centered_fracs)

        summary = {
            'episodes': n,
            'success_rate': float(success_rate),
            'collision_rate': float(collision_rate),
            'timeout_rate': float(timeout_rate),
            'mean_final_distance': float(mean_dist),
            'std_final_distance': float(std_dist),
            'mean_reward': float(mean_reward),
            'mean_search_time_s': float(mean_search) if mean_search else None,
            'mean_visible_fraction': float(mean_visible),
            'mean_centered_fraction': float(mean_centered),
        }

        # Save results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Generate LaTeX table
        latex = self._generate_latex(summary)
        with open(output_dir / 'evaluation_table.tex', 'w') as f:
            f.write(latex)

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Episodes:          {n}")
        print(f"  Success Rate:      {success_rate:.1%}")
        print(f"  Collision Rate:    {collision_rate:.1%}")
        print(f"  Timeout Rate:      {timeout_rate:.1%}")
        print(f"  Mean Distance:     {mean_dist:.2f} ± {std_dist:.2f} m")
        print(f"  Mean Reward:       {mean_reward:.1f}")
        if mean_search is not None:
            print(f"  Mean Search Time:  {mean_search:.2f} s")
        print(f"  Mean Visible:      {mean_visible:.1%}")
        print(f"  Mean Centered:     {mean_centered:.1%}")
        print("=" * 60)
        print(f"\nResults saved to {output_dir}")

    def _generate_latex(self, s):
        search_str = f"{s['mean_search_time_s']:.2f}" if s['mean_search_time_s'] else "N/A"
        return (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\caption{Resultados de evaluación del controlador goal-conditioned}\n"
            "\\label{tab:goal_eval}\n"
            "\\begin{tabular}{lr}\n"
            "\\toprule\n"
            "\\textbf{Métrica} & \\textbf{Valor} \\\\\n"
            "\\midrule\n"
            f"Episodios evaluados & {s['episodes']} \\\\\n"
            f"Tasa de éxito (\\%) & {s['success_rate']*100:.1f} \\\\\n"
            f"Tasa de colisión (\\%) & {s['collision_rate']*100:.1f} \\\\\n"
            f"Tasa de timeout (\\%) & {s['timeout_rate']*100:.1f} \\\\\n"
            f"Distancia final media (m) & {s['mean_final_distance']:.2f} $\\pm$ {s['std_final_distance']:.2f} \\\\\n"
            f"Recompensa media & {s['mean_reward']:.1f} \\\\\n"
            f"Tiempo de búsqueda (s) & {search_str} \\\\\n"
            f"Fracción visible (\\%) & {s['mean_visible_fraction']*100:.1f} \\\\\n"
            f"Fracción centrada (\\%) & {s['mean_centered_fraction']*100:.1f} \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )


def main():
    args = parse_args()
    app = EvalApp(args)
    app.run_evaluation()


if __name__ == "__main__":
    main()
