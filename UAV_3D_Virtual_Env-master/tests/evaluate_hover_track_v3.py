#!/usr/bin/env python
"""
Multi-checkpoint evaluator for hover-track v3.

Loads the Panda3D scene ONCE and evaluates multiple checkpoints
sequentially.  For each checkpoint it runs N episodes per difficulty
tier and writes a per-checkpoint summary + a global comparison table.

Prioritised checkpoints (best Phase-B region + latest):
    750k, 800k, 850k, 900k, latest

Usage:
    python tests/evaluate_hover_track_v3.py
    python tests/evaluate_hover_track_v3.py --checkpoints 800000 850000 900000
    python tests/evaluate_hover_track_v3.py --episodes-per-tier 10 --duration 20
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

import torch  # noqa: F401
from panda3d.core import Filename, loadPrcFile, loadPrcFileData
from direct.showbase.ShowBase import ShowBase

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from stable_baselines3 import SAC

from src.simulation.world_setup import world_setup, quad_setup
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv

# ── Defaults ──────────────────────────────────────────────────────────
CKPT_DIR = './models/hover_track_v3/checkpoints'
DURATION_S = 20
EPISODES_PER_TIER = 10

# Priority checkpoints (from training analysis: peak at ~800-850k)
DEFAULT_STEPS = [750_000, 800_000, 850_000, 900_000]

TIERS = {
    'easy':   {'offset': 0.2, 'vel': 0.10, 'ang': 0.05},
    'medium': {'offset': 0.6, 'vel': 0.25, 'ang': 0.10},
    'hard':   {'offset': 1.0, 'vel': 0.35, 'ang': 0.15},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate hover-track v3 checkpoints across tiers")
    p.add_argument('--checkpoint-dir', type=str, default=CKPT_DIR)
    p.add_argument('--checkpoints', nargs='+', type=int, default=None,
                   help="Specific checkpoint steps to evaluate "
                        "(default: 750k 800k 850k 900k + latest)")
    p.add_argument('--duration', type=int, default=DURATION_S,
                   help="Episode duration in seconds")
    p.add_argument('--episodes-per-tier', type=int, default=EPISODES_PER_TIER)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--output-dir', type=str,
                   default='./experiments/hover_track_v3')
    return p.parse_args()


def discover_checkpoints(ckpt_dir, requested_steps):
    """Find checkpoint files, prioritising requested steps + latest."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    # Discover all available checkpoints
    available = {}
    for f in ckpt_dir.glob('model_*_steps.zip'):
        try:
            step = int(f.stem.split('_')[1])
            available[step] = f
        except (ValueError, IndexError):
            pass

    if not available:
        print(f"ERROR: No checkpoints found in {ckpt_dir}")
        sys.exit(1)

    latest_step = max(available.keys())

    # Build ordered list: requested + latest (no duplicates)
    if requested_steps is None:
        requested_steps = DEFAULT_STEPS

    selected = []
    seen = set()
    for s in requested_steps:
        if s in available and s not in seen:
            selected.append((s, available[s]))
            seen.add(s)
        elif s not in available:
            # Find closest available
            closest = min(available.keys(), key=lambda x: abs(x - s))
            if closest not in seen:
                print(f"  Note: {s} not found, using closest: {closest}")
                selected.append((closest, available[closest]))
                seen.add(closest)

    # Always include latest
    if latest_step not in seen:
        selected.append((latest_step, available[latest_step]))

    return selected


# ══════════════════════════════════════════════════════════════════════
# Panda3D Evaluation App
# ══════════════════════════════════════════════════════════════════════

class EvalV3App(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)

        # FPV camera (downward)
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, 0.01)
        self.fpv_camera.cam.lookAt(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Environment — v3 reward
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='fixed',
            target_range=3.0,
            target_speed=0.0,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            enable_collisions=False,
            n=args.duration * 100 + 50,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            filming_mode=True,
            camera_down=True,
            hover_height=1.394,
            centroid_obs=True,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.3,
            init_vel_range=0.10,
            init_ang_range=0.05,
            reward_version='v3',
        )

        # Discover checkpoints
        self.checkpoints = discover_checkpoints(
            args.checkpoint_dir, args.checkpoints)
        print(f"\nCheckpoints to evaluate: "
              f"{[s for s, _ in self.checkpoints]}\n")

        # Warm-up Panda3D rendering pipeline (buffers, textures)
        for _ in range(5):
            self.graphicsEngine.renderFrame()

    # ──────────────────────────────────────────────────────────────────
    def _run_single_episode(self, model, tier_name, tier_cfg, ep_idx):
        """Run one episode, return episode summary dict."""
        self.env.init_vel_range = tier_cfg['vel']
        self.env.init_ang_range = tier_cfg['ang']
        self.env.stabilization_only = False

        obs, info = self.env.reset()

        # Apply target offset
        drone_pos = self.env.base_env.state[0:5:2].copy()
        angle = np.random.uniform(0, 2 * np.pi)
        off = tier_cfg['offset']
        dx = off * np.cos(angle)
        dy = off * np.sin(angle)
        self.env.target_pos = np.array([
            drone_pos[0] + dx,
            drone_pos[1] + dy,
            drone_pos[2] - self.env.hover_height,
        ])
        self.env._update_target_marker_pos()

        self.graphicsEngine.renderFrame()
        self.env._capture_camera_images(force_capture=True)
        state = self.env.base_env.state.astype(np.float32)
        obs = self.env._build_observation(state)

        # Warm-up
        neutral = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        for _ in range(10):
            obs, _, _, _, info = self.env.step(neutral)
            self.graphicsEngine.renderFrame()

        total_steps = self.args.duration * 100
        rewards = []
        visible_count = 0
        centering_dists = []
        fractions = []
        action_mags = []
        r_stab_list = []
        r_cent_list = []
        r_scale_list = []
        prev_action = None
        action_jerks = []
        terminated_early = False

        for step in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.graphicsEngine.renderFrame()

            rewards.append(reward)
            act_mag = float(np.mean(np.abs(action)))
            action_mags.append(act_mag)

            if prev_action is not None:
                action_jerks.append(
                    float(np.mean(np.abs(action - prev_action))))
            prev_action = action.copy()

            vt = info.get('visual_tracking', {})
            vis = vt.get('target_visible', False)
            if vis:
                visible_count += 1
                cd = vt.get('centering_dist', np.nan)
                centering_dists.append(cd)
                fractions.append(vt.get('target_fraction', 0.0))

            r_stab_list.append(vt.get('r_stability', 0.0))
            r_cent_list.append(vt.get('r_centering', 0.0))
            r_scale_list.append(vt.get('r_scale', 0.0))

            if terminated or truncated:
                terminated_early = True
                break

        _m = lambda lst: float(np.mean(lst)) if lst else 0.0

        return {
            'tier': tier_name,
            'episode': ep_idx,
            'steps': len(rewards),
            'terminated_early': terminated_early,
            'total_reward': round(sum(rewards), 2),
            'visibility_pct': round(
                100 * visible_count / max(len(rewards), 1), 1),
            'mean_centering_dist': round(_m(centering_dists), 4),
            'mean_fraction': round(_m(fractions), 5),
            'mean_action_mag': round(_m(action_mags), 4),
            'mean_action_jerk': round(_m(action_jerks), 4),
            'mean_r_stability': round(_m(r_stab_list), 4),
            'mean_r_centering': round(_m(r_cent_list), 4),
            'mean_r_scale': round(_m(r_scale_list), 4),
        }

    # ──────────────────────────────────────────────────────────────────
    def _evaluate_checkpoint(self, step_num, model_path):
        """Evaluate one checkpoint across all tiers."""
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_path.name}  ({step_num:,} steps)")
        print(f"{'='*60}")

        model = SAC.load(str(model_path), env=None)
        ep_per_tier = self.args.episodes_per_tier
        all_episodes = []
        ep_global = 0

        for tier_name in ('easy', 'medium', 'hard'):
            cfg = TIERS[tier_name]
            print(f"  -- {tier_name.upper()} "
                  f"(off={cfg['offset']}m vel={cfg['vel']} "
                  f"ang={cfg['ang']}) --")

            for i in range(ep_per_tier):
                ep_global += 1
                summary = self._run_single_episode(
                    model, tier_name, cfg, ep_global)
                all_episodes.append(summary)

                tag = "EARLY" if summary['terminated_early'] else "OK"
                print(f"    Ep {ep_global:2d}  "
                      f"R={summary['total_reward']:8.1f}  "
                      f"vis={summary['visibility_pct']:5.1f}%  "
                      f"cent={summary['mean_centering_dist']:.3f}  "
                      f"frac={summary['mean_fraction']:.4f}  [{tag}]")

        return all_episodes

    # ──────────────────────────────────────────────────────────────────
    def _agg(self, eps):
        """Aggregate episode stats."""
        if not eps:
            return {}
        keys = [
            'total_reward', 'visibility_pct', 'mean_centering_dist',
            'mean_fraction', 'mean_action_mag', 'mean_action_jerk',
            'mean_r_stability', 'mean_r_centering', 'mean_r_scale',
        ]
        agg = {}
        for k in keys:
            vals = [e[k] for e in eps]
            agg[k] = {
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
            }
        n = len(eps)
        agg['episodes'] = n
        early = sum(1 for e in eps if e['terminated_early'])
        agg['early_terminations'] = early
        agg['survival_rate_pct'] = round(100 * (1 - early / n), 1)
        return agg

    # ──────────────────────────────────────────────────────────────────
    def run_evaluation(self):
        """Main loop: evaluate all checkpoints sequentially."""
        all_results = {}
        all_episodes = []

        for step_num, model_path in self.checkpoints:
            episodes = self._evaluate_checkpoint(step_num, model_path)

            # Tag episodes with checkpoint step
            for ep in episodes:
                ep['checkpoint'] = step_num
            all_episodes.extend(episodes)

            result = {
                'global': self._agg(episodes),
                'tiers': {},
            }
            for tier in ('easy', 'medium', 'hard'):
                tier_eps = [e for e in episodes if e['tier'] == tier]
                result['tiers'][tier] = self._agg(tier_eps)

            all_results[step_num] = result

        # ── Save full results JSON ──
        output = {
            'duration_s': self.args.duration,
            'episodes_per_tier': self.args.episodes_per_tier,
            'checkpoints': {
                str(s): r for s, r in all_results.items()
            },
        }
        summary_path = self.output_dir / 'checkpoint_comparison.json'
        with open(summary_path, 'w') as f:
            json.dump(output, f, indent=2)

        # ── Save episodes CSV ──
        if all_episodes:
            csv_path = self.output_dir / 'checkpoint_episodes.csv'
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=all_episodes[0].keys())
                w.writeheader()
                w.writerows(all_episodes)

        # ── Print comparison table ──
        self._print_comparison(all_results)

        # ── Plots ──
        if not self.args.no_plots:
            self._generate_plots(all_results)

    # ──────────────────────────────────────────────────────────────────
    def _print_comparison(self, all_results):
        print(f"\n{'='*90}")
        print("  CHECKPOINT COMPARISON — HOVER-TRACK v3")
        print(f"{'='*90}")
        print(f"  {'Checkpoint':>12} | {'Surv%':>6} | {'Reward':>8} | "
              f"{'Vis%':>6} | {'Center':>7} | {'Frac':>7} | "
              f"{'R_stab':>7} | {'R_cent':>7} | {'R_scale':>7}")
        print(f"  {'-'*84}")

        for step_num in sorted(all_results.keys()):
            g = all_results[step_num]['global']
            print(f"  {step_num:>10,}k | "
                  f"{g['survival_rate_pct']:>5.1f}% | "
                  f"{g['total_reward']['mean']:>8.1f} | "
                  f"{g['visibility_pct']['mean']:>5.1f}% | "
                  f"{g['mean_centering_dist']['mean']:>7.3f} | "
                  f"{g['mean_fraction']['mean']:>7.4f} | "
                  f"{g['mean_r_stability']['mean']:>7.4f} | "
                  f"{g['mean_r_centering']['mean']:>7.4f} | "
                  f"{g['mean_r_scale']['mean']:>7.4f}")

        # Per-tier for best checkpoint
        best_step = max(all_results.keys(),
                        key=lambda s: all_results[s]['global']
                        ['total_reward']['mean'])
        best = all_results[best_step]
        print(f"\n  Best checkpoint: {best_step:,} steps")
        print(f"  {'Tier':<8} | {'Surv%':>6} | {'Reward':>8} | "
              f"{'Vis%':>6} | {'Center':>7} | {'Frac':>7}")
        print(f"  {'-'*52}")
        for tier in ('easy', 'medium', 'hard'):
            t = best['tiers'][tier]
            print(f"  {tier:<8} | "
                  f"{t['survival_rate_pct']:>5.1f}% | "
                  f"{t['total_reward']['mean']:>8.1f} | "
                  f"{t['visibility_pct']['mean']:>5.1f}% | "
                  f"{t['mean_centering_dist']['mean']:>7.3f} | "
                  f"{t['mean_fraction']['mean']:>7.4f}")

        print(f"\n  Saved to {self.output_dir}/checkpoint_comparison.json")
        print(f"{'='*90}\n")

    # ──────────────────────────────────────────────────────────────────
    def _generate_plots(self, all_results):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib not found — skipping plots)")
            return

        steps = sorted(all_results.keys())
        step_labels = [f'{s//1000}k' for s in steps]

        # ── 1. Global metrics across checkpoints ──
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        metrics = [
            ('total_reward', 'Total Reward'),
            ('visibility_pct', 'Visibility (%)'),
            ('mean_centering_dist', 'Centering Distance'),
            ('mean_fraction', 'Target Fraction'),
            ('mean_r_stability', 'R_stability'),
            ('mean_r_centering', 'R_centering'),
        ]
        for ax, (key, title) in zip(axes.flat, metrics):
            means = [all_results[s]['global'][key]['mean'] for s in steps]
            stds = [all_results[s]['global'][key]['std'] for s in steps]
            ax.errorbar(range(len(steps)), means, yerr=stds,
                        marker='o', capsize=4, linewidth=2)
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels(step_labels, rotation=45)
            ax.set_title(title)
            ax.grid(alpha=0.3)
        fig.suptitle('Checkpoint Comparison — Global Metrics', fontsize=14)
        fig.tight_layout()
        fig.savefig(str(self.output_dir / 'checkpoint_global.png'), dpi=150)
        plt.close(fig)

        # ── 2. Per-tier survival + reward ──
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        tier_colors = {'easy': '#4CAF50', 'medium': '#2196F3',
                       'hard': '#FF5722'}
        for tier in ('easy', 'medium', 'hard'):
            surv = [all_results[s]['tiers'][tier]['survival_rate_pct']
                    for s in steps]
            rew = [all_results[s]['tiers'][tier]['total_reward']['mean']
                   for s in steps]
            axes[0].plot(range(len(steps)), surv, marker='o',
                         label=tier.capitalize(), color=tier_colors[tier],
                         linewidth=2)
            axes[1].plot(range(len(steps)), rew, marker='o',
                         label=tier.capitalize(), color=tier_colors[tier],
                         linewidth=2)

        for ax, title in zip(axes, ['Survival Rate (%)', 'Mean Reward']):
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels(step_labels, rotation=45)
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle('Checkpoint Comparison — Per Tier', fontsize=14)
        fig.tight_layout()
        fig.savefig(str(self.output_dir / 'checkpoint_tiers.png'), dpi=150)
        plt.close(fig)

        print("  Plots saved.")


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = EvalV3App(args)
    app.run_evaluation()
