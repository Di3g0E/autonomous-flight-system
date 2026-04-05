#!/usr/bin/env python
"""
Pipeline evaluator for the hover-track v2 SAC model.

Runs N episodes across three difficulty tiers (matching the curriculum
phases used during training) and produces:

  1. telemetry.csv          — per-step raw data (every recorded step).
  2. episodes.csv           — per-episode aggregated statistics.
  3. summary.json           — global stats + per-tier breakdown.
  4. reward_components.png  — stacked reward decomposition per tier.
  5. centering_timeline.png — centering-dist over time, one line per episode.
  6. tier_boxplots.png      — box-plots comparing tiers.

Tiers
-----
  Easy   (×4 ep): offset 0.2 m, vel 0.10, ang 0.05
  Medium (×4 ep): offset 0.6 m, vel 0.25, ang 0.10
  Hard   (×4 ep): offset 1.0 m, vel 0.35, ang 0.15

Usage:
    python tests/evaluate_hover_track_v2.py
    python tests/evaluate_hover_track_v2.py --episodes-per-tier 8 --duration 10
    python tests/evaluate_hover_track_v2.py --model-path ./models/hover_track_v2/best_model.zip
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
MODEL_PATH = './models/hover_track_v2/best_model.zip'
DURATION_S = 5
EPISODES_PER_TIER = 4

TIERS = {
    'easy':   {'offset': 0.2, 'vel': 0.10, 'ang': 0.05},
    'medium': {'offset': 0.6, 'vel': 0.25, 'ang': 0.10},
    'hard':   {'offset': 1.0, 'vel': 0.35, 'ang': 0.15},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate hover-track v2 pipeline across difficulty tiers")
    p.add_argument('--model-path', type=str, default=MODEL_PATH)
    p.add_argument('--duration', type=int, default=DURATION_S,
                   help="Episode duration in seconds (default: 5)")
    p.add_argument('--episodes-per-tier', type=int, default=EPISODES_PER_TIER)
    p.add_argument('--no-display', action='store_true',
                   help="Minimise Panda3D window")
    p.add_argument('--no-plots', action='store_true',
                   help="Skip plot generation (avoids matplotlib dependency)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Panda3D App
# ══════════════════════════════════════════════════════════════════════

class EvalApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        self.output_dir = Path(project_root) / "experiments" / "hover_track_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 3D scene
        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)

        # FPV camera (downward)
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, 0.01)
        self.fpv_camera.cam.lookAt(0, 0, 0)
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
        )

        # Model
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model not found: {args.model_path}")
            sys.exit(1)
        self.model = SAC.load(args.model_path, env=None)
        print(f"Model loaded: {args.model_path}")

        self.taskMgr.doMethodLater(0.5, self._run_evaluation, 'eval')

    # ──────────────────────────────────────────────────────────────────
    def _run_single_episode(self, tier_name, tier_cfg, ep_idx):
        """Run one episode and return (step_rows, episode_summary)."""

        # Apply tier init ranges
        self.env.init_vel_range = tier_cfg['vel']
        self.env.init_ang_range = tier_cfg['ang']

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

        # Recapture after moving target
        self.graphicsEngine.renderFrame()
        self.env._capture_camera_images(force_capture=True)
        state = self.env.base_env.state.astype(np.float32)
        obs = self.env._build_observation(state)

        # Warm-up
        neutral = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        for _ in range(10):
            obs, _, _, _, info = self.env.step(neutral)
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()

        total_steps = self.args.duration * 100
        step_rows = []
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
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.taskMgr.step()

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
            cent_d = vt.get('centering_dist', np.nan)
            frac = vt.get('target_fraction', 0.0)
            r_stab = vt.get('r_stability', 0.0)
            r_cent = vt.get('r_centering', 0.0)
            r_scl = vt.get('r_scale', 0.0)

            if vis:
                centering_dists.append(cent_d)
                fractions.append(frac)
            r_stab_list.append(r_stab)
            r_cent_list.append(r_cent)
            r_scale_list.append(r_scl)

            # Drone position from state
            st = self.env.base_env.state
            step_rows.append({
                'tier': tier_name,
                'episode': ep_idx,
                'step': step,
                'drone_x': float(st[0]),
                'drone_vx': float(st[1]),
                'drone_y': float(st[2]),
                'drone_vy': float(st[3]),
                'drone_z': float(st[4]),
                'drone_vz': float(st[5]),
                'target_visible': int(vis),
                'centering_dist': round(cent_d, 4) if vis else '',
                'fraction': round(frac, 5),
                'r_stability': round(r_stab, 4),
                'r_centering': round(r_cent, 4),
                'r_scale': round(r_scl, 4),
                'reward': round(float(reward), 4),
                'action_mag': round(act_mag, 4),
            })

            if terminated or truncated:
                terminated_early = True
                break

        # Aggregates
        _m = lambda lst: float(np.mean(lst)) if lst else 0.0
        _s = lambda lst: float(np.std(lst)) if len(lst) > 1 else 0.0
        vis_pct = 100 * visible_count / max(len(rewards), 1)

        summary = {
            'tier': tier_name,
            'episode': ep_idx,
            'steps': len(rewards),
            'terminated_early': terminated_early,
            'target_offset': off,
            'init_vel_range': tier_cfg['vel'],
            'init_ang_range': tier_cfg['ang'],
            'total_reward': round(sum(rewards), 2),
            'mean_reward': round(_m(rewards), 4),
            'std_reward': round(_s(rewards), 4),
            'visibility_pct': round(vis_pct, 1),
            'mean_centering_dist': round(_m(centering_dists), 4),
            'std_centering_dist': round(_s(centering_dists), 4),
            'mean_fraction': round(_m(fractions), 5),
            'std_fraction': round(_s(fractions), 5),
            'mean_action_mag': round(_m(action_mags), 4),
            'mean_action_jerk': round(_m(action_jerks), 4),
            'mean_r_stability': round(_m(r_stab_list), 4),
            'mean_r_centering': round(_m(r_cent_list), 4),
            'mean_r_scale': round(_m(r_scale_list), 4),
        }
        return step_rows, summary

    # ──────────────────────────────────────────────────────────────────
    def _run_evaluation(self, task):
        ep_per_tier = self.args.episodes_per_tier
        total_ep = ep_per_tier * len(TIERS)
        print(f"\nRunning {total_ep} episodes "
              f"({ep_per_tier} per tier × {len(TIERS)} tiers)...\n")

        all_steps = []
        all_episodes = []
        ep_global = 0

        for tier_name in ('easy', 'medium', 'hard'):
            cfg = TIERS[tier_name]
            print(f"  ── Tier: {tier_name.upper()}  "
                  f"(offset={cfg['offset']}m  vel={cfg['vel']}  "
                  f"ang={cfg['ang']}) ──")

            for i in range(ep_per_tier):
                ep_global += 1
                rows, summary = self._run_single_episode(
                    tier_name, cfg, ep_global)
                all_steps.extend(rows)
                all_episodes.append(summary)

                tag = "EARLY" if summary['terminated_early'] else "OK"
                print(f"    Ep {ep_global:2d}/{total_ep}  "
                      f"R={summary['total_reward']:7.1f}  "
                      f"vis={summary['visibility_pct']:5.1f}%  "
                      f"cent={summary['mean_centering_dist']:.3f}  "
                      f"frac={summary['mean_fraction']:.4f}  "
                      f"[{tag}]")

        # ── Write telemetry CSV ──
        telem_path = self.output_dir / 'telemetry.csv'
        if all_steps:
            with open(telem_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=all_steps[0].keys())
                w.writeheader()
                w.writerows(all_steps)

        # ── Write episodes CSV ──
        ep_path = self.output_dir / 'episodes.csv'
        with open(ep_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_episodes[0].keys())
            w.writeheader()
            w.writerows(all_episodes)

        # ── Build summary JSON ──
        summary = self._build_summary(all_episodes)
        summary_path = self.output_dir / 'evaluation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # ── Print report ──
        self._print_report(summary)

        # ── Plots ──
        if not self.args.no_plots:
            self._generate_plots(all_episodes, all_steps)

        self.userExit()
        return task.done

    # ──────────────────────────────────────────────────────────────────
    def _build_summary(self, episodes):
        """Compute global + per-tier statistics."""

        def _agg(eps):
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
                    'min': round(float(np.min(vals)), 4),
                    'max': round(float(np.max(vals)), 4),
                }
            n = len(eps)
            agg['episodes'] = n
            agg['early_terminations'] = sum(
                1 for e in eps if e['terminated_early'])
            agg['survival_rate_pct'] = round(
                100 * (1 - agg['early_terminations'] / n), 1)
            return agg

        summary = {
            'model': self.args.model_path,
            'duration_s': self.args.duration,
            'episodes_per_tier': self.args.episodes_per_tier,
            'global': _agg(episodes),
            'tiers': {},
        }
        for tier_name in ('easy', 'medium', 'hard'):
            tier_eps = [e for e in episodes if e['tier'] == tier_name]
            summary['tiers'][tier_name] = _agg(tier_eps)

        return summary

    # ──────────────────────────────────────────────────────────────────
    def _print_report(self, summary):
        g = summary['global']
        print(f"\n{'='*70}")
        print(f"  HOVER-TRACK v2 — PIPELINE EVALUATION REPORT")
        print(f"{'='*70}")
        print(f"  Model:      {summary['model']}")
        print(f"  Duration:   {summary['duration_s']}s per episode")
        print(f"  Episodes:   {g['episodes']} total "
              f"({summary['episodes_per_tier']} per tier)")
        print(f"  Survival:   {g['survival_rate_pct']}% "
              f"({g['early_terminations']} early terminations)")

        print(f"\n  {'Metric':<24} {'Mean':>8} {'Std':>8} "
              f"{'Min':>8} {'Max':>8}")
        print(f"  {'-'*56}")
        rows = [
            ('Total reward',        'total_reward'),
            ('Visibility (%)',      'visibility_pct'),
            ('Centering dist',     'mean_centering_dist'),
            ('Target fraction',     'mean_fraction'),
            ('Action magnitude',    'mean_action_mag'),
            ('Action jerk',         'mean_action_jerk'),
            ('R_stability',         'mean_r_stability'),
            ('R_centering',         'mean_r_centering'),
            ('R_scale',             'mean_r_scale'),
        ]
        for label, key in rows:
            v = g[key]
            print(f"  {label:<24} {v['mean']:>8.3f} {v['std']:>8.3f} "
                  f"{v['min']:>8.3f} {v['max']:>8.3f}")

        print(f"\n  Per-tier breakdown:")
        print(f"  {'Tier':<8} {'Reward':>8} {'Vis%':>6} "
              f"{'Center':>8} {'Frac':>8} {'Surv%':>6}")
        print(f"  {'-'*46}")
        for tier in ('easy', 'medium', 'hard'):
            t = summary['tiers'][tier]
            print(f"  {tier:<8} "
                  f"{t['total_reward']['mean']:>8.1f} "
                  f"{t['visibility_pct']['mean']:>5.1f}% "
                  f"{t['mean_centering_dist']['mean']:>8.3f} "
                  f"{t['mean_fraction']['mean']:>8.4f} "
                  f"{t['survival_rate_pct']:>5.1f}%")

        print(f"\n  Saved to {self.output_dir}/")
        print(f"    telemetry.csv            — per-step raw data")
        print(f"    episodes.csv             — per-episode stats")
        print(f"    evaluation_summary.json  — full statistics")
        if not self.args.no_plots:
            print(f"    reward_components.png    — reward decomposition")
            print(f"    centering_timeline.png   — centering over time")
            print(f"    tier_boxplots.png        — tier comparison")
        print(f"{'='*70}\n")

    # ──────────────────────────────────────────────────────────────────
    def _generate_plots(self, episodes, steps):
        """Create evaluation plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib not found — skipping plots)")
            return

        out = self.output_dir

        # ── 1. Reward components bar chart per tier ──
        fig, ax = plt.subplots(figsize=(8, 5))
        tiers = ['easy', 'medium', 'hard']
        comps = ['mean_r_stability', 'mean_r_centering', 'mean_r_scale']
        labels = ['Stability', 'Centering', 'Scale']
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        x = np.arange(len(tiers))
        width = 0.22
        for i, (comp, label, color) in enumerate(zip(comps, labels, colors)):
            vals = []
            errs = []
            for tier in tiers:
                tier_eps = [e for e in episodes if e['tier'] == tier]
                v = [e[comp] for e in tier_eps]
                vals.append(np.mean(v))
                errs.append(np.std(v))
            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=label, color=color, capsize=3)
        ax.set_xticks(x + width)
        ax.set_xticklabels([t.capitalize() for t in tiers])
        ax.set_ylabel('Mean reward component')
        ax.set_title('Reward Components by Difficulty Tier')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out / 'reward_components.png'), dpi=150)
        plt.close(fig)

        # ── 2. Centering distance timeline ──
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, tier in zip(axes, tiers):
            tier_steps = [s for s in steps if s['tier'] == tier]
            ep_ids = sorted(set(s['episode'] for s in tier_steps))
            for ep_id in ep_ids:
                ep_s = [s for s in tier_steps if s['episode'] == ep_id]
                t = [s['step'] for s in ep_s]
                cd = [s['centering_dist'] if s['centering_dist'] != '' else np.nan
                      for s in ep_s]
                ax.plot(t, cd, alpha=0.6, linewidth=0.8)
            ax.set_title(f'{tier.capitalize()}')
            ax.set_xlabel('Step')
            ax.set_ylim(0, 1.5)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel('Centering distance')
        fig.suptitle('Centering Distance Over Time', fontsize=13)
        fig.tight_layout()
        fig.savefig(str(out / 'centering_timeline.png'), dpi=150)
        plt.close(fig)

        # ── 3. Tier box-plots ──
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        metrics = [
            ('total_reward', 'Total Reward'),
            ('visibility_pct', 'Visibility (%)'),
            ('mean_centering_dist', 'Centering Dist'),
            ('mean_action_jerk', 'Action Jerk'),
        ]
        for ax, (key, title) in zip(axes, metrics):
            data = []
            for tier in tiers:
                tier_eps = [e for e in episodes if e['tier'] == tier]
                data.append([e[key] for e in tier_eps])
            bp = ax.boxplot(data, tick_labels=[t.capitalize() for t in tiers],
                            patch_artist=True)
            tier_colors = ['#81C784', '#64B5F6', '#FF8A65']
            for patch, color in zip(bp['boxes'], tier_colors):
                patch.set_facecolor(color)
            ax.set_title(title)
            ax.grid(axis='y', alpha=0.3)
        fig.suptitle('Performance Distribution by Tier', fontsize=13)
        fig.tight_layout()
        fig.savefig(str(out / 'tier_boxplots.png'), dpi=150)
        plt.close(fig)

        print("  Plots saved.")


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = EvalApp(args)
    app.run()
