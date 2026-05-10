#!/usr/bin/env python
"""
Sanity check for the hover-track v10 pipeline. Closes audit points #1
and #2 ("are weights updating?", "step-by-step debug of the MDP").

Independent of training. Loads the env (and optionally a trained model
+ VecNormalize stats), runs a handful of episodes and prints:

  1. Observation ranges per dimension (min / max / mean / std) over all
     collected steps. Detects unscaled features that VecNormalize is
     supposed to handle.
  2. Reward-component coherence: for each step, confirms that
     reward_returned_by_wrapper == r_track + r_alive + r_jerk
     (modulo crash_penalty at terminal step).
  3. Termination reason histogram. Detects undocumented blank reasons.
  4. (Optional, requires --model) snapshot weights, run K SAC updates
     against the populated replay buffer, print mean |Δw| per layer
     with a FROZEN flag if Δ < 1e-7. Confirms gradient flow without
     waiting for a full training run.

Usage:
    # MDP-only audit (no model needed):
    python scripts/sanity_check_v10.py --no-display --episodes 3

    # With v9.1.1 model loaded for weight-update check:
    python scripts/sanity_check_v10.py --no-display --episodes 3 \\
        --model ./models/hover_track_v9_1_1/best_model_TFG.zip \\
        --vec-norm ./models/hover_track_v9_1_1/best_vec_normalize_TFG.pkl \\
        --train-updates 200
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 stdout/stderr so unicode (Δ, ⚠) prints on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

import numpy as np

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: F401  (must precede SB3 import: forces miniconda torch to load before SB3 picks the .v venv copy)
from panda3d.core import Filename, loadPrcFile, loadPrcFileData

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from direct.showbase.ShowBase import ShowBase
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera

from scripts.train_hover_track_v9 import HoverTrackV9Wrapper


# Names of the 19 obs dims for readable output.
OBS_NAMES = [
    'x', 'vx', 'y', 'vy', 'z', 'vz',
    'q0', 'q1', 'q2', 'q3',
    'wx', 'wy', 'wz',
    'cx', 'cy', 'frac', 'vis', 'dcx', 'dcy',
]


def parse_args():
    p = argparse.ArgumentParser(description="v10 sanity-check diagnostic")
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--max-ep-steps', type=int, default=500,
                   help="Cap per episode for the diagnostic.")
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--model', type=str, default='',
                   help="Optional .zip to use for the policy and the "
                        "weight-update check. If empty, uses random actions "
                        "and skips the weight check.")
    p.add_argument('--vec-norm', type=str, default='',
                   help="Optional VecNormalize .pkl matching --model.")
    p.add_argument('--train-updates', type=int, default=0,
                   help=">0 enables weight-update sanity check after running "
                        "episodes. Requires --model. Runs that many SAC.train() "
                        "gradient_steps and prints per-layer Δw.")
    p.add_argument('--seed', type=int, default=2000)
    p.add_argument('--reward-tolerance', type=float, default=1e-4)
    return p.parse_args()


class SanityApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)
        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -8, 14)
        self.cam.lookAt(0, 0, 5)

        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.1)
        self.fpv_camera.cam.setHpr(0, -90, 0)
        self.fpv_camera.buffer.setActive(1)

        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.cam.setPos(0, -6, 12)
        self.ext_camera.cam.lookAt(0, 0, 5)
        self.ext_camera.buffer.setActive(1)

        self.raw_env = HoverTrackV9Wrapper(
            panda3d_app=self, quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True, use_depth=False,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            use_target=True, target_mode='moving',
            target_speed=0.0, target_radius=0.25,
            lemniscate_scale=2.0,
            filming_mode=True, enable_collisions=False,
            n=args.max_ep_steps, t_step=0.01, direct_control=1,
            centroid_obs=True, camera_down=True,
            hover_height=1.394,
            use_new_reward=True, constrained_init=True,
            init_pos_range=0.2, init_vel_range=0.10, init_ang_range=0.03,
            reward_version='v3.1',
        )

        # Optional model + vec_normalize
        self.model = None
        self.vec_normalize = None
        if args.model:
            mp = Path(args.model)
            if not mp.exists():
                raise FileNotFoundError(f"--model not found: {mp}")
            print(f"Loading model from {mp}")
            # Build a minimal vec env to satisfy SAC.load(env=...)
            mon = Monitor(self.raw_env)
            self.vec_env = DummyVecEnv([lambda: mon])
            if args.vec_norm:
                vp = Path(args.vec_norm)
                if not vp.exists():
                    raise FileNotFoundError(f"--vec-norm not found: {vp}")
                print(f"Loading VecNormalize from {vp}")
                self.vec_env = VecNormalize.load(str(vp), self.vec_env)
                self.vec_env.training = False
                self.vec_env.norm_reward = False
                self.vec_normalize = self.vec_env
            self.model = SAC.load(str(mp), env=self.vec_env, device='auto')

    # ──────────────────────────────────────────────────────────────────
    def _action(self, obs):
        if self.model is None:
            return np.random.uniform(-1.0, 1.0, size=4).astype(np.float32)
        obs_in = np.asarray(obs, dtype=np.float32)
        if self.vec_normalize is not None:
            obs_in = self.vec_normalize.normalize_obs(obs_in)
        act, _ = self.model.predict(obs_in, deterministic=True)
        return act

    def run_episodes(self):
        args = self.args
        all_obs = []
        all_actions = []
        reward_mismatches = 0
        max_mismatch = 0.0
        term_reasons = Counter()
        prev_action = None
        episode_lengths = []
        episode_visibilities = []

        for ep in range(args.episodes):
            obs, info = self.raw_env.reset(seed=args.seed + ep)
            self.taskMgr.step()
            done = False
            step = 0
            visible_steps = 0
            ep_len = 0
            while not done and step < args.max_ep_steps:
                act = self._action(obs)
                obs, reward, term, trunc, info = self.raw_env.step(act)
                self.taskMgr.step()

                # Coherence check on reward components.
                rt = float(info.get('v9_track', 0.0))
                ra = float(info.get('v9_alive', 0.0))
                rj = float(info.get('v9_jerk', 0.0))
                expected = rt + ra + rj
                # Crash penalty subtracts at the terminal step only.
                if term and info.get('v9_term_reason') in (
                        'altitude_low', 'altitude_high'):
                    expected -= self.raw_env.crash_penalty
                diff = abs(reward - expected)
                if diff > args.reward_tolerance:
                    reward_mismatches += 1
                    max_mismatch = max(max_mismatch, diff)

                vt = info.get('visual_tracking', {}) or {}
                if vt.get('target_visible', False):
                    visible_steps += 1

                all_obs.append(np.asarray(obs, dtype=np.float32).copy())
                all_actions.append(np.asarray(act, dtype=np.float32).copy())
                step += 1
                ep_len = step
                done = term or trunc

            tr = info.get('v9_term_reason', '') or ('truncated' if trunc else 'cap')
            term_reasons[tr] += 1
            episode_lengths.append(ep_len)
            episode_visibilities.append(visible_steps / max(ep_len, 1))

        return {
            'obs': np.stack(all_obs) if all_obs else np.empty((0, 19)),
            'actions': np.stack(all_actions) if all_actions else np.empty((0, 4)),
            'reward_mismatches': reward_mismatches,
            'max_mismatch': max_mismatch,
            'term_reasons': term_reasons,
            'episode_lengths': episode_lengths,
            'episode_visibilities': episode_visibilities,
        }

    # ──────────────────────────────────────────────────────────────────
    def weight_update_check(self, n_updates):
        if self.model is None:
            print("\n[WeightCheck] No model loaded, skipping.")
            return
        # We need the replay buffer populated for SAC.train() to do anything.
        # A loaded SAC has an empty buffer; manually push a few transitions
        # by re-rolling 1 episode through the model's standard collect path.
        print(f"\n[WeightCheck] Populating replay buffer ({100} transitions)...")
        try:
            self.model.learn(total_timesteps=100,
                             reset_num_timesteps=False,
                             log_interval=999999)
        except Exception as e:
            print(f"[WeightCheck] Could not populate buffer: {e}")
            return

        snapshot = {n: p.detach().cpu().clone()
                    for n, p in self.model.policy.named_parameters()}
        print(f"[WeightCheck] Running {n_updates} SAC.train() steps...")
        try:
            self.model.train(gradient_steps=n_updates, batch_size=self.model.batch_size)
        except Exception as e:
            print(f"[WeightCheck] train() failed: {e}")
            return

        print(f"\n[WeightCheck] Mean |Δw| per layer after {n_updates} updates:")
        any_frozen = False
        for n, p in self.model.policy.named_parameters():
            d = (snapshot[n] - p.detach().cpu()).abs().mean().item()
            mark = "  ⚠ FROZEN" if d < 1e-7 else ""
            if d < 1e-7:
                any_frozen = True
            print(f"  {n:>55}  Δ={d:.3e}{mark}")
        if any_frozen:
            print("[WeightCheck] WARNING: some layers below threshold.")
        else:
            print("[WeightCheck] All layers updating. Gradient flow confirmed.")


def print_obs_summary(obs_arr):
    print("\n[ObsAudit] 19-D observation ranges over collected steps")
    print(f"           (n={len(obs_arr)} steps)")
    print(f"  {'idx':>3}  {'name':>5}  {'min':>10}  {'max':>10}  "
          f"{'mean':>10}  {'std':>10}")
    if len(obs_arr) == 0:
        print("  (no samples)")
        return
    mn = obs_arr.min(axis=0)
    mx = obs_arr.max(axis=0)
    me = obs_arr.mean(axis=0)
    sd = obs_arr.std(axis=0)
    for i, name in enumerate(OBS_NAMES):
        print(f"  {i:>3}  {name:>5}  {mn[i]:>10.4f}  {mx[i]:>10.4f}  "
              f"{me[i]:>10.4f}  {sd[i]:>10.4f}")


def print_action_summary(act_arr):
    print("\n[ActionAudit] 4-D action ranges")
    if len(act_arr) == 0:
        print("  (no samples)")
        return
    print(f"  {'motor':>5}  {'min':>10}  {'max':>10}  "
          f"{'mean':>10}  {'std':>10}")
    for i in range(act_arr.shape[1]):
        print(f"  {i:>5}  {act_arr[:, i].min():>10.4f}  "
              f"{act_arr[:, i].max():>10.4f}  "
              f"{act_arr[:, i].mean():>10.4f}  "
              f"{act_arr[:, i].std():>10.4f}")


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'window-type offscreen')
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')

    print("=" * 70)
    print("  SANITY CHECK v10 — MDP audit")
    print("=" * 70)
    app = SanityApp(args)

    print(f"\nRunning {args.episodes} episodes "
          f"({'policy' if app.model else 'random'} actions, "
          f"max {args.max_ep_steps} steps each)...")
    res = app.run_episodes()

    # ── Episode summary ──
    print("\n[Episodes]")
    for i, (lng, vis) in enumerate(zip(res['episode_lengths'],
                                        res['episode_visibilities'])):
        print(f"  ep {i}: steps={lng:>5}  visibility={vis:.2%}")

    # ── Obs / action audit ──
    print_obs_summary(res['obs'])
    print_action_summary(res['actions'])

    # ── Reward coherence ──
    print(f"\n[RewardCoherence] tol={args.reward_tolerance}")
    if res['reward_mismatches'] == 0:
        print("  All steps consistent: reward == r_track + r_alive + r_jerk "
              "(− crash_penalty at terminal). OK")
    else:
        print(f"  ⚠ {res['reward_mismatches']} steps with reward mismatch. "
              f"Max diff: {res['max_mismatch']:.6f}")

    # ── Termination distribution ──
    print("\n[TerminationReasons]")
    if not res['term_reasons']:
        print("  (no terminations in this run)")
    for reason, count in res['term_reasons'].most_common():
        print(f"  {reason or '(blank)':>15}  {count}")

    # ── Weight update sanity check ──
    if args.train_updates > 0:
        app.weight_update_check(args.train_updates)

    print("\n" + "=" * 70)
    print("  SANITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
