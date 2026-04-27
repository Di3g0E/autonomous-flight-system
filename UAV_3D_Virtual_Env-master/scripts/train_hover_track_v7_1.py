#!/usr/bin/env python
"""
Hover-track v7.1 — survival-first reward (post-mortem fixes from v7).

The v7 run collapsed at the Phase A → B transition (~step 40k):
  • Episodes shrunk from ~150 to ~16 steps in 15k steps.
  • Reward drifted toward 0 not by improvement but by *episode brevity*:
    short episodes accumulated less invisibility / jerk cost than long
    ones — a textbook reward-hacking local optimum.
  • Visibility stayed at 0% across all evaluations because the policy
    never learnt to maintain visual lock; it learned to die early.

v7.1 targeted fixes:
  1. **w_alive = 0.50** (was 0.10). Long-vs-short episode reward gap
     widens 5×, eliminating the "die early" local optimum.
     A 3000-step episode ⇒ +1500; a 16-step episode ⇒ +8.
  2. **Phase A jitter ramp**: jitter_xy now grows 0.05 → 0.30
     gradually inside Phase A instead of jumping at the B boundary.
     Phase B onward keeps 0.30 fixed.
  3. **Phase A extended 20% → 35%** (40k → 70k steps). The other
     phases compress proportionally:
       A [0–35%)   B [35–50%)   C [50–75%)   D [75–100%]
     This gives the agent more time to consolidate hover under the
     full jitter range before the target starts moving.
  4. **invisible_term_steps 200 → 60** (2.0 s → 0.6 s). Reduces the
     accumulated invisibility penalty per episode, so a brief loss
     of lock is no longer a stronger signal to die than to recover.

Inherits all v7 design choices: deterministic spawn above target,
α-ramped tracking weight, soft altitude termination, VecFrameStack(3),
lex best by (survival, visibility, -jerk), early-stop on regression.

Usage:
    python scripts/train_hover_track_v7_1.py --no-display
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: F401
from panda3d.core import Filename, loadPrcFile, loadPrcFileData

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from direct.showbase.ShowBase import ShowBase
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.utils.episode_recorder import EpisodeRecorder


# ══════════════════════════════════════════════════════════════════════
# v7 wrapper — deterministic spawn + survival-first reward
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV7Wrapper(Panda3DQuadrotorEnv):
    """
    Reset:
      Drone is spawned at (target_xy + jitter_xy, hover_height + jitter_z)
      with a yaw jitter. The sphere is therefore in view by construction.

    Step (v7 reward, on top of base env):
        r_alive       = +0.1                      (dense survival reward)
        r_track       = α(t) * r_centering        if r_stability > 0.3
                      = 0                         otherwise
        r_invisible   = -0.05                     if target not visible
        r_jerk        = -0.4 * ||Δaction||²
      The base env's reward is REPLACED — we don't add on top of v3.1.
    """

    def __init__(self, *args, jitter_xy=0.30, jitter_z=0.10,
                 jitter_yaw_deg=15.0,
                 alpha_min=0.2, alpha_max=1.0, alpha_warmup_steps=80_000,
                 w_jerk=0.4, w_alive=0.1, w_invisible=0.05,
                 stability_gate=0.3,
                 alt_soft_limit=1.5, alt_soft_steps=100,
                 alt_term_penalty=5.0,
                 invisible_term_steps=200, invisible_term_penalty=2.0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Spawn jitter
        self.jitter_xy = jitter_xy
        self.jitter_z = jitter_z
        self.jitter_yaw_rad = np.deg2rad(jitter_yaw_deg)

        # Reward weights
        self.w_alive = w_alive
        self.w_invisible = w_invisible
        self.w_jerk = w_jerk
        self.stability_gate = stability_gate

        # α schedule (controlled externally by callback)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_warmup_steps = alpha_warmup_steps
        self.current_alpha = alpha_min

        # Soft termination thresholds
        self.alt_soft_limit = alt_soft_limit
        self.alt_soft_steps = alt_soft_steps
        self.alt_term_penalty = alt_term_penalty
        self.invisible_term_steps = invisible_term_steps
        self.invisible_term_penalty = invisible_term_penalty

        # Episode counters
        self._alt_violation_streak = 0
        self._invisible_streak = 0
        self._prev_action_v7 = None

        # Curriculum-controlled (set by callback)
        self.target_speed_range = (0.0, 0.0)

        # min_start_distance unused — drone spawns above target deterministically
        self.min_start_distance = 0.0

    # ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        # Sample episode target speed from curriculum range
        self.target_speed = float(np.random.uniform(*self.target_speed_range))

        obs, info = super().reset(seed=seed, options=options)

        # ── Deterministic spawn above target ──
        state = self.base_env.state.copy()

        dx = float(np.random.uniform(-self.jitter_xy, self.jitter_xy))
        dy = float(np.random.uniform(-self.jitter_xy, self.jitter_xy))
        dz = float(np.random.uniform(-self.jitter_z, self.jitter_z))

        state[0] = self.target_pos[0] + dx
        state[2] = self.target_pos[1] + dy
        state[4] = self.target_pos[2] + self.hover_height + dz

        # Small velocity perturbation (not curriculum-scaled — keep it bounded)
        state[1] = float(np.random.uniform(-0.05, 0.05))
        state[3] = float(np.random.uniform(-0.05, 0.05))
        state[5] = float(np.random.uniform(-0.02, 0.02))

        # Yaw jitter via quaternion: leave roll/pitch ≈ 0, randomise yaw only
        yaw = float(np.random.uniform(-self.jitter_yaw_rad,
                                      self.jitter_yaw_rad))
        state[6] = np.cos(yaw / 2.0)   # q0
        state[7] = 0.0                  # q1
        state[8] = 0.0                  # q2
        state[9] = np.sin(yaw / 2.0)   # q3
        state[10:13] = 0.0              # angular velocities

        self.base_env.state = state.copy()
        self.base_env.previous_state = state.copy()

        self._update_visualization()
        if self.panda3d_app:
            self.panda3d_app.graphicsEngine.renderFrame()
        if self.use_camera:
            self._capture_camera_images(force_capture=True)

        obs = self._build_observation(state.astype(np.float32))

        # Reset episode counters
        self._alt_violation_streak = 0
        self._invisible_streak = 0
        self._prev_action_v7 = None
        return obs, info

    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        # ── Robust physics step ──
        try:
            obs, _base_reward, terminated, truncated, info = super().step(action)
        except RuntimeError as e:
            # solve_ivp divergence — terminate the episode cleanly
            obs = self._build_observation(self.base_env.state.astype(np.float32))
            info = {'physics_diverged': str(e), 'target_visible': False,
                    'r_stability': 0.0, 'r_centering': 0.0}
            return obs, -10.0, True, False, info

        # ── v7 reward (replaces base) ──
        r_stab = float(info.get('r_stability', 0.0))
        r_centering = float(info.get('r_centering', 0.0))
        target_visible = bool(info.get('target_visible', False))

        r_alive = self.w_alive

        if r_stab > self.stability_gate and target_visible:
            r_track = self.current_alpha * r_centering
        else:
            r_track = 0.0

        r_invisible = -self.w_invisible if not target_visible else 0.0

        if self._prev_action_v7 is not None:
            delta = float(np.linalg.norm(
                np.asarray(action) - self._prev_action_v7))
            r_jerk = -self.w_jerk * delta ** 2
        else:
            r_jerk = 0.0
        self._prev_action_v7 = np.asarray(action, dtype=np.float32).copy()

        reward = r_alive + r_track + r_invisible + r_jerk

        # ── Soft altitude termination ──
        drone_z = float(self.base_env.state[4])
        ideal_z = float(self.target_pos[2]) + self.hover_height
        if abs(drone_z - ideal_z) > self.alt_soft_limit:
            self._alt_violation_streak += 1
        else:
            self._alt_violation_streak = 0
        if self._alt_violation_streak >= self.alt_soft_steps:
            terminated = True
            reward -= self.alt_term_penalty
            info['v7_term_reason'] = 'altitude_drift'

        # ── Visibility-loss truncation (2 s) ──
        if not target_visible:
            self._invisible_streak += 1
        else:
            self._invisible_streak = 0
        if self._invisible_streak >= self.invisible_term_steps:
            truncated = True
            reward -= self.invisible_term_penalty
            info['v7_term_reason'] = 'lost_target'

        # Expose components for logging
        info['v7_alive'] = r_alive
        info['v7_track'] = r_track
        info['v7_invisible'] = r_invisible
        info['v7_jerk'] = r_jerk
        info['v7_alpha'] = self.current_alpha
        info['v7_alt_streak'] = self._alt_violation_streak
        info['v7_invis_streak'] = self._invisible_streak

        return obs, reward, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════
# Panda3D render callback — drives the Panda3D task manager each step
# ══════════════════════════════════════════════════════════════════════

class Panda3DRenderCallback(BaseCallback):
    def __init__(self, app, verbose=0):
        super().__init__(verbose)
        self.app = app

    def _on_step(self):
        self.app.taskMgr.step()
        return True


# ══════════════════════════════════════════════════════════════════════
# 4-phase curriculum + α schedule + per-episode metrics CSV
# ══════════════════════════════════════════════════════════════════════

class CurriculumV7Callback(BaseCallback):
    """
    Phases (fractions of total_timesteps):
      A [0.00, 0.20)  hover         speed 0          jitter_xy 0.10
      B [0.20, 0.40)  static FOV    speed 0          jitter_xy 0.30
      C [0.40, 0.70)  slow lemnisc. speed 0.05→0.10  jitter_xy 0.30
      D [0.70, 1.00]  med lemnisc.  speed 0.10→0.20  jitter_xy 0.30

    Smooth transitions: in the last 20% of each phase, with probability
    p (0 → 1 linearly) we sample from the next phase's parameters.

    α schedule (independent of phases): linear 0.2 → 1.0 over the first
    `alpha_warmup_steps` steps.
    """

    CSV_HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'visibility_pct', 'survived',
        'mean_action_mag', 'mean_action_jerk',
        'r_alive_sum', 'r_track_sum', 'r_invis_sum', 'r_jerk_sum',
        'phase', 'alpha', 'target_speed', 'jitter_xy',
        'term_reason',
    ]

    # v7.1: Phase A extended (35%), jitter ramps inside Phase A (see picker).
    PHASES = [
        ('A', 0.00, 0.35, (0.0, 0.0),     None),  # jitter ramp 0.05→0.30
        ('B', 0.35, 0.50, (0.0, 0.0),     0.30),
        ('C', 0.50, 0.75, (0.05, 0.10),   0.30),
        ('D', 0.75, 1.00, (0.10, 0.20),   0.30),
    ]
    # Phase A jitter ramp endpoints
    PHASE_A_JITTER_START = 0.05
    PHASE_A_JITTER_END = 0.30

    def __init__(self, raw_env, output_dir, total_timesteps,
                 alpha_min=0.2, alpha_max=1.0, alpha_warmup_steps=80_000,
                 verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_timesteps = total_timesteps
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_warmup_steps = alpha_warmup_steps

        self.episode_count = 0
        self.episode_rewards = deque(maxlen=50)

        self.current_phase = 'A'
        self._last_phase = 'A'

        self._reset_accum()
        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        self.start_time = time.time()
        self.last_log_time = time.time()

    def _reset_accum(self):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_visible = 0
        self._ep_action_mags = []
        self._ep_action_jerks = []
        self._ep_alive = 0.0
        self._ep_track = 0.0
        self._ep_invis = 0.0
        self._ep_jerk = 0.0
        self._ep_term_reason = ''
        self._prev_action = None

    def _on_training_start(self):
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADERS)

    def _pick_phase_params(self, progress):
        # Find current phase by progress
        for i, (name, start, end, speed_rng, jit_xy) in enumerate(self.PHASES):
            if start <= progress < end:
                phase_span = end - start
                p_in = (progress - start) / phase_span  # 0 → 1 inside phase

                # Phase A: jitter_xy ramps 0.05 → 0.30 across the whole phase
                if name == 'A':
                    jit_xy = (self.PHASE_A_JITTER_START
                              + p_in * (self.PHASE_A_JITTER_END
                                        - self.PHASE_A_JITTER_START))

                # Smooth transition: in last 20% of the phase, mix with next
                tail_start = end - 0.20 * phase_span
                if (progress >= tail_start and i + 1 < len(self.PHASES)
                        and np.random.random()
                        < (progress - tail_start) / (end - tail_start)):
                    nxt = self.PHASES[i + 1]
                    nxt_jit = nxt[4] if nxt[4] is not None else self.PHASE_A_JITTER_END
                    return name, nxt[3], nxt_jit  # current name, next params

                # Phase C/D: ramp speed_hi inside the phase
                if name in ('C', 'D'):
                    lo = speed_rng[0]
                    hi = speed_rng[0] + p_in * (speed_rng[1] - speed_rng[0])
                    return name, (lo, max(hi, lo + 1e-3)), jit_xy
                return name, speed_rng, jit_xy
        # Past end → final phase params
        last = self.PHASES[-1]
        return last[0], last[3], last[4]

    def _on_rollout_end(self):
        progress = min(self.num_timesteps / max(self.total_timesteps, 1), 1.0)
        prev_phase = self.current_phase

        name, speed_range, jit_xy = self._pick_phase_params(progress)
        self.current_phase = name
        self.raw_env.target_speed_range = speed_range
        self.raw_env.jitter_xy = jit_xy

        # α schedule
        a_p = min(self.num_timesteps / max(self.alpha_warmup_steps, 1), 1.0)
        self.raw_env.current_alpha = (
            self.alpha_min + a_p * (self.alpha_max - self.alpha_min))

        if name != prev_phase:
            print(f"\n  Phase {prev_phase} -> {name}  "
                  f"(speed={speed_range}  jit_xy={jit_xy:.2f}  "
                  f"α={self.raw_env.current_alpha:.2f})")

        if hasattr(self.model, 'ep_info_buffer'):
            for ep in self.model.ep_info_buffer:
                if 'r' in ep:
                    self.episode_rewards.append(ep['r'])

        now = time.time()
        if now - self.last_log_time > 5.0:
            self._log()
            self.last_log_time = now

    def _on_step(self):
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones = self.locals.get('dones', [False])
        actions = self.locals.get('actions', None)

        if not infos:
            return True
        info = infos[0]

        self._ep_reward += float(rewards[0])
        self._ep_steps += 1

        if info.get('target_visible', False):
            self._ep_visible += 1

        self._ep_alive += float(info.get('v7_alive', 0.0))
        self._ep_track += float(info.get('v7_track', 0.0))
        self._ep_invis += float(info.get('v7_invisible', 0.0))
        self._ep_jerk += float(info.get('v7_jerk', 0.0))
        if info.get('v7_term_reason'):
            self._ep_term_reason = info['v7_term_reason']

        if actions is not None:
            act = actions[0]
            self._ep_action_mags.append(float(np.mean(np.abs(act))))
            if self._prev_action is not None:
                self._ep_action_jerks.append(
                    float(np.mean(np.abs(act - self._prev_action))))
            self._prev_action = np.asarray(act).copy()

        if dones[0]:
            self._write_csv(info)
            self._reset_accum()

        return True

    def _write_csv(self, last_info):
        if not self._csv_writer:
            return
        self.episode_count += 1
        survived = int(self._ep_steps >= self.raw_env.base_env.n)
        _m = lambda lst: round(float(np.mean(lst)), 4) if lst else 0.0
        self._csv_writer.writerow([
            self.episode_count,
            self.num_timesteps,
            round(self._ep_reward, 2),
            self._ep_steps,
            round(100 * self._ep_visible / max(self._ep_steps, 1), 1),
            survived,
            _m(self._ep_action_mags),
            _m(self._ep_action_jerks),
            round(self._ep_alive, 3),
            round(self._ep_track, 3),
            round(self._ep_invis, 3),
            round(self._ep_jerk, 3),
            self.current_phase,
            round(self.raw_env.current_alpha, 3),
            round(self.raw_env.target_speed, 4),
            round(self.raw_env.jitter_xy, 3),
            self._ep_term_reason,
        ])
        self._csv_file.flush()

    def _log(self):
        elapsed = time.time() - self.start_time
        ts = self.num_timesteps
        pct = 100 * ts / max(self.total_timesteps, 1)
        fps = ts / max(elapsed, 1)
        mean_r = (float(np.mean(self.episode_rewards))
                  if self.episode_rewards else 0)
        print(f"  [{pct:5.1f}%] Step {ts:>7,}/{self.total_timesteps:,} | "
              f"Ep={self.episode_count} | R={mean_r:7.2f} | "
              f"Phase={self.current_phase} α={self.raw_env.current_alpha:.2f} "
              f"spd={self.raw_env.target_speed_range[1]:.2f} | "
              f"{fps:.0f} fps")

    def save_metrics(self, extras=None):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        summary = {
            'total_episodes': self.episode_count,
            'final_mean_reward': (float(np.mean(self.episode_rewards))
                                  if self.episode_rewards else 0),
            'final_phase': self.current_phase,
            'csv_path': str(self.csv_path),
        }
        if extras:
            summary.update(extras)
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
# Best-by-survival evaluator + early stop on regression
# ══════════════════════════════════════════════════════════════════════

class BestSurvivalEvalCallback(BaseCallback):
    """
    Evaluate every ``eval_freq`` steps over ``n_eval_episodes`` with FIXED
    seeds. Save best by lex order (survival, visibility, -jerk).

    Early-stop the training if survival drops more than
    ``regression_tolerance`` from the best for two consecutive evaluations.
    """

    def __init__(self, raw_env, model_ref_getter, output_dir,
                 training_vec_env, n_stack,
                 eval_freq=10_000, n_eval_episodes=5,
                 regression_tolerance=0.20, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.model_ref_getter = model_ref_getter
        self.output_dir = Path(output_dir)
        self.training_vec_env = training_vec_env
        self.n_stack = n_stack
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.regression_tolerance = regression_tolerance
        self.eval_seeds = list(range(1000, 1000 + n_eval_episodes))

        self.best_survival = -1.0
        self.best_visibility = -1.0
        self.best_jerk = float('inf')
        self.best_step = 0
        self.consecutive_regressions = 0
        self.last_eval_step = 0
        self.eval_log = []

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps
        self._evaluate()
        return True

    def _evaluate(self):
        model = self.model_ref_getter()
        survivals, visibilities, jerks = [], [], []
        ep_steps_list = []
        max_steps = self.raw_env.base_env.n
        obs_dim = int(self.raw_env.observation_space.shape[0])

        # Lock target speed at typical mid-range for evaluation
        prev_range = self.raw_env.target_speed_range
        self.raw_env.target_speed_range = (0.10, 0.10)

        for seed in self.eval_seeds:
            obs, _ = self.raw_env.reset(seed=seed)
            obs = np.asarray(obs, dtype=np.float32)
            # Build initial frame-stack by replicating the first obs.
            # VecFrameStack convention: oldest first, newest last.
            stack = np.tile(obs, self.n_stack).astype(np.float32)
            done = False
            step = 0
            visible = 0
            prev_act = None
            jerk_sum = 0.0
            jerk_n = 0
            while not done and step < max_steps:
                act, _ = model.predict(stack, deterministic=True)
                obs, _r, term, trunc, info = self.raw_env.step(act)
                obs = np.asarray(obs, dtype=np.float32)
                # Shift stack left by one frame and append new obs at the end
                stack = np.concatenate([stack[obs_dim:], obs])
                if info.get('target_visible', False):
                    visible += 1
                if prev_act is not None:
                    jerk_sum += float(np.mean(np.abs(act - prev_act)))
                    jerk_n += 1
                prev_act = act
                step += 1
                done = term or trunc
            survivals.append(int(step >= max_steps))
            visibilities.append(visible / max(step, 1))
            jerks.append(jerk_sum / max(jerk_n, 1))
            ep_steps_list.append(step)

        self.raw_env.target_speed_range = prev_range

        # Re-sync the training vec_env after the eval mutated the raw_env:
        # SAC caches `_last_obs`; after our eval that cache is stale.
        # Forcing a vec_env reset gives SAC a fresh, consistent observation.
        new_obs = self.training_vec_env.reset()
        model._last_obs = new_obs
        if hasattr(model, '_last_episode_starts'):
            model._last_episode_starts = np.ones(
                (self.training_vec_env.num_envs,), dtype=bool)

        survival_rate = float(np.mean(survivals))
        visibility = float(np.mean(visibilities))
        jerk = float(np.mean(jerks))
        mean_steps = float(np.mean(ep_steps_list))

        self.eval_log.append({
            'timestep': int(self.num_timesteps),
            'survival': survival_rate,
            'visibility': visibility,
            'jerk': jerk,
            'mean_steps': mean_steps,
        })
        with open(self.output_dir / 'eval_log.json', 'w') as f:
            json.dump(self.eval_log, f, indent=2)

        # Lex order: (survival, visibility, -jerk)
        better = (
            (survival_rate, visibility, -jerk) >
            (self.best_survival, self.best_visibility, -self.best_jerk)
        )
        msg = (f"  ▸ Eval @{self.num_timesteps:>7,} | "
               f"surv={survival_rate:.2f} vis={visibility:.2%} "
               f"jerk={jerk:.4f} steps={mean_steps:.0f}")

        if better:
            self.best_survival = survival_rate
            self.best_visibility = visibility
            self.best_jerk = jerk
            self.best_step = self.num_timesteps
            model.save(str(self.output_dir / 'best_model'))
            print(msg + "  ★ NEW BEST")
        else:
            print(msg)

        # Early stop on sustained regression
        if (self.best_survival > 0.0 and
                survival_rate < self.best_survival - self.regression_tolerance):
            self.consecutive_regressions += 1
            print(f"    ⚠ Regression {self.consecutive_regressions}/2 "
                  f"(best={self.best_survival:.2f})")
            if self.consecutive_regressions >= 2:
                print(f"\n  ⛔ Early stop — survival regressed > "
                      f"{self.regression_tolerance:.0%} from best for 2 evals.")
                return False  # stops training
        else:
            self.consecutive_regressions = 0
        return True


# ══════════════════════════════════════════════════════════════════════
# Bounded video recorder — at most ``max_videos`` recordings total
# ══════════════════════════════════════════════════════════════════════

class BoundedVideoCallback(BaseCallback):
    def __init__(self, raw_env, ext_camera, curriculum_cb,
                 output_dir, total_timesteps, max_videos=10, fps=10, verbose=0):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.ext_camera = ext_camera
        self.curriculum_cb = curriculum_cb
        self.fps = fps
        self.frame_step = max(1, 100 // fps)

        self.recorder = EpisodeRecorder(
            output_dir=str(Path(output_dir) / 'recordings'),
            fps=fps,
            resolution=(480, 360),
        )

        # Schedule: max_videos triggers spaced linearly across training.
        # First trigger at ~total/max_videos, last shortly before the end.
        self.triggers = [int((i + 1) * total_timesteps / (max_videos + 1))
                         for i in range(max_videos)]
        self.next_trigger_idx = 0

        self._is_recording = False
        self._step_in_ep = 0
        self._episode_count = 0
        self._armed = False

    def _on_step(self):
        # Arm when we cross a trigger threshold
        if (self.next_trigger_idx < len(self.triggers) and
                self.num_timesteps >= self.triggers[self.next_trigger_idx]):
            self._armed = True
            self.next_trigger_idx += 1

        infos = self.locals.get('infos', [{}])
        dones = self.locals.get('dones', [False])
        rewards = self.locals.get('rewards', [0.0])
        info = infos[0] if infos else {}

        if self._is_recording:
            self._step_in_ep += 1
            if self._step_in_ep % self.frame_step == 0:
                self._capture(info, rewards[0])

        if dones[0]:
            if self._is_recording:
                self.recorder.end_episode()
                self._is_recording = False
            self._episode_count += 1
            if self._armed:
                self._armed = False
                self._is_recording = True
                self._step_in_ep = 0
                self.recorder.start_episode(self._episode_count)
                self._position_ext_camera()
        return True

    def _capture(self, info, reward):
        fpv_img = self.raw_env._last_high_freq_image
        bird_img = None
        if self.ext_camera is not None:
            ok, rgba = self.ext_camera.get_image()
            if ok:
                bird_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        overlay = {
            'visual_tracking': info.get('visual_tracking', {}),
            'target': info.get('target', {}),
            'Step': self._step_in_ep,
            'Timestep': self.num_timesteps,
            'Reward': round(float(reward), 2),
            'Phase': self.curriculum_cb.current_phase,
            'Alpha': round(self.raw_env.current_alpha, 2),
            'Speed': round(self.raw_env.target_speed, 3),
        }
        self.recorder.capture_frame(fpv_img, bird_img, overlay)

    def _position_ext_camera(self):
        if self.ext_camera is None:
            return
        drone_pos = self.raw_env.base_env.state[0:5:2]
        target_pos = self.raw_env.target_pos
        mx = (drone_pos[0] + target_pos[0]) / 2
        my = (drone_pos[1] + target_pos[1]) / 2
        mz = drone_pos[2] + 5
        cam_dist = 6.0
        self.ext_camera.cam.setPos(
            mx - cam_dist * 0.6,
            my - cam_dist * 0.6,
            mz + cam_dist * 0.4,
        )
        self.ext_camera.cam.lookAt(float(mx), float(my), float(mz - 0.5))

    def compile_timelapse(self):
        if self.recorder.episode_files:
            return self.recorder.compile_timelapse(
                "v7_training_timelapse.mp4", max_frames_per_ep=150)
        return None


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Hover-track v7 — train from scratch (survival-first)")
    p.add_argument('--timesteps', type=int, default=200_000)
    p.add_argument('--hover-height', type=float, default=1.394)
    p.add_argument('--max-ep-steps', type=int, default=3000,
                   help="3000 steps = 30 s episode")
    p.add_argument('--lemniscate-scale', type=float, default=2.0)
    p.add_argument('--output-dir', type=str, default='./models/hover_track_v7_1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')

    # SAC hyperparameters
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=500_000)
    p.add_argument('--target-entropy', type=float, default=-2.0)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--gamma', type=float, default=0.99)

    # v7 reward weights
    p.add_argument('--w-alive', type=float, default=0.50,
                   help="v7.1: bumped 0.10 → 0.50 to kill 'die early' "
                        "reward-hacking observed in v7.")
    p.add_argument('--w-jerk', type=float, default=0.40)
    p.add_argument('--w-invisible', type=float, default=0.05)
    p.add_argument('--stability-gate', type=float, default=0.30)

    # α schedule
    p.add_argument('--alpha-min', type=float, default=0.20)
    p.add_argument('--alpha-max', type=float, default=1.00)
    p.add_argument('--alpha-warmup-steps', type=int, default=80_000)

    # Spawn jitter
    p.add_argument('--jitter-xy', type=float, default=0.30)
    p.add_argument('--jitter-z', type=float, default=0.10)
    p.add_argument('--jitter-yaw-deg', type=float, default=15.0)

    # Soft termination
    p.add_argument('--alt-soft-limit', type=float, default=1.5)
    p.add_argument('--alt-soft-steps', type=int, default=100)
    p.add_argument('--invisible-term-steps', type=int, default=60,
                   help="Truncate episode after N consecutive steps without "
                        "the sphere visible (v7.1: 60 = 0.6 s, was 200/2.0 s "
                        "in v7 — long invisibility windows accumulated too "
                        "much penalty and incentivised early termination).")

    # Eval & checkpoints
    p.add_argument('--eval-freq', type=int, default=10_000)
    p.add_argument('--n-eval-episodes', type=int, default=5)
    p.add_argument('--regression-tolerance', type=float, default=0.20)
    p.add_argument('--checkpoint-freq', type=int, default=25_000)

    # Frame stacking
    p.add_argument('--n-stack', type=int, default=3)

    # Videos (max 10 over training)
    p.add_argument('--max-videos', type=int, default=10)
    p.add_argument('--record-fps', type=int, default=10)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Application
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV7App(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D scene...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -8, 14)
        self.cam.lookAt(0, 0, 5)

        # FPV camera (down-facing) — used by the centroid pipeline
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.1)
        self.fpv_camera.cam.setHpr(0, -90, 0)
        self.fpv_camera.buffer.setActive(1)

        # External camera for video
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.cam.setPos(0, -6, 12)
        self.ext_camera.cam.lookAt(0, 0, 5)
        self.ext_camera.buffer.setActive(1)

        print("Creating environment (hover-track v7, from scratch)...")
        self.raw_env = HoverTrackV7Wrapper(
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
            target_mode='moving',
            target_speed=0.0,
            target_radius=0.25,
            lemniscate_scale=args.lemniscate_scale,
            filming_mode=True,
            enable_collisions=False,
            n=args.max_ep_steps,
            t_step=0.01,
            direct_control=1,
            centroid_obs=True,
            camera_down=True,
            hover_height=args.hover_height,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.2,
            init_vel_range=0.10,
            init_ang_range=0.03,
            reward_version='v3.1',  # base reward, OVERRIDDEN in step()
            # v7 wrapper kwargs
            jitter_xy=args.jitter_xy,
            jitter_z=args.jitter_z,
            jitter_yaw_deg=args.jitter_yaw_deg,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_warmup_steps=args.alpha_warmup_steps,
            w_alive=args.w_alive,
            w_jerk=args.w_jerk,
            w_invisible=args.w_invisible,
            stability_gate=args.stability_gate,
            alt_soft_limit=args.alt_soft_limit,
            alt_soft_steps=args.alt_soft_steps,
            invisible_term_steps=args.invisible_term_steps,
        )

        self.env = Monitor(self.raw_env)
        base_vec = DummyVecEnv([lambda: self.env])
        self.vec_env = VecFrameStack(base_vec, n_stack=args.n_stack)

        # ── Build a fresh SAC model (no fine-tune) ──
        self.model = SAC(
            policy='MlpPolicy',
            env=self.vec_env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            gamma=args.gamma,
            ent_coef='auto',
            target_entropy=args.target_entropy,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 128]),
            seed=args.seed,
            verbose=0,
        )

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        total_params = sum(p.numel() for p in self.model.policy.parameters())
        ep_duration = args.max_ep_steps * 0.01

        print("\n" + "=" * 70)
        print("  HOVER-TRACK v7.1 — SAC FROM SCRATCH (v7 fixes applied)")
        print("=" * 70)
        print(f"  Hover height:      {args.hover_height} m")
        print(f"  Lemniscate scale:  {args.lemniscate_scale} m")
        print(f"  Observation:       19-D × {args.n_stack} stack "
              f"= {19*args.n_stack}-D")
        print(f"  Policy:            MlpPolicy [256, 128]")
        print(f"  Algorithm:         SAC (target_entropy={args.target_entropy})")
        print(f"  Learning rate:     {args.learning_rate}")
        print(f"  Buffer size:       {args.buffer_size:,}")
        print(f"  Timesteps:         {args.timesteps:,}")
        print(f"  Episode steps:     {args.max_ep_steps} ({ep_duration:.0f}s)")
        print(f"  Policy params:     {total_params:,}")
        print(f"  Reward weights:    alive={args.w_alive} "
              f"jerk={args.w_jerk} invisible={args.w_invisible} "
              f"stab_gate={args.stability_gate}")
        print(f"  α schedule:        {args.alpha_min} → {args.alpha_max} "
              f"over {args.alpha_warmup_steps:,} steps")
        print(f"  Spawn jitter:      XY±{args.jitter_xy} m  "
              f"Z±{args.jitter_z} m  yaw±{args.jitter_yaw_deg}°")
        print(f"  Soft alt term:     |Δz|>{args.alt_soft_limit} m for "
              f"{args.alt_soft_steps} steps")
        print(f"  Lost-target trunc: {args.invisible_term_steps} steps "
              f"({args.invisible_term_steps/100:.1f} s)")
        print(f"  Curriculum:        A[0-35%] B[35-50%] C[50-75%] D[75-100%]")
        print(f"  Phase A jit_xy:    {0.05:.2f} → {0.30:.2f} (ramp inside A)")
        print(f"  Eval:              every {args.eval_freq:,} steps × "
              f"{args.n_eval_episodes} eps (fixed seeds)")
        print(f"  Best metric:       (survival, visibility, -jerk) lex")
        print(f"  Early stop:        survival -{args.regression_tolerance:.0%} "
              f"× 2 evals")
        print(f"  Videos:            max {args.max_videos} across training")
        print(f"  Output:            {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        args = self.args
        render_cb = Panda3DRenderCallback(self)

        self.curriculum_cb = CurriculumV7Callback(
            raw_env=self.raw_env,
            output_dir=str(self.output_dir),
            total_timesteps=args.timesteps,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_warmup_steps=args.alpha_warmup_steps,
        )

        self.eval_cb = BestSurvivalEvalCallback(
            raw_env=self.raw_env,
            model_ref_getter=lambda: self.model,
            output_dir=str(self.output_dir),
            training_vec_env=self.vec_env,
            n_stack=args.n_stack,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            regression_tolerance=args.regression_tolerance,
        )

        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cb = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(ckpt_dir),
            name_prefix='model',
        )

        self.video_cb = None
        callbacks = [render_cb, self.curriculum_cb, ckpt_cb, self.eval_cb]
        if args.max_videos > 0:
            self.video_cb = BoundedVideoCallback(
                raw_env=self.raw_env,
                ext_camera=self.ext_camera,
                curriculum_cb=self.curriculum_cb,
                output_dir=str(self.output_dir),
                total_timesteps=args.timesteps,
                max_videos=args.max_videos,
                fps=args.record_fps,
            )
            callbacks.append(self.video_cb)

        print("\nStarting training...\n")
        start = time.time()
        self.model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True,
        )
        elapsed = time.time() - start

        # Save final model (separate from best)
        final_path = self.output_dir / 'final_model'
        self.model.save(str(final_path))

        self.curriculum_cb.save_metrics(extras={
            'total_timesteps': args.timesteps,
            'training_time_seconds': elapsed,
            'algorithm': 'SAC',
            'version': 'v7.1',
            'fine_tune': False,
            'observation_dim': 19,
            'n_frame_stack': args.n_stack,
            'net_arch': [256, 128],
            'learning_rate': args.learning_rate,
            'target_entropy': args.target_entropy,
            'max_ep_steps': args.max_ep_steps,
            'episode_duration_s': args.max_ep_steps * 0.01,
            'reward_version': 'v7 (custom: alive + α·track + invisible + jerk)',
            'reward_weights': {
                'alive': args.w_alive,
                'jerk': args.w_jerk,
                'invisible': args.w_invisible,
                'stability_gate': args.stability_gate,
            },
            'alpha_schedule': {
                'min': args.alpha_min,
                'max': args.alpha_max,
                'warmup_steps': args.alpha_warmup_steps,
            },
            'spawn': {
                'jitter_xy': args.jitter_xy,
                'jitter_z': args.jitter_z,
                'jitter_yaw_deg': args.jitter_yaw_deg,
                'mode': 'deterministic_above_target',
            },
            'soft_termination': {
                'altitude_limit_m': args.alt_soft_limit,
                'altitude_streak_steps': args.alt_soft_steps,
                'invisible_streak_steps': args.invisible_term_steps,
            },
            'curriculum_phases': {
                'A': '0-35% (hover, target stationary, jit_xy ramps 0.05→0.30)',
                'B': '35-50% (static FOV, target stationary, jit_xy=0.30)',
                'C': '50-75% (lemniscate slow 0.05-0.10 m/s)',
                'D': '75-100% (lemniscate medium 0.10-0.20 m/s)',
            },
            'eval': {
                'best_survival': self.eval_cb.best_survival,
                'best_visibility': self.eval_cb.best_visibility,
                'best_jerk': self.eval_cb.best_jerk,
                'best_step': self.eval_cb.best_step,
            },
            'policy_params': sum(
                p.numel() for p in self.model.policy.parameters()),
        })

        timelapse_path = None
        if self.video_cb is not None:
            print("\nCompiling training timelapse...")
            timelapse_path = self.video_cb.compile_timelapse()

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Time:         {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Episodes:     {self.curriculum_cb.episode_count}")
        print(f"  Best @ step:  {self.eval_cb.best_step:,}  "
              f"(survival={self.eval_cb.best_survival:.2f}  "
              f"visibility={self.eval_cb.best_visibility:.2%}  "
              f"jerk={self.eval_cb.best_jerk:.4f})")
        print(f"  Best model:   {self.output_dir / 'best_model'}.zip")
        print(f"  Final model:  {final_path}.zip")
        print(f"  Log:          {self.curriculum_cb.csv_path}")
        if timelapse_path:
            print(f"  Timelapse:    {timelapse_path}")
        n_recordings = (len(self.video_cb.recorder.episode_files)
                        if self.video_cb else 0)
        print(f"  Recordings:   {n_recordings}")
        print("=" * 70)

        self.vec_env.close()


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = HoverTrackV7App(args)
    try:
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            p = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(p))
            print(f"Model saved to {p}.zip")
        if hasattr(app, 'curriculum_cb'):
            app.curriculum_cb.save_metrics()
        if hasattr(app, 'video_cb') and app.video_cb is not None:
            print("Compiling timelapse from recorded episodes...")
            app.video_cb.compile_timelapse()


if __name__ == "__main__":
    main()
