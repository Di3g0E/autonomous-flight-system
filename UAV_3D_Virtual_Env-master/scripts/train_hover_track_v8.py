#!/usr/bin/env python
"""
Hover-track v8 — minimal-reward redesign.

After v7.0–v7.5 stacked many ad-hoc fixes (α schedule, 4-phase curriculum,
growing alive, asymmetric soft-term, action-mag cap, stability gate, ...)
without achieving survival > 0, this version starts over with a deliberately
small set of ingredients:

  • SPAWN: drone at 1.5 m above the (static) sphere with ±0.2 m XY jitter.
  • REWARD (per step):
        r =   r_track + r_alive + r_jerk
      where:
        r_track = exp(-3 * dist_to_center²)  if visible, else -1.0
        r_alive = +0.1                       (small constant)
        r_jerk  = -0.2 * ||Δaction||²        (mild smoothness)
    All terms are O(1). No α schedule, no growing terms, no stability gate.
  • TERMINATION:
        z < 0.5 or z > 3.0  (hard physical bounds, drone leaves working zone)
        invisible 100 steps consecutively (1 s of lost lock)
        crash penalty: -10 on any termination
  • CURRICULUM: NONE. Target is static throughout the run.
  • OBSERVATION: 19-D centroid_obs (state + cx,cy,frac,vis,dcx,dcy).
    NO frame stacking — dcx/dcy already encodes centroid velocity for a
    static target. Frame stacking will be added in v9 when we re-introduce
    a moving target (lemniscate).
  • SAC DEFAULTS: lr=3e-4, target_entropy=auto, batch=256, buffer=200k.

Goal: establish a clean baseline that either succeeds with minimal shaping
(in which case complexity was the problem in v7) or fails (telling us the
core problem is environmental/geometric and reward shaping won't fix it).

Usage:
    python scripts/train_hover_track_v8.py --no-display
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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.utils.episode_recorder import EpisodeRecorder


# ══════════════════════════════════════════════════════════════════════
# Wrapper: minimal reward + minimal termination
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV8Wrapper(Panda3DQuadrotorEnv):
    """Minimal hover-track environment.

    Spawn: drone placed exactly above the static target sphere with bounded
    XY jitter. Reward overrides the env base completely.
    """

    def __init__(self, *args,
                 spawn_height=1.5, jitter_xy=0.20,
                 w_alive=0.10, w_jerk=0.20, w_invisible=1.0,
                 z_min=0.5, z_max=3.0,
                 invisible_term_steps=100, crash_penalty=10.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.spawn_height = spawn_height
        self.jitter_xy = jitter_xy
        self.w_alive = w_alive
        self.w_jerk = w_jerk
        self.w_invisible = w_invisible
        self.z_min = z_min
        self.z_max = z_max
        self.invisible_term_steps = invisible_term_steps
        self.crash_penalty = crash_penalty
        self._invisible_streak = 0
        self._prev_action = None

    def reset(self, seed=None, options=None):
        # Force the lemniscate target_speed to 0 so the sphere is static
        # regardless of whatever the env defaults to.
        self.target_speed = 0.0

        obs, info = super().reset(seed=seed, options=options)

        state = self.base_env.state.copy()
        dx = float(np.random.uniform(-self.jitter_xy, self.jitter_xy))
        dy = float(np.random.uniform(-self.jitter_xy, self.jitter_xy))
        state[0] = self.target_pos[0] + dx           # x
        state[2] = self.target_pos[1] + dy           # y
        state[4] = self.target_pos[2] + self.spawn_height  # z
        state[1] = 0.0                                # vx
        state[3] = 0.0                                # vy
        state[5] = 0.0                                # vz
        state[6] = 1.0                                # quaternion identity
        state[7:10] = 0.0
        state[10:13] = 0.0                            # angular velocities

        self.base_env.state = state.copy()
        self.base_env.previous_state = state.copy()

        self._update_visualization()
        if self.panda3d_app:
            self.panda3d_app.graphicsEngine.renderFrame()
        if self.use_camera:
            self._capture_camera_images(force_capture=True)
        obs = self._build_observation(state.astype(np.float32))

        self._invisible_streak = 0
        self._prev_action = None
        return obs, info

    def step(self, action):
        try:
            obs, _, terminated, truncated, info = super().step(action)
        except RuntimeError as e:
            obs = self._build_observation(self.base_env.state.astype(np.float32))
            info = {'physics_diverged': str(e),
                    'visual_tracking': {'target_visible': False,
                                        'cx': 0.0, 'cy': 0.0}}
            return obs, -self.crash_penalty, True, False, info

        vt = info.get('visual_tracking', {}) or {}
        target_visible = bool(vt.get('target_visible', False))

        # ── Reward (3 terms, all O(1)) ──
        if target_visible:
            cx = float(vt.get('target_center', (0.0, 0.0))[0])
            cy = float(vt.get('target_center', (0.0, 0.0))[1])
            dist_sq = cx * cx + cy * cy
            r_track = float(np.exp(-3.0 * dist_sq))
        else:
            r_track = -self.w_invisible

        r_alive = self.w_alive

        if self._prev_action is not None:
            delta = float(np.linalg.norm(
                np.asarray(action) - self._prev_action))
            r_jerk = -self.w_jerk * delta * delta
        else:
            r_jerk = 0.0
        self._prev_action = np.asarray(action, dtype=np.float32).copy()

        reward = r_track + r_alive + r_jerk

        # ── Termination: hard altitude bounds + lost-lock truncation ──
        drone_z = float(self.base_env.state[4])
        target_z = float(self.target_pos[2])
        z_rel = drone_z - target_z
        term_reason = ''
        if z_rel < self.z_min or z_rel > self.z_max:
            terminated = True
            reward -= self.crash_penalty
            term_reason = 'altitude_low' if z_rel < self.z_min else 'altitude_high'

        if not target_visible:
            self._invisible_streak += 1
        else:
            self._invisible_streak = 0
        if self._invisible_streak >= self.invisible_term_steps:
            truncated = True
            term_reason = term_reason or 'lost_target'

        info['v8_track'] = r_track
        info['v8_alive'] = r_alive
        info['v8_jerk'] = r_jerk
        info['v8_term_reason'] = term_reason
        return obs, reward, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════
# Panda3D render driver (same pattern as v7)
# ══════════════════════════════════════════════════════════════════════

class Panda3DRenderCallback(BaseCallback):
    def __init__(self, app):
        super().__init__()
        self.app = app

    def _on_step(self):
        self.app.taskMgr.step()
        return True


# ══════════════════════════════════════════════════════════════════════
# Episode metrics CSV
# ══════════════════════════════════════════════════════════════════════

class MetricsCallback(BaseCallback):
    HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'visibility_pct', 'survived',
        'mean_action_mag', 'mean_action_jerk',
        'r_track_sum', 'r_alive_sum', 'r_jerk_sum',
        'term_reason',
    ]

    def __init__(self, raw_env, output_dir, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_timesteps = total_timesteps
        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=50)
        self._reset_acc()
        self.start_time = time.time()
        self.last_log_time = time.time()

    def _reset_acc(self):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_visible = 0
        self._ep_action_mags = []
        self._ep_action_jerks = []
        self._ep_track = 0.0
        self._ep_alive = 0.0
        self._ep_jerk = 0.0
        self._ep_term_reason = ''
        self._prev_action = None

    def _on_training_start(self):
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.HEADERS)

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

        vt = info.get('visual_tracking', {}) or {}
        if vt.get('target_visible', False):
            self._ep_visible += 1
        self._ep_track += float(info.get('v8_track', 0.0))
        self._ep_alive += float(info.get('v8_alive', 0.0))
        self._ep_jerk += float(info.get('v8_jerk', 0.0))
        if info.get('v8_term_reason'):
            self._ep_term_reason = info['v8_term_reason']

        if actions is not None:
            act = actions[0]
            self._ep_action_mags.append(float(np.mean(np.abs(act))))
            if self._prev_action is not None:
                self._ep_action_jerks.append(
                    float(np.mean(np.abs(act - self._prev_action))))
            self._prev_action = np.asarray(act).copy()

        if dones[0]:
            self._write_row()
            self._reset_acc()

        # log every ~5 s
        now = time.time()
        if now - self.last_log_time > 5.0:
            self._log_progress()
            self.last_log_time = now
        return True

    def _write_row(self):
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
            round(self._ep_track, 3),
            round(self._ep_alive, 3),
            round(self._ep_jerk, 3),
            self._ep_term_reason,
        ])
        self._csv_file.flush()
        self.episode_rewards.append(self._ep_reward)

    def _log_progress(self):
        elapsed = time.time() - self.start_time
        ts = self.num_timesteps
        pct = 100 * ts / max(self.total_timesteps, 1)
        fps = ts / max(elapsed, 1)
        mean_r = (float(np.mean(self.episode_rewards))
                  if self.episode_rewards else 0)
        print(f"  [{pct:5.1f}%] Step {ts:>7,}/{self.total_timesteps:,} | "
              f"Ep={self.episode_count} | R={mean_r:7.2f} | {fps:.0f} fps")

    def save_summary(self, extras=None):
        if self._csv_file:
            self._csv_file.close()
        summary = {
            'total_episodes': self.episode_count,
            'final_mean_reward': (float(np.mean(self.episode_rewards))
                                  if self.episode_rewards else 0),
            'csv_path': str(self.csv_path),
        }
        if extras:
            summary.update(extras)
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
# Eval callback (best by survival, vis, -jerk)
# ══════════════════════════════════════════════════════════════════════

class BestSurvivalEval(BaseCallback):
    def __init__(self, raw_env, training_vec_env, model_getter, output_dir,
                 eval_freq=10_000, n_episodes=5, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.training_vec_env = training_vec_env
        self.model_getter = model_getter
        self.output_dir = Path(output_dir)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.eval_seeds = list(range(1000, 1000 + n_episodes))
        self.best_survival = -1.0
        self.best_visibility = -1.0
        self.best_jerk = float('inf')
        self.best_step = 0
        self.last_eval_step = 0
        self.eval_log = []

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps
        self._evaluate()
        return True

    def _evaluate(self):
        model = self.model_getter()
        max_steps = self.raw_env.base_env.n
        panda_app = getattr(self.raw_env, 'panda3d_app', None)
        survivals, visibilities, jerks, ep_steps_list = [], [], [], []

        for seed in self.eval_seeds:
            obs, _ = self.raw_env.reset(seed=seed)
            if panda_app is not None:
                panda_app.taskMgr.step()
            done = False
            step = 0
            visible = 0
            prev_act = None
            jerk_sum = 0.0
            jerk_n = 0
            while not done and step < max_steps:
                act, _ = model.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = self.raw_env.step(act)
                if panda_app is not None:
                    panda_app.taskMgr.step()
                vt = info.get('visual_tracking', {}) or {}
                if vt.get('target_visible', False):
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

        # re-sync the training vec_env after we mutated raw_env
        new_obs = self.training_vec_env.reset()
        model._last_obs = new_obs
        if hasattr(model, '_last_episode_starts'):
            model._last_episode_starts = np.ones(
                (self.training_vec_env.num_envs,), dtype=bool)

        surv = float(np.mean(survivals))
        vis = float(np.mean(visibilities))
        jerk = float(np.mean(jerks))
        steps = float(np.mean(ep_steps_list))
        self.eval_log.append({
            'timestep': int(self.num_timesteps),
            'survival': surv, 'visibility': vis,
            'jerk': jerk, 'mean_steps': steps,
        })
        with open(self.output_dir / 'eval_log.json', 'w') as f:
            json.dump(self.eval_log, f, indent=2)

        better = ((surv, vis, -jerk) >
                  (self.best_survival, self.best_visibility, -self.best_jerk))
        msg = (f"  -> Eval @{self.num_timesteps:>7,} | "
               f"surv={surv:.2f} vis={vis:.2%} "
               f"jerk={jerk:.4f} steps={steps:.0f}")
        if better:
            self.best_survival = surv
            self.best_visibility = vis
            self.best_jerk = jerk
            self.best_step = self.num_timesteps
            model.save(str(self.output_dir / 'best_model'))
            print(msg + "  * NEW BEST")
        else:
            print(msg)
        return True


# ══════════════════════════════════════════════════════════════════════
# Bounded video recorder (max N videos across whole run)
# ══════════════════════════════════════════════════════════════════════

class BoundedVideoCallback(BaseCallback):
    def __init__(self, raw_env, ext_camera, output_dir, total_timesteps,
                 max_videos=8, fps=10):
        super().__init__()
        self.raw_env = raw_env
        self.ext_camera = ext_camera
        self.fps = fps
        self.frame_step = max(1, 100 // fps)
        self.recorder = EpisodeRecorder(
            output_dir=str(Path(output_dir) / 'recordings'),
            fps=fps, resolution=(480, 360),
        )
        self.triggers = [int((i + 1) * total_timesteps / (max_videos + 1))
                         for i in range(max_videos)]
        self.next_trigger_idx = 0
        self._is_recording = False
        self._step_in_ep = 0
        self._episode_count = 0
        self._armed = False

    def _on_step(self):
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
        }
        self.recorder.capture_frame(fpv_img, bird_img, overlay)

    def compile_timelapse(self):
        if self.recorder.episode_files:
            return self.recorder.compile_timelapse(
                "v8_training_timelapse.mp4", max_frames_per_ep=150)
        return None


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Hover-track v8 — minimal-reward redesign")
    p.add_argument('--timesteps', type=int, default=200_000)
    p.add_argument('--hover-height', type=float, default=1.394,
                   help="Used by env's r_scale/r_centering; v8 wrapper "
                        "ignores this for spawn (see --spawn-height).")
    p.add_argument('--spawn-height', type=float, default=1.5)
    p.add_argument('--max-ep-steps', type=int, default=3000)
    p.add_argument('--lemniscate-scale', type=float, default=2.0)
    p.add_argument('--output-dir', type=str, default='./models/hover_track_v8')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')

    # SAC defaults (kept default lr, default target_entropy)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=200_000)

    # Reward weights
    p.add_argument('--w-alive', type=float, default=0.10)
    p.add_argument('--w-jerk', type=float, default=0.20)
    p.add_argument('--w-invisible', type=float, default=1.0)

    # Spawn jitter
    p.add_argument('--jitter-xy', type=float, default=0.20)

    # Termination bounds (relative to target z)
    p.add_argument('--z-min', type=float, default=0.5)
    p.add_argument('--z-max', type=float, default=3.0)
    p.add_argument('--invisible-term-steps', type=int, default=100)
    p.add_argument('--crash-penalty', type=float, default=10.0)

    # Eval / videos
    p.add_argument('--eval-freq', type=int, default=10_000)
    p.add_argument('--n-eval-episodes', type=int, default=5)
    p.add_argument('--checkpoint-freq', type=int, default=25_000)
    p.add_argument('--max-videos', type=int, default=8)
    p.add_argument('--record-fps', type=int, default=10)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Application
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV8App(ShowBase):
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

        print("Creating env (hover-track v8 minimal)...")
        self.raw_env = HoverTrackV8Wrapper(
            panda3d_app=self, quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True, use_depth=False,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            use_target=True, target_mode='moving',
            target_speed=0.0, target_radius=0.25,
            lemniscate_scale=args.lemniscate_scale,
            filming_mode=True, enable_collisions=False,
            n=args.max_ep_steps, t_step=0.01, direct_control=1,
            centroid_obs=True, camera_down=True,
            hover_height=args.hover_height,
            use_new_reward=True, constrained_init=True,
            init_pos_range=0.2, init_vel_range=0.10, init_ang_range=0.03,
            reward_version='v3.1',  # base reward overridden by wrapper
            # v8 wrapper kwargs
            spawn_height=args.spawn_height,
            jitter_xy=args.jitter_xy,
            w_alive=args.w_alive, w_jerk=args.w_jerk,
            w_invisible=args.w_invisible,
            z_min=args.z_min, z_max=args.z_max,
            invisible_term_steps=args.invisible_term_steps,
            crash_penalty=args.crash_penalty,
        )

        self.env = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # Plain SAC, default hyperparameters as much as possible
        self.model = SAC(
            policy='MlpPolicy', env=self.vec_env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            ent_coef='auto',
            policy_kwargs=dict(net_arch=[256, 128]),
            seed=args.seed, verbose=0,
        )

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        total_params = sum(p.numel() for p in self.model.policy.parameters())

        print("\n" + "=" * 70)
        print("  HOVER-TRACK v8 — MINIMAL-REWARD BASELINE")
        print("=" * 70)
        print(f"  Reward:           r_track + r_alive + r_jerk")
        print(f"                    r_track = exp(-3*dist²) (or -{args.w_invisible} if invis)")
        print(f"                    r_alive = +{args.w_alive}/step")
        print(f"                    r_jerk  = -{args.w_jerk} * ||Δa||²")
        print(f"  Spawn:            z={args.spawn_height} m  XY±{args.jitter_xy} m")
        print(f"  Termination:      z<{args.z_min} or z>{args.z_max} (rel target)")
        print(f"                    invisible {args.invisible_term_steps} steps -> truncate")
        print(f"                    crash_penalty = -{args.crash_penalty}")
        print(f"  Curriculum:       NONE (target static)")
        print(f"  Frame stack:      NO (centroid obs already has dcx/dcy)")
        print(f"  Algorithm:        SAC defaults (lr={args.learning_rate}, "
              f"target_entropy=auto)")
        print(f"  Policy:           MlpPolicy [256, 128] ({total_params:,} params)")
        print(f"  Timesteps:        {args.timesteps:,}")
        print(f"  Episode steps:    {args.max_ep_steps} ({args.max_ep_steps/100:.0f} s)")
        print(f"  Eval:             every {args.eval_freq:,} steps × "
              f"{args.n_eval_episodes} eps")
        print(f"  Output:           {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        args = self.args
        render_cb = Panda3DRenderCallback(self)

        self.metrics_cb = MetricsCallback(
            raw_env=self.raw_env,
            output_dir=str(self.output_dir),
            total_timesteps=args.timesteps)

        self.eval_cb = BestSurvivalEval(
            raw_env=self.raw_env,
            training_vec_env=self.vec_env,
            model_getter=lambda: self.model,
            output_dir=str(self.output_dir),
            eval_freq=args.eval_freq,
            n_episodes=args.n_eval_episodes)

        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cb = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(ckpt_dir),
            name_prefix='model')

        self.video_cb = None
        callbacks = [render_cb, self.metrics_cb, ckpt_cb, self.eval_cb]
        if args.max_videos > 0:
            self.video_cb = BoundedVideoCallback(
                raw_env=self.raw_env, ext_camera=self.ext_camera,
                output_dir=str(self.output_dir),
                total_timesteps=args.timesteps,
                max_videos=args.max_videos, fps=args.record_fps)
            callbacks.append(self.video_cb)

        print("\nStarting training...\n")
        start = time.time()
        self.model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True)
        elapsed = time.time() - start

        final_path = self.output_dir / 'final_model'
        self.model.save(str(final_path))

        self.metrics_cb.save_summary(extras={
            'total_timesteps': args.timesteps,
            'training_time_seconds': elapsed,
            'algorithm': 'SAC',
            'version': 'v8',
            'reward_version': 'v8 (minimal: track + alive + jerk)',
            'reward_weights': {
                'alive': args.w_alive,
                'jerk': args.w_jerk,
                'invisible': args.w_invisible,
            },
            'spawn': {'height_m': args.spawn_height,
                      'jitter_xy': args.jitter_xy},
            'termination': {'z_min': args.z_min, 'z_max': args.z_max,
                            'invisible_steps': args.invisible_term_steps,
                            'crash_penalty': args.crash_penalty},
            'eval': {
                'best_survival': self.eval_cb.best_survival,
                'best_visibility': self.eval_cb.best_visibility,
                'best_jerk': self.eval_cb.best_jerk,
                'best_step': self.eval_cb.best_step,
            },
            'policy_params': sum(p.numel()
                                  for p in self.model.policy.parameters()),
        })

        timelapse_path = None
        if self.video_cb is not None:
            print("\nCompiling training timelapse...")
            timelapse_path = self.video_cb.compile_timelapse()

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Time:        {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Episodes:    {self.metrics_cb.episode_count}")
        print(f"  Best @ step: {self.eval_cb.best_step:,}  "
              f"(surv={self.eval_cb.best_survival:.2f}  "
              f"vis={self.eval_cb.best_visibility:.2%}  "
              f"jerk={self.eval_cb.best_jerk:.4f})")
        print(f"  Best model:  {self.output_dir / 'best_model'}.zip")
        print(f"  Final model: {final_path}.zip")
        if timelapse_path:
            print(f"  Timelapse:   {timelapse_path}")
        print("=" * 70)
        self.vec_env.close()


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = HoverTrackV8App(args)
    try:
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            p = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(p))
            print(f"Model saved to {p}.zip")
        if hasattr(app, 'metrics_cb'):
            app.metrics_cb.save_summary()
        if hasattr(app, 'video_cb') and app.video_cb is not None:
            app.video_cb.compile_timelapse()


if __name__ == "__main__":
    main()
