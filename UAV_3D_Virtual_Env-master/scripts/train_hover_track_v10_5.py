#!/usr/bin/env python
"""
Hover-track v10.5 — robustness fine-tuning over v10.4.

Goal: close the OOD gap that breaks the SEARCH→TRACK transition in the
full-flight demo. v10.4 was trained with very tight near-hover init
(vel ≤ 0.10 m/s, tilt ≤ 0.03 rad, ang vel ≤ 0.03 rad/s) so when the
spiral hands it control with vel ~0.5-1.5 m/s, tilt ~0.1-0.25 rad and
yaw rate ~1.0-1.8 rad/s, the policy operates entirely out of
distribution and emits erratic motor commands.

Fix: fine-tune v10.4 with a MIXED RESET. Each episode samples one of:
  * near-hover regime  (50 %): identical to v10.4 — preserves the
    tracking quality the model already learned.
  * perturbed regime   (50 %): broader init that covers the typical
    state right after the spiral hands control over — teaches recovery
    without forgetting the original task.

Transfer learning from v10.4's `best_model_TFG.zip` + `best_vec_normalize_TFG.pkl`
(use --init-from / --init-vec-norm-from to point elsewhere).

Lower learning rate (1e-4) to avoid devaluing v10.4 — see the memory
note about TL+SAC quality preservation.

Usage:
    python scripts/train_hover_track_v10_5.py --no-display --timesteps 150000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera

from scripts.train_hover_track_v9 import Panda3DRenderCallback
from scripts.train_hover_track_v10 import HoverTrackV10Wrapper
from src.simulation.quaternion_euler_utility import euler_quat


# ══════════════════════════════════════════════════════════════════════
# Wrapper: mixed-regime reset on top of v10.4 wrapper.
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV10_5Wrapper(HoverTrackV10Wrapper):
    """v10.4 wrapper + per-episode regime sampling.

    Each reset() chooses between near-hover (preserve v10.4) and
    perturbed (learn recovery from post-spiral states). The regime is
    surfaced via info['init_regime'] for logging.
    """

    def __init__(self, *args,
                 perturbed_init_prob=0.5,
                 perturbed_pos_range=0.5,
                 perturbed_vel_range=1.5,
                 perturbed_ang_range=0.3,
                 perturbed_ang_vel_range=1.8,
                 nearhover_pos_range=0.2,
                 nearhover_vel_range=0.10,
                 nearhover_ang_range=0.03,
                 nearhover_ang_vel_range=0.03,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.perturbed_init_prob = float(perturbed_init_prob)
        self.perturbed_pos_range = float(perturbed_pos_range)
        self.perturbed_vel_range = float(perturbed_vel_range)
        self.perturbed_ang_range = float(perturbed_ang_range)
        self.perturbed_ang_vel_range = float(perturbed_ang_vel_range)
        self.nearhover_pos_range = float(nearhover_pos_range)
        self.nearhover_vel_range = float(nearhover_vel_range)
        self.nearhover_ang_range = float(nearhover_ang_range)
        self.nearhover_ang_vel_range = float(nearhover_ang_vel_range)
        self._last_init_regime = 'nearhover'

    def reset(self, seed=None, options=None):
        is_perturbed = np.random.rand() < self.perturbed_init_prob
        self._last_init_regime = 'perturbed' if is_perturbed else 'nearhover'

        obs, info = super().reset(seed=seed, options=options)

        if not is_perturbed:
            # PURE near-hover: identical to v10.4's training distribution.
            # The v9 reset already gave us:
            #   - drone at target_xy ± jitter_xy (±0.20), z = target_z + 1.5
            #   - linear velocity = 0 exactly
            #   - quaternion = identity (level) exactly
            #   - angular velocity = 0 exactly
            # We must NOT inject any extra perturbation here, otherwise
            # v10.4 sees states it never saw during training and crashes
            # even in the supposedly "preserved" regime. This branch is
            # the anti-forgetting anchor — keep it byte-equivalent to v10.4.
            info['init_regime'] = self._last_init_regime
            return obs, info

        # PERTURBED branch: inject the post-spiral-like state v10.4 needs
        # to learn to recover from. We overwrite v9's zeros AFTER super()
        # so the env actually sees the perturbation.
        pos_r = self.perturbed_pos_range
        vel_r = self.perturbed_vel_range
        ang_r = self.perturbed_ang_range
        avel_r = self.perturbed_ang_vel_range

        state = self.base_env.state.copy()
        # Extra positional jitter on top of v9's spawn (keeps the target
        # roughly under the downward camera so the centroid stays visible).
        state[0] += float(np.random.uniform(-pos_r, pos_r))
        state[2] += float(np.random.uniform(-pos_r, pos_r))
        state[4] += float(np.random.uniform(-pos_r, pos_r))
        # Linear velocities — overwrite the zeros v9 just set.
        state[1] = float(np.random.uniform(-vel_r, vel_r))
        state[3] = float(np.random.uniform(-vel_r, vel_r))
        state[5] = float(np.random.uniform(-vel_r, vel_r))
        # Attitude — overwrite the identity quaternion v9 just set.
        eul = (np.random.rand(3) - 0.5) * 2 * ang_r
        q = euler_quat(eul).flatten()
        state[6:10] = q
        # Angular velocity — overwrite the zeros v9 just set.
        state[10:13] = (np.random.rand(3) - 0.5) * 2 * avel_r

        self.base_env.state = state.copy()
        self.base_env.previous_state = state.copy()

        # Re-render and re-capture obs from the perturbed state so the
        # policy sees the correct first observation.
        self._update_visualization()
        if self.panda3d_app is not None:
            self.panda3d_app.graphicsEngine.renderFrame()
        if self.use_camera:
            self._capture_camera_images(force_capture=True)
        obs = self._build_observation(state.astype(np.float32))

        info['init_regime'] = self._last_init_regime
        return obs, info

    def step(self, action):
        # Propagate the regime label into every step's info so the
        # callback (which only sees step infos, not reset infos) can
        # attribute episode metrics to the right regime.
        obs, reward, terminated, truncated, info = super().step(action)
        info['init_regime'] = self._last_init_regime
        return obs, reward, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════
# Lightweight progress / regime-aware metrics callback.
# ══════════════════════════════════════════════════════════════════════

class V10_5MetricsCallback(BaseCallback):
    """Tracks survival per regime so we can verify v10.5 doesn't devalue
    near-hover tracking while learning to recover from perturbed init."""

    def __init__(self, raw_env, output_dir, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.output_dir / 'training_log.csv'
        self._csv = None

        self._ep_count = 0
        self._ep_steps = 0
        self._ep_reward = 0.0
        self._ep_regime = 'nearhover'

        self._near_recent_steps = []
        self._pert_recent_steps = []

        self.start_time = time.time()
        self.last_log_time = time.time()

    def _on_training_start(self):
        import csv
        self._csv_file = open(self._csv_path, 'w', newline='')
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(['episode', 'timestep', 'regime', 'steps', 'reward'])

    def _on_step(self):
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones = self.locals.get('dones', [False])
        if not infos:
            return True
        info = infos[0]
        self._ep_steps += 1
        self._ep_reward += float(rewards[0])
        if 'init_regime' in info:
            self._ep_regime = info['init_regime']

        if dones[0]:
            self._ep_count += 1
            self._csv.writerow([
                self._ep_count, self.num_timesteps, self._ep_regime,
                self._ep_steps, round(self._ep_reward, 2),
            ])
            self._csv_file.flush()
            if self._ep_regime == 'perturbed':
                self._pert_recent_steps.append(self._ep_steps)
                if len(self._pert_recent_steps) > 50:
                    self._pert_recent_steps = self._pert_recent_steps[-50:]
            else:
                self._near_recent_steps.append(self._ep_steps)
                if len(self._near_recent_steps) > 50:
                    self._near_recent_steps = self._near_recent_steps[-50:]
            self._ep_steps = 0
            self._ep_reward = 0.0

        now = time.time()
        if now - self.last_log_time > 5.0:
            self._log()
            self.last_log_time = now
        return True

    def _log(self):
        near_m = float(np.mean(self._near_recent_steps)) if self._near_recent_steps else 0.0
        pert_m = float(np.mean(self._pert_recent_steps)) if self._pert_recent_steps else 0.0
        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / max(elapsed, 1)
        print(f"  [{self.num_timesteps:>7,} ts]  Ep={self._ep_count}  "
              f"near_mean_steps={near_m:.0f}  pert_mean_steps={pert_m:.0f}  "
              f"{fps:.0f} fps")

    def save(self):
        if self._csv is not None:
            self._csv_file.close()
            self._csv = None


# ══════════════════════════════════════════════════════════════════════
# Curriculum callback (target speed) — same shape as v10's.
# ══════════════════════════════════════════════════════════════════════

class CurriculumTargetSpeedCallback(BaseCallback):
    def __init__(self, raw_env, schedule, verbose=0):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.schedule = {int(k): float(v) for k, v in schedule.items()}
        self._applied_keys = set()

    def _on_step(self):
        ts = self.num_timesteps
        for k in sorted(self.schedule.keys()):
            if k in self._applied_keys:
                continue
            if ts >= k:
                prev = getattr(self.raw_env, '_curriculum_target_speed', None)
                self.raw_env._curriculum_target_speed = self.schedule[k]
                self._applied_keys.add(k)
                print(f"\n  [Curriculum @ {ts:,} ts] "
                      f"target_speed: {prev} -> {self.schedule[k]}")
        return True


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune v10.4 → v10.5 with mixed-regime reset")
    p.add_argument('--timesteps', type=int, default=150_000)
    p.add_argument('--output-dir', type=str,
                   default='./models/hover_track_v10_5')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')

    # Transfer learning
    p.add_argument('--init-from', type=str,
                   default='./models/hover_track_v10_4/best_model_TFG.zip')
    p.add_argument('--init-vec-norm-from', type=str,
                   default='./models/hover_track_v10_4/best_vec_normalize_TFG.pkl')

    # Mixed reset
    p.add_argument('--perturbed-init-prob', type=float, default=0.5,
                   help="Fraction of episodes with perturbed init.")
    p.add_argument('--perturbed-pos-range', type=float, default=0.5)
    p.add_argument('--perturbed-vel-range', type=float, default=1.5)
    p.add_argument('--perturbed-ang-range', type=float, default=0.3)
    p.add_argument('--perturbed-ang-vel-range', type=float, default=1.8)

    # Fine-tuning hyperparams (lower LR to preserve v10.4)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=200_000)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--tau', type=float, default=0.01)
    p.add_argument('--target-entropy', type=float, default=-1.0)
    p.add_argument('--learning-starts', type=int, default=2000)

    # Env params — match v10.4
    p.add_argument('--max-ep-steps', type=int, default=3000)
    p.add_argument('--hover-height', type=float, default=1.394)
    p.add_argument('--spawn-height', type=float, default=1.5)
    p.add_argument('--lemniscate-scale', type=float, default=2.0)
    p.add_argument('--jitter-xy', type=float, default=0.20)
    p.add_argument('--z-min', type=float, default=0.5)
    p.add_argument('--z-max', type=float, default=3.0)
    p.add_argument('--invisible-term-steps', type=int, default=100)
    p.add_argument('--w-alive', type=float, default=0.10)
    p.add_argument('--w-jerk', type=float, default=0.20)
    p.add_argument('--w-invisible', type=float, default=1.0)
    p.add_argument('--crash-penalty', type=float, default=2.0)
    p.add_argument('--reward-version', type=str, default='v3.1')
    p.add_argument('--clip-obs', type=float, default=10.0)

    # v10.4 features kept on
    p.add_argument('--action-hover-bias', type=float, default=0.45)
    p.add_argument('--last-known-centroid', action='store_true', default=True)
    p.add_argument('--target-speed-curriculum', type=str,
                   default='{"0": 0.10, "50000": 0.30}',
                   help="JSON: {step: target_speed}. v10.4 ended at 0.3.")
    # Inference-only sanity test: run the loaded model through v10.5's
    # exact pipeline (wrapper + VecNormalize + SAC.load) without any
    # gradient updates. If v10.4 fails here but works in
    # record_v10_4_demo.py, the v10.5 environment chain itself is broken.
    p.add_argument('--inference-only', action='store_true',
                   help="Skip training. Run --inference-episodes episodes "
                        "with the loaded model (predict deterministic), "
                        "report survival per regime.")
    p.add_argument('--inference-episodes', type=int, default=10)
    p.add_argument('--inference-target-speed', type=float, default=0.0,
                   help="Target speed during inference. 0.0 = static.")
    p.add_argument('--inference-perturbed-prob', type=float, default=None,
                   help="Override --perturbed-init-prob during inference. "
                        "Use 0.0 to force all near-hover (pure v10.4 baseline).")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV10_5App(ShowBase):
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

        self.curriculum_schedule = (
            json.loads(args.target_speed_curriculum)
            if args.target_speed_curriculum else {})
        use_curriculum = bool(self.curriculum_schedule)

        print("Creating env (v10.5 — mixed-regime reset)...")
        self.raw_env = HoverTrackV10_5Wrapper(
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
            # Init ranges are overwritten per-episode by the wrapper,
            # but we still need to pass non-None defaults.
            init_pos_range=0.2, init_vel_range=0.10, init_ang_range=0.03,
            init_ang_vel_range=0.03,
            reward_version=args.reward_version,
            spawn_height=args.spawn_height,
            jitter_xy=args.jitter_xy,
            w_alive=args.w_alive, w_jerk=args.w_jerk,
            w_invisible=args.w_invisible,
            z_min=args.z_min, z_max=args.z_max,
            invisible_term_steps=args.invisible_term_steps,
            crash_penalty=args.crash_penalty,
            action_hover_bias=args.action_hover_bias,
            last_known_centroid=args.last_known_centroid,
            use_curriculum_speed=use_curriculum,
            # Mixed-regime parameters
            perturbed_init_prob=args.perturbed_init_prob,
            perturbed_pos_range=args.perturbed_pos_range,
            perturbed_vel_range=args.perturbed_vel_range,
            perturbed_ang_range=args.perturbed_ang_range,
            perturbed_ang_vel_range=args.perturbed_ang_vel_range,
        )

        self.env = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # VecNormalize: load v10.4 stats if available
        vp = Path(args.init_vec_norm_from)
        if vp.exists():
            print(f"Loading VecNormalize stats from {vp}")
            self.vec_env = VecNormalize.load(str(vp), self.vec_env)
            self.vec_env.training = True
            self.vec_env.norm_reward = False
        else:
            print(f"WARNING: {vp} not found — using fresh VecNormalize.")
            self.vec_env = VecNormalize(
                self.vec_env, norm_obs=True, norm_reward=False,
                clip_obs=args.clip_obs, gamma=args.gamma,
            )

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = self.output_dir / 'tb'
        tb_dir.mkdir(parents=True, exist_ok=True)

        # Load v10.4 SAC weights
        ipath = Path(args.init_from)
        if not ipath.exists():
            raise FileNotFoundError(f"--init-from not found: {ipath}")
        print(f"Loading v10.4 weights from {ipath}")
        custom_objects = {
            'learning_rate': args.learning_rate,
            'lr_schedule': (lambda _: args.learning_rate),
        }
        self.model = SAC.load(
            str(ipath), env=self.vec_env, device='auto',
            custom_objects=custom_objects,
            tensorboard_log=str(tb_dir),
        )
        # Hyperparam overrides
        self.model.gamma = args.gamma
        self.model.tau = args.tau
        self.model.target_entropy = args.target_entropy
        self.model.learning_starts = args.learning_starts
        self.model.batch_size = args.batch_size

        total_params = sum(p.numel() for p in self.model.policy.parameters())

        print("\n" + "=" * 70)
        print("  HOVER-TRACK v10.5 — robustness fine-tuning")
        print("=" * 70)
        print(f"  Init regime mix:        "
              f"{args.perturbed_init_prob:.0%} perturbed / "
              f"{(1 - args.perturbed_init_prob):.0%} near-hover")
        print(f"  Perturbed pos:          ±{args.perturbed_pos_range} m")
        print(f"  Perturbed vel:          ±{args.perturbed_vel_range} m/s")
        print(f"  Perturbed ang:          ±{args.perturbed_ang_range} rad "
              f"(≈ {np.rad2deg(args.perturbed_ang_range):.0f}°)")
        print(f"  Perturbed ang vel:      ±{args.perturbed_ang_vel_range} rad/s")
        print(f"  Near-hover (preserves v10.4): identical to training "
              f"of v10.4 (vel ±0.10, ang ±0.03)")
        print(f"  Transfer from:          {args.init_from}")
        print(f"  VecNorm from:           {args.init_vec_norm_from}")
        print(f"  Learning rate:          {args.learning_rate}  (lower than v10.4)")
        print(f"  Curriculum:             {self.curriculum_schedule}")
        print(f"  Timesteps:              {args.timesteps:,}")
        print(f"  Params:                 {total_params:,}")
        print(f"  Output:                 {args.output_dir}")
        print("=" * 70)

    def run_inference(self):
        """Sanity check: run the loaded v10.4 model through v10.5's exact
        env / VecNormalize / model pipeline without any gradient updates.

        If near-hover episodes don't reach v10.4's baseline (~3000 steps)
        here, the v10.5 environment chain itself has a discrepancy with
        the v10.4 training conditions (so training would never preserve
        the model). If they DO reach baseline, the pipeline is fine and
        the issue is elsewhere (LR, forgetting, etc.)."""
        args = self.args

        # Freeze VecNormalize so it doesn't drift during inference.
        if hasattr(self.vec_env, 'training'):
            self.vec_env.training = False
        if hasattr(self.vec_env, 'norm_reward'):
            self.vec_env.norm_reward = False

        # Force static (or user-chosen) target speed via the curriculum
        # attribute that the v10 wrapper reads inside reset().
        self.raw_env._curriculum_target_speed = float(args.inference_target_speed)

        # Optional override of the perturbed-init probability for the test.
        if args.inference_perturbed_prob is not None:
            self.raw_env.perturbed_init_prob = float(args.inference_perturbed_prob)

        n_eps = int(args.inference_episodes)
        print("\n" + "=" * 70)
        print(f"  INFERENCE-ONLY  ({n_eps} episodes, "
              f"target_speed={args.inference_target_speed}, "
              f"perturbed_prob={self.raw_env.perturbed_init_prob})")
        print("=" * 70)

        near_steps, pert_steps = [], []
        obs = self.vec_env.reset()
        self.taskMgr.step()
        ep_steps = 0
        ep_cum_reward = 0.0
        ep_count = 0

        while ep_count < n_eps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.vec_env.step(action)
            self.taskMgr.step()
            ep_steps += 1
            ep_cum_reward += float(rewards[0])

            if dones[0]:
                regime = infos[0].get('init_regime', 'nearhover')
                term_reason = infos[0].get('v9_term_reason', '') or '(timeout)'
                ep_count += 1
                print(f"  Ep {ep_count:>2}/{n_eps}: regime={regime:>9s}  "
                      f"steps={ep_steps:>4}  cum_reward={ep_cum_reward:+.1f}  "
                      f"end={term_reason}")
                if regime == 'perturbed':
                    pert_steps.append(ep_steps)
                else:
                    near_steps.append(ep_steps)
                ep_steps = 0
                ep_cum_reward = 0.0

        print("\n" + "=" * 70)
        print("  RESULTS")
        print("=" * 70)
        if near_steps:
            print(f"  near-hover:  n={len(near_steps)}  "
                  f"mean={np.mean(near_steps):.0f}  "
                  f"max={max(near_steps)}  min={min(near_steps)}")
        if pert_steps:
            print(f"  perturbed:   n={len(pert_steps)}  "
                  f"mean={np.mean(pert_steps):.0f}  "
                  f"max={max(pert_steps)}  min={min(pert_steps)}")
        print()
        print("  Expected:")
        print("    near-hover → close to max_ep_steps "
              f"({args.max_ep_steps}). Lower means the pipeline differs "
              "from v10.4's training conditions.")
        print("    perturbed  → much lower. The model has not been trained")
        print("                 to recover from these states yet.")
        print("=" * 70)

    def run_training(self):
        args = self.args
        render_cb = Panda3DRenderCallback(self)
        self.metrics_cb = V10_5MetricsCallback(
            raw_env=self.raw_env, output_dir=str(self.output_dir))
        callbacks = [render_cb, self.metrics_cb]
        if self.curriculum_schedule:
            callbacks.append(CurriculumTargetSpeedCallback(
                raw_env=self.raw_env, schedule=self.curriculum_schedule))

        print("\nStarting training...\n")
        start = time.time()
        self.model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        elapsed = time.time() - start

        model_path = self.output_dir / 'best_model_TFG'
        self.model.save(str(model_path))
        vn_path = self.output_dir / 'best_vec_normalize_TFG.pkl'
        self.vec_env.save(str(vn_path))
        self.metrics_cb.save()

        print("\n" + "=" * 70)
        print("  v10.5 TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Time:     {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Model:    {model_path}.zip")
        print(f"  VecNorm:  {vn_path}")
        print("=" * 70)


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'window-type offscreen')
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = HoverTrackV10_5App(args)
    try:
        if args.inference_only:
            app.run_inference()
            return
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            p = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(p))
            print(f"Model saved to {p}.zip")
        if hasattr(app, 'vec_env'):
            vp = Path(args.output_dir) / 'best_vec_normalize_TFG.pkl'
            app.vec_env.save(str(vp))
        if hasattr(app, 'metrics_cb'):
            app.metrics_cb.save()


if __name__ == "__main__":
    main()
