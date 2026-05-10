#!/usr/bin/env python
"""
Hover-track v10 — iterative rollout with one feature flag per subversion.

The v10 line continues from v9.1.1 (`best_model_TFG.zip`). Each subversion
adds exactly one factor to the previous one to keep ablation-quality
attribution. See PROJECT_HISTORY.md (entry 2026-05-07) for the full plan.

Defaults reproduce v9.1.1 + transfer learning. Flags layer features:

  v10.0: defaults (TL from v9.1.1, target static)
  v10.1: + --lr-schedule linear --patience 5
  v10.2: + --action-hover-bias 0.45
  v10.3: + --last-known-centroid
  v10.4: + --target-speed-curriculum '{"0":0.0,"30000":0.1,"60000":0.3}'
  v10.5: + --frame-stack 3                       (TL incompatible: 19D vs 57D)
  v10.6: + --algo tqc                            (no TL: distributional Q)
  v10.7: + --use-layernorm --weight-decay 1e-5

Usage example:
    python scripts/train_hover_track_v10.py --no-display --variant v10.0 \\
        --init-from ./models/hover_track_v9_1_1/best_model_TFG.zip \\
        --init-vec-norm-from ./models/hover_track_v9_1_1/best_vec_normalize_TFG.pkl \\
        --timesteps 50000 --weight-sanity-check \\
        --output-dir ./models/hover_track_v10_0
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

# Force UTF-8 on stdout/stderr so unicode chars in help text and prints
# (Δ, ⚠, ²) work on Windows cp1252 consoles. No-op on terminals that
# already use UTF-8 (most Linux/macOS).
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

import cv2
import numpy as np

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from panda3d.core import Filename, loadPrcFile, loadPrcFileData

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from direct.showbase.ShowBase import ShowBase
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecNormalize, VecFrameStack)
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Optional TQC. If sb3-contrib is not installed, --algo tqc errors loudly.
try:
    from sb3_contrib import TQC
    HAVE_TQC = True
except ImportError:
    TQC = None
    HAVE_TQC = False

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.utils.episode_recorder import EpisodeRecorder

# Reuse v9 building blocks. The wrapper, callbacks and render driver are
# already battle-tested; v10 only adds capabilities on top.
from scripts.train_hover_track_v9 import (
    HoverTrackV9Wrapper,
    Panda3DRenderCallback,
    MetricsCallback,
    BestSurvivalEval,
    VecNormalizeCheckpointCallback,
    BoundedVideoCallback,
)


# ══════════════════════════════════════════════════════════════════════
# v10 wrapper: extends v9 with three optional features.
#
# Defaults are no-ops so v10.0 with default flags reproduces v9 behavior.
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV10Wrapper(HoverTrackV9Wrapper):
    """v9 wrapper + curriculum target speed + action hover bias + last-known centroid."""

    def __init__(self, *args,
                 action_hover_bias=0.0,
                 last_known_centroid=False,
                 use_curriculum_speed=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.action_hover_bias = float(action_hover_bias)
        self.last_known_centroid = bool(last_known_centroid)
        self.use_curriculum_speed = bool(use_curriculum_speed)
        # The curriculum callback writes to this; reset() reads it.
        self._curriculum_target_speed = 0.0
        self._last_visible_cxcy = np.zeros(2, dtype=np.float32)
        self._last_visible_dcxdcy = np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # v9 forces target_speed=0.0 inside reset. We override the value
        # AFTER super().reset() so the env builds correctly, then the
        # next step() onwards uses the curriculum value.
        obs, info = super().reset(seed=seed, options=options)
        if self.use_curriculum_speed:
            self.target_speed = float(self._curriculum_target_speed)
        self._last_visible_cxcy[:] = 0.0
        self._last_visible_dcxdcy[:] = 0.0
        return obs, info

    def step(self, action):
        if self.action_hover_bias != 0.0:
            action = np.asarray(action, dtype=np.float32) + self.action_hover_bias
            action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = super().step(action)

        if self.last_known_centroid:
            vt = info.get('visual_tracking', {}) or {}
            visible = bool(vt.get('target_visible', False))
            # obs layout (centroid_obs=True, 19-D):
            #   [0:13]  base state
            #   [13]    cx, [14] cy, [15] frac, [16] vis, [17] dcx, [18] dcy
            obs = np.asarray(obs, dtype=np.float32).copy()
            if visible:
                self._last_visible_cxcy[:] = obs[13:15]
                self._last_visible_dcxdcy[:] = obs[17:19]
            else:
                obs[13:15] = self._last_visible_cxcy
                obs[17:19] = self._last_visible_dcxdcy

        return obs, reward, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════
# Curriculum callback: drives wrapper._curriculum_target_speed by step.
# ══════════════════════════════════════════════════════════════════════

class TargetSpeedCurriculumCallback(BaseCallback):
    """Step-function curriculum: at training step >= ts_i set target_speed to v_i."""

    def __init__(self, raw_env, schedule, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.schedule = sorted(((int(k), float(v)) for k, v in schedule.items()),
                               key=lambda kv: kv[0])
        self._announced = set()

    def _on_step(self):
        cur = 0.0
        for ts, sp in self.schedule:
            if self.num_timesteps >= ts:
                cur = sp
        if hasattr(self.raw_env, '_curriculum_target_speed'):
            prev = self.raw_env._curriculum_target_speed
            self.raw_env._curriculum_target_speed = cur
            if cur != prev and self.verbose:
                print(f"  [Curriculum] step={self.num_timesteps:,}  "
                      f"target_speed: {prev:.2f} -> {cur:.2f}")
        return True


# ══════════════════════════════════════════════════════════════════════
# Patience early-stop: returns False (stops training) after N evals
# without an update of eval_cb.best_step.
# ══════════════════════════════════════════════════════════════════════

class PatienceEarlyStopCallback(BaseCallback):
    def __init__(self, eval_cb, patience, verbose=1):
        super().__init__(verbose)
        self.eval_cb = eval_cb
        self.patience = int(patience)
        self._last_seen_best_step = -1
        self._evals_without_improvement = 0
        self._last_eval_count = 0

    def _on_step(self):
        if self.patience <= 0:
            return True
        n = len(self.eval_cb.eval_log)
        if n <= self._last_eval_count:
            return True
        self._last_eval_count = n
        if self.eval_cb.best_step > self._last_seen_best_step:
            self._last_seen_best_step = self.eval_cb.best_step
            self._evals_without_improvement = 0
        else:
            self._evals_without_improvement += 1
            if self.verbose:
                print(f"  [Patience] {self._evals_without_improvement}/"
                      f"{self.patience} evals without improvement")
            if self._evals_without_improvement >= self.patience:
                print(f"\n[Patience] {self.patience} consecutive evals "
                      f"without best update -> stopping training.")
                return False
        return True


# ══════════════════════════════════════════════════════════════════════
# Weight sanity check: snapshot at training_start, delta print at +N steps.
# Prints a per-layer mean |Δw| with FROZEN flag if Δ < 1e-7.
# Closes audit point #1 ("are weights actually updating?") empirically.
# ══════════════════════════════════════════════════════════════════════

class WeightSanityCallback(BaseCallback):
    def __init__(self, check_after_steps=2000, threshold=1e-7, verbose=1):
        super().__init__(verbose)
        self.check_after = int(check_after_steps)
        self.threshold = float(threshold)
        self._snapshot = None
        self._done = False

    def _on_training_start(self):
        self._snapshot = {n: p.detach().cpu().clone()
                          for n, p in self.model.policy.named_parameters()}

    def _on_step(self):
        if self._done or self.num_timesteps < self.check_after:
            return True
        any_frozen = False
        print(f"\n[WeightSanity] Mean |Δw| per layer after "
              f"{self.num_timesteps:,} training steps:")
        for n, p in self.model.policy.named_parameters():
            d = (self._snapshot[n] - p.detach().cpu()).abs().mean().item()
            mark = "  ⚠ FROZEN" if d < self.threshold else ""
            if d < self.threshold:
                any_frozen = True
            print(f"  {n:>55}  Δ={d:.3e}{mark}")
        if any_frozen:
            print("[WeightSanity] WARNING: some layers below threshold. "
                  "Verify gradient flow.")
        else:
            print("[WeightSanity] All layers updating above threshold. OK.")
        self._done = True
        return True


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Hover-track v10 — iterative rollout from v9.1.1")

    # Documentation
    p.add_argument('--variant', type=str, default='v10.0',
                   help="Documentation tag (e.g. v10.0..v10.7). "
                        "Does not affect computation.")

    # Run config
    p.add_argument('--timesteps', type=int, default=50_000)
    p.add_argument('--output-dir', type=str,
                   default='./models/hover_track_v10_0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')

    # Algorithm
    p.add_argument('--algo', choices=['sac', 'tqc'], default='sac')
    p.add_argument('--top-quantiles-to-drop', type=int, default=2,
                   help="TQC only. SB3-contrib default is 2.")

    # Transfer learning
    p.add_argument('--init-from', type=str, default='',
                   help="Path to .zip with weights to initialize from. "
                        "Default: empty (fresh init). Recommended for "
                        "v10.0-v10.4: ./models/hover_track_v9_1_1/best_model_TFG.zip")
    p.add_argument('--init-vec-norm-from', type=str, default='',
                   help="Path to matching VecNormalize .pkl.")

    # Optimizer / SAC core
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--lr-schedule', choices=['constant', 'linear'],
                   default='constant')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=200_000)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--tau', type=float, default=0.01)
    p.add_argument('--target-entropy', type=float, default=-1.0)
    p.add_argument('--learning-starts', type=int, default=5000)
    p.add_argument('--weight-decay', type=float, default=0.0,
                   help=">0 switches Adam -> AdamW with this weight_decay.")

    # Reward weights (same as v9)
    p.add_argument('--w-alive', type=float, default=0.10)
    p.add_argument('--w-jerk', type=float, default=0.20)
    p.add_argument('--w-invisible', type=float, default=1.0)
    p.add_argument('--crash-penalty', type=float, default=2.0)

    # Spawn / termination
    p.add_argument('--hover-height', type=float, default=1.394)
    p.add_argument('--spawn-height', type=float, default=1.5)
    p.add_argument('--max-ep-steps', type=int, default=3000)
    p.add_argument('--lemniscate-scale', type=float, default=2.0)
    p.add_argument('--jitter-xy', type=float, default=0.20)
    p.add_argument('--z-min', type=float, default=0.5)
    p.add_argument('--z-max', type=float, default=3.0)
    p.add_argument('--invisible-term-steps', type=int, default=100)

    # v10 features (default OFF -> v9.1.1 behaviour)
    p.add_argument('--patience', type=int, default=0,
                   help="Early stop: 0=disabled. Recommended: 5 from v10.1.")
    p.add_argument('--action-hover-bias', type=float, default=0.0,
                   help="Add to action before clip. 0.45 -> action 0 ~ hover.")
    p.add_argument('--last-known-centroid', action='store_true',
                   help="When target invisible, replace cx,cy,dcx,dcy with "
                        "values from last visible frame (eliminates discontinuity).")
    p.add_argument('--target-speed-curriculum', type=str, default='',
                   help="JSON dict {timestep:speed}. Empty = static target.")
    p.add_argument('--frame-stack', type=int, default=1,
                   help="VecFrameStack n_stack. 1 = no stacking. >1 breaks "
                        "TL from v9.1.1 (obs dim mismatch).")
    p.add_argument('--use-layernorm', action='store_true',
                   help="LayerNorm on actor MLP. v10.7.")

    # Normalisation / observation
    p.add_argument('--no-vec-normalize', action='store_true')
    p.add_argument('--clip-obs', type=float, default=10.0)

    # Diagnostics
    p.add_argument('--weight-sanity-check', action='store_true',
                   help="Snapshot weights at start, print Δw at +2k steps.")

    # Eval / videos
    p.add_argument('--eval-freq', type=int, default=10_000)
    p.add_argument('--n-eval-episodes', type=int, default=5)
    p.add_argument('--checkpoint-freq', type=int, default=25_000)
    p.add_argument('--max-videos', type=int, default=8)
    p.add_argument('--record-fps', type=int, default=10)

    # Reward version (kept for compatibility — base env reward is discarded
    # by the wrapper, but the parameter still configures the base env).
    p.add_argument('--reward-version', type=str, default='v3.1')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Optional LayerNorm policy_kwargs for v10.7
# ══════════════════════════════════════════════════════════════════════

def _make_policy_kwargs(args):
    """
    SB3 net_arch=[256, 128] is the v9.1.1 default. We don't override the
    feature extractor here because LayerNorm in SB3 stock policies is not
    trivially configurable from policy_kwargs alone. v10.7 uses
    activation_fn=nn.GELU as a soft proxy plus AdamW (weight_decay) and
    documents the limitation. A full LayerNorm-on-actor variant requires
    a custom BasePolicy and is deferred to v10.7+.
    """
    kwargs = dict(net_arch=[256, 128])
    if args.use_layernorm:
        kwargs['activation_fn'] = nn.GELU
    if args.weight_decay > 0.0:
        # SB3 builds optimizer via policy.optimizer_class + optimizer_kwargs
        kwargs['optimizer_class'] = torch.optim.AdamW
        kwargs['optimizer_kwargs'] = dict(weight_decay=args.weight_decay)
    return kwargs


# ══════════════════════════════════════════════════════════════════════
# LR schedule
# ══════════════════════════════════════════════════════════════════════

def _make_lr(args):
    if args.lr_schedule == 'linear':
        # SB3 convention: schedule(progress_remaining) where 1.0 -> start.
        lr0 = float(args.learning_rate)
        return lambda progress_remaining: progress_remaining * lr0
    return float(args.learning_rate)


# ══════════════════════════════════════════════════════════════════════
# Application
# ══════════════════════════════════════════════════════════════════════

class HoverTrackV10App(ShowBase):
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

        # ── Curriculum schedule parsing ──
        self.curriculum_schedule = {}
        if args.target_speed_curriculum:
            self.curriculum_schedule = json.loads(args.target_speed_curriculum)
            print(f"Target-speed curriculum: {self.curriculum_schedule}")
        use_curriculum = bool(self.curriculum_schedule)

        # ── Wrapper (v10 extends v9) ──
        print(f"Creating env (hover-track {args.variant})...")
        self.raw_env = HoverTrackV10Wrapper(
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
            reward_version=args.reward_version,
            spawn_height=args.spawn_height,
            jitter_xy=args.jitter_xy,
            w_alive=args.w_alive, w_jerk=args.w_jerk,
            w_invisible=args.w_invisible,
            z_min=args.z_min, z_max=args.z_max,
            invisible_term_steps=args.invisible_term_steps,
            crash_penalty=args.crash_penalty,
            # v10-specific
            action_hover_bias=args.action_hover_bias,
            last_known_centroid=args.last_known_centroid,
            use_curriculum_speed=use_curriculum,
        )

        self.env = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # ── VecNormalize ──
        # If --init-vec-norm-from is given AND --frame-stack=1 AND
        # not --no-vec-normalize, load existing stats; else create new.
        self.vec_normalize = None
        loaded_vec_norm = False
        if not args.no_vec_normalize:
            if args.init_vec_norm_from and args.frame_stack == 1:
                vp = Path(args.init_vec_norm_from)
                if vp.exists():
                    print(f"Loading VecNormalize stats from {vp}")
                    self.vec_env = VecNormalize.load(str(vp), self.vec_env)
                    self.vec_env.training = True  # keep updating during v10 training
                    self.vec_env.norm_reward = False
                    self.vec_normalize = self.vec_env
                    loaded_vec_norm = True
                else:
                    print(f"WARNING: --init-vec-norm-from {vp} not found, "
                          f"creating fresh VecNormalize.")
            if not loaded_vec_norm:
                self.vec_env = VecNormalize(
                    self.vec_env,
                    norm_obs=True,
                    norm_reward=False,
                    clip_obs=args.clip_obs,
                    gamma=args.gamma,
                )
                self.vec_normalize = self.vec_env

        # ── Frame stacking (after VecNormalize per SB3 convention) ──
        if args.frame_stack > 1:
            print(f"VecFrameStack with n_stack={args.frame_stack}")
            self.vec_env = VecFrameStack(self.vec_env, n_stack=args.frame_stack)

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = self.output_dir / 'tb'
        tb_dir.mkdir(parents=True, exist_ok=True)

        # ── Algorithm selection ──
        if args.algo == 'tqc':
            if not HAVE_TQC:
                raise RuntimeError(
                    "--algo tqc requires sb3-contrib. Install with:\n"
                    "    pip install sb3-contrib")
            MODEL_CLS = TQC
        else:
            MODEL_CLS = SAC

        lr = _make_lr(args)
        policy_kwargs = _make_policy_kwargs(args)

        common_kwargs = dict(
            policy='MlpPolicy', env=self.vec_env,
            learning_rate=lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            gamma=args.gamma,
            tau=args.tau,
            ent_coef='auto',
            target_entropy=args.target_entropy,
            learning_starts=args.learning_starts,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(tb_dir),
        )
        if args.algo == 'tqc':
            common_kwargs['top_quantiles_to_drop_per_net'] = (
                args.top_quantiles_to_drop)

        # ── Model creation: from scratch OR transfer learning ──
        loaded_model = False
        if args.init_from:
            ipath = Path(args.init_from)
            if not ipath.exists():
                raise FileNotFoundError(f"--init-from not found: {ipath}")
            if args.algo == 'tqc':
                print(f"WARNING: --init-from with --algo tqc is not supported "
                      f"(SAC and TQC have different critic architectures). "
                      f"Ignoring --init-from and starting fresh.")
            elif args.frame_stack > 1:
                print(f"WARNING: --frame-stack {args.frame_stack} with "
                      f"--init-from is not supported (obs dim mismatch). "
                      f"Ignoring --init-from and starting fresh.")
            else:
                print(f"Loading SAC weights from {ipath}")
                custom_objects = {
                    'learning_rate': lr,
                    'lr_schedule': (lr if callable(lr)
                                    else (lambda _: lr)),
                }
                self.model = MODEL_CLS.load(
                    str(ipath), env=self.vec_env, device='auto',
                    custom_objects=custom_objects,
                    tensorboard_log=str(tb_dir),
                )
                # Override hyperparams that may differ from the saved model.
                self.model.gamma = args.gamma
                self.model.tau = args.tau
                self.model.target_entropy = args.target_entropy
                self.model.learning_starts = args.learning_starts
                self.model.batch_size = args.batch_size
                # Note: replay buffer is NOT restored (we want fresh data
                # for the new MDP/curriculum/wrapper-feature change).
                loaded_model = True

        if not loaded_model:
            self.model = MODEL_CLS(**common_kwargs)

        total_params = sum(p.numel() for p in self.model.policy.parameters())

        print("\n" + "=" * 70)
        print(f"  HOVER-TRACK {args.variant.upper()} — iterative rollout")
        print("=" * 70)
        print(f"  Algorithm:            {args.algo.upper()}"
              + (f"  (top_quantiles_to_drop={args.top_quantiles_to_drop})"
                 if args.algo == 'tqc' else ""))
        print(f"  Transfer learning:    {'YES from ' + args.init_from if loaded_model else 'NO (fresh init)'}")
        print(f"  VecNormalize:         "
              f"{'LOADED' if loaded_vec_norm else 'FRESH' if self.vec_normalize else 'OFF'}")
        print(f"  Frame stack:          {args.frame_stack}")
        print(f"  Reward (same as v9):  r_track + r_alive + r_jerk")
        print(f"                        r_track = exp(-3*dist²) "
              f"(or -{args.w_invisible} if invis)")
        print(f"                        r_alive = +{args.w_alive}/step")
        print(f"                        r_jerk  = -{args.w_jerk} * ||Δa||²")
        print(f"  Spawn:                z={args.spawn_height} m  "
              f"XY±{args.jitter_xy} m")
        print(f"  Termination:          z<{args.z_min} or z>{args.z_max} (rel target)")
        print(f"                        crash_penalty = -{args.crash_penalty}")
        print(f"  ──── v10 feature flags ────")
        print(f"  LR schedule:          {args.lr_schedule}  (start={args.learning_rate})")
        print(f"  Patience early-stop:  {args.patience}")
        print(f"  Action hover bias:    {args.action_hover_bias}")
        print(f"  Last-known centroid:  {'YES' if args.last_known_centroid else 'no'}")
        print(f"  Target curriculum:    {self.curriculum_schedule or 'static (off)'}")
        print(f"  LayerNorm/AdamW:      "
              f"{'ON' if args.use_layernorm or args.weight_decay > 0 else 'off'}"
              f"  (wd={args.weight_decay})")
        print(f"  Weight sanity check:  {'YES @ +2k' if args.weight_sanity_check else 'no'}")
        print(f"  ──── core hyperparams ────")
        print(f"  gamma:                {args.gamma}")
        print(f"  tau:                  {args.tau}")
        print(f"  target_entropy:       {args.target_entropy}")
        print(f"  learning_starts:      {args.learning_starts:,}")
        print(f"  batch / buffer:       {args.batch_size} / {args.buffer_size:,}")
        print(f"  Policy:               MlpPolicy [256, 128] ({total_params:,} params)")
        print(f"  Timesteps:            {args.timesteps:,}")
        print(f"  Episode steps:        {args.max_ep_steps} ({args.max_ep_steps/100:.0f} s)")
        print(f"  Eval / checkpoint:    every {args.eval_freq:,} / "
              f"{args.checkpoint_freq:,} steps")
        print(f"  TensorBoard:          {tb_dir}")
        print(f"  Output:               {args.output_dir}")
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
            vec_normalize=self.vec_normalize,
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
        vec_norm_ckpt_cb = VecNormalizeCheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(ckpt_dir),
            vec_normalize=self.vec_normalize)

        callbacks = [render_cb, self.metrics_cb,
                     ckpt_cb, vec_norm_ckpt_cb, self.eval_cb]

        if args.weight_sanity_check:
            # Fire 2k steps AFTER the first SAC update (i.e. learning_starts).
            # Before learning_starts, SAC only collects data and does no
            # gradient updates, so a check there spuriously reports FROZEN.
            callbacks.append(WeightSanityCallback(
                check_after_steps=args.learning_starts + 2000))

        if self.curriculum_schedule:
            callbacks.append(TargetSpeedCurriculumCallback(
                raw_env=self.raw_env, schedule=self.curriculum_schedule))

        if args.patience > 0:
            callbacks.append(PatienceEarlyStopCallback(
                eval_cb=self.eval_cb, patience=args.patience))

        self.video_cb = None
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
            reset_num_timesteps=True,
            tb_log_name=args.variant.replace('.', '_'))
        elapsed = time.time() - start

        final_path = self.output_dir / 'final_model'
        self.model.save(str(final_path))
        if self.vec_normalize is not None:
            self.vec_normalize.save(
                str(self.output_dir / 'final_vec_normalize.pkl'))

        self.metrics_cb.save_summary(extras={
            'total_timesteps': args.timesteps,
            'training_time_seconds': elapsed,
            'algorithm': args.algo.upper(),
            'variant': args.variant,
            'transfer_learning_from': args.init_from or None,
            'vec_normalize_loaded_from': (args.init_vec_norm_from
                                          if self.vec_normalize else None),
            'lr_schedule': args.lr_schedule,
            'patience': args.patience,
            'action_hover_bias': args.action_hover_bias,
            'last_known_centroid': args.last_known_centroid,
            'curriculum_schedule': self.curriculum_schedule or None,
            'frame_stack': args.frame_stack,
            'use_layernorm': args.use_layernorm,
            'weight_decay': args.weight_decay,
            'sac_hyperparams': {
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'buffer_size': args.buffer_size,
                'gamma': args.gamma,
                'tau': args.tau,
                'target_entropy': args.target_entropy,
                'learning_starts': args.learning_starts,
            },
            'reward_weights': {'alive': args.w_alive, 'jerk': args.w_jerk,
                               'invisible': args.w_invisible,
                               'crash_penalty': args.crash_penalty},
            'spawn': {'height_m': args.spawn_height,
                      'jitter_xy': args.jitter_xy},
            'termination': {'z_min': args.z_min, 'z_max': args.z_max,
                            'invisible_steps': args.invisible_term_steps},
            'eval': {
                'best_survival': self.eval_cb.best_survival,
                'best_steps': self.eval_cb.best_steps,
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
              f"steps={self.eval_cb.best_steps:.0f}  "
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
        loadPrcFileData('', 'window-type offscreen')
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = HoverTrackV10App(args)
    try:
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            p = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(p))
            if app.vec_normalize is not None:
                app.vec_normalize.save(
                    str(Path(args.output_dir) / 'interrupted_vec_normalize.pkl'))
            print(f"Model saved to {p}.zip")
        if hasattr(app, 'metrics_cb'):
            app.metrics_cb.save_summary()
        if hasattr(app, 'video_cb') and app.video_cb is not None:
            app.video_cb.compile_timelapse()


if __name__ == "__main__":
    main()
