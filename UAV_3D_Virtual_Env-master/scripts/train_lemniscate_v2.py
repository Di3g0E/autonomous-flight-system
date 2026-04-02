#!/usr/bin/env python
"""
Train a PPO agent to follow a lemniscate trajectory (v2).

Key improvements over v1:
  - Multi-component dense reward (stability, centering, scale, discovery)
  - Transfer learning from goal_controller/best_model.zip
  - 3-phase curriculum: fixed target -> slow lemniscate -> full speed
  - Constrained near-hover initialisation with progressive domain randomisation
  - Logs all reward components for debugging

Usage:
    python scripts/train_lemniscate_v2.py --timesteps 1000000
    python scripts/train_lemniscate_v2.py --timesteps 500000 --no-transfer
"""

import argparse
import csv
import json
import os
import sys
import time
import numpy as np
from collections import deque
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.agents.feature_extractors import StateCameraExtractor
from src.utils.episode_recorder import EpisodeRecorder


# ──────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────

class Panda3DRenderCallback(BaseCallback):
    """Advance Panda3D's task manager on every SB3 training step."""
    def __init__(self, app, verbose=0):
        super().__init__(verbose)
        self.app = app

    def _on_step(self):
        self.app.taskMgr.step()
        return True


class LemniscateV2Callback(BaseCallback):
    """
    Metrics, 3-phase curriculum, and periodic recording for v2 training.

    Phases (by training progress):
      A  [0 , phase_b)  fixed target at 2 m, speed = 0
      B  [phase_b, phase_c)  moving lemniscate, speed ramps from initial to mid
      C  [phase_c, 1.0]      moving lemniscate, speed ramps from mid to max
    """

    CSV_HEADERS = [
        'episode', 'timestep', 'reward', 'steps',
        'mean_distance', 'visibility_pct',
        'mean_fraction', 'mean_fraction_error',
        'target_speed', 'phase',
        # v2 reward components (means over episode)
        'r_survival', 'r_stability', 'r_centering',
        'r_scale', 'r_discovery', 'r_not_visible',
    ]

    def __init__(self, env, output_dir, *,
                 initial_speed=0.02, max_speed=0.3,
                 phase_b=0.30, phase_c=0.70,
                 ent_coef_base=0.01, ent_coef_bump=0.03,
                 ent_coef_decay_chunks=30,
                 video_recorder=None, record_interval=50,
                 metrics_window=50, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.episode_count = 0
        self.episode_rewards = deque(maxlen=metrics_window)

        # CSV
        self.csv_path = self.output_dir / 'training_log.csv'
        self._csv_file = None
        self._csv_writer = None

        # Accumulators
        self._reset_accum()

        # Phase boundaries
        self.phase_b = phase_b
        self.phase_c = phase_c
        self._prev_phase = 'A'

        # Speed curriculum
        self.initial_speed = initial_speed
        self.max_speed = max_speed
        self.mid_speed = (initial_speed + max_speed) / 2
        self.current_speed = 0.0
        self.current_phase = 'A'

        # Entropy coefficient bump on phase transitions
        self.ent_coef_base = ent_coef_base
        self.ent_coef_bump = ent_coef_bump
        self.ent_coef_decay_chunks = ent_coef_decay_chunks
        self._ent_bump_chunk = -999  # chunk when last bump happened

        # Feature-extractor freeze tracking
        self._extractor_unfrozen = False

        # Performance-based domain randomisation
        self._dr_level = 0.0  # 0 = easiest, 1 = hardest

        # Recording
        self.video_recorder = video_recorder
        self.record_interval = record_interval
        self._chunk = 0

        self.start_time = time.time()
        self.last_log_time = time.time()

    # ── accumulators ──────────────────────────────────────────────────
    def _reset_accum(self):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_distances = []
        self._ep_visible = 0
        self._ep_fractions = []
        self._ep_errors = []
        # v2 component accumulators
        self._ep_r_survival = []
        self._ep_r_stability = []
        self._ep_r_centering = []
        self._ep_r_scale = []
        self._ep_r_discovery = 0.0
        self._ep_r_not_visible = []

    # ── lifecycle hooks ───────────────────────────────────────────────
    def _on_training_start(self):
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADERS)

    def _on_rollout_end(self):
        self._chunk += 1
        raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env

        # ── Curriculum ────────────────────────────────────────────────
        total = self.locals.get(
            'total_timesteps',
            getattr(self.model, '_total_timesteps', 1))
        progress = min(self.num_timesteps / max(total, 1), 1.0)

        if progress < self.phase_b:
            self.current_phase = 'A'
            self.current_speed = 0.0
            raw.target_mode = 'fixed'
        elif progress < self.phase_c:
            self.current_phase = 'B'
            local_p = (progress - self.phase_b) / (self.phase_c - self.phase_b)
            self.current_speed = self.initial_speed + local_p * (self.mid_speed - self.initial_speed)
            raw.target_mode = 'moving'
        else:
            self.current_phase = 'C'
            local_p = (progress - self.phase_c) / (1.0 - self.phase_c)
            self.current_speed = self.mid_speed + local_p * (self.max_speed - self.mid_speed)
            raw.target_mode = 'moving'

        raw.target_speed = self.current_speed

        # ── Phase transition: ent_coef bump ───────────────────────────
        # On each A→B or B→C transition, temporarily increase entropy to
        # encourage re-exploration under the new task dynamics.  The bump
        # decays linearly back to base over `ent_coef_decay_chunks` rollouts.
        if self.current_phase != self._prev_phase:
            self._ent_bump_chunk = self._chunk
            print(f"\n  Phase transition {self._prev_phase} -> "
                  f"{self.current_phase}  |  ent_coef bumped to "
                  f"{self.ent_coef_base + self.ent_coef_bump:.3f}")
            self._prev_phase = self.current_phase

        chunks_since_bump = self._chunk - self._ent_bump_chunk
        if chunks_since_bump < self.ent_coef_decay_chunks:
            decay = 1.0 - chunks_since_bump / self.ent_coef_decay_chunks
            self.model.ent_coef = self.ent_coef_base + self.ent_coef_bump * decay
        else:
            self.model.ent_coef = self.ent_coef_base

        # ── Feature extractor: freeze in Phase A, unfreeze in B ───────
        # During Phase A the CNN weights from transfer learning are kept
        # frozen so only the policy/value heads adapt to the new reward.
        # At Phase B we unfreeze with a dedicated low lr param group.
        if self.current_phase != 'A' and not self._extractor_unfrozen:
            extractor = self.model.policy.features_extractor
            for p in extractor.parameters():
                p.requires_grad = True
            # Add extractor params as a separate param group with low lr
            self.model.policy.optimizer.add_param_group({
                'params': list(extractor.parameters()),
                'lr': 1e-5,
            })
            self._extractor_unfrozen = True
            print(f"  Feature extractor unfrozen (lr=1e-5)")

        # ── Domain randomisation tied to performance ──────────────────
        # Instead of widening init ranges by temporal progress, we gate
        # it on the rolling mean reward.  The DR level only ratchets UP
        # (never decreases) to avoid oscillation.
        #
        # Thresholds (reward per step, approximate):
        #   hover-only ≈ 0.55/step → raw ep reward ≈ 0.55 * ep_len
        #   good tracking ≈ 4.0/step → raw ep reward ≈ 4.0 * ep_len
        # We use total episode reward normalised by max_ep_steps.
        if self.episode_rewards:
            mean_r = float(np.mean(self.episode_rewards))
            max_steps = raw.base_env.n
            normalised = mean_r / max(max_steps, 1)
            # Map normalised reward [0.5, 3.0] → DR level [0, 1]
            new_level = np.clip((normalised - 0.5) / 2.5, 0.0, 1.0)
            self._dr_level = max(self._dr_level, new_level)  # ratchet up

        dr = self._dr_level
        raw.init_pos_range = 0.2 + dr * 0.8       # 0.2 → 1.0
        raw.init_vel_range = 0.1 + dr * 0.4       # 0.1 → 0.5
        raw.init_ang_range = 0.05 + dr * 0.15     # 0.05 → 0.20

        # Collect ep rewards from SB3 buffer
        if hasattr(self.model, 'ep_info_buffer'):
            for ep in self.model.ep_info_buffer:
                if 'r' in ep:
                    self.episode_rewards.append(ep['r'])

        # Log every 5s
        now = time.time()
        if now - self.last_log_time > 5.0:
            self._log()
            self.last_log_time = now

        # Record
        if self.video_recorder and self._chunk % self.record_interval == 0:
            self._record()

    def _on_step(self):
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones = self.locals.get('dones', [False])

        if infos:
            info = infos[0]
            self._ep_reward += float(rewards[0])
            self._ep_steps += 1

            # Distance
            t = info.get('target', {})
            if 'distance_to_target' in t:
                self._ep_distances.append(t['distance_to_target'])

            # Visual tracking (v2 components)
            vt = info.get('visual_tracking', {})
            if vt:
                if vt.get('target_visible', False):
                    self._ep_visible += 1
                    self._ep_fractions.append(vt.get('target_fraction', 0))
                    self._ep_errors.append(vt.get('fraction_error', 0))
                # Components
                self._ep_r_survival.append(vt.get('r_survival', 0))
                self._ep_r_stability.append(vt.get('r_stability', 0))
                self._ep_r_centering.append(vt.get('r_centering', 0))
                self._ep_r_scale.append(vt.get('r_scale', 0))
                self._ep_r_discovery += vt.get('r_discovery', 0)
                self._ep_r_not_visible.append(vt.get('r_not_visible', 0))

            if dones[0]:
                self._write_csv()
                self._reset_accum()

        return True

    # ── CSV ────────────────────────────────────────────────────────────
    def _write_csv(self):
        if not self._csv_writer:
            return
        self.episode_count += 1
        d = self._ep_distances
        _m = lambda lst: round(float(np.mean(lst)), 4) if lst else 0.0
        self._csv_writer.writerow([
            self.episode_count,
            self.num_timesteps,
            round(self._ep_reward, 2),
            self._ep_steps,
            round(float(np.mean(d)), 3) if d else 0,
            round(100 * self._ep_visible / max(self._ep_steps, 1), 1),
            _m(self._ep_fractions),
            _m(self._ep_errors),
            round(self.current_speed, 4),
            self.current_phase,
            _m(self._ep_r_survival),
            _m(self._ep_r_stability),
            _m(self._ep_r_centering),
            _m(self._ep_r_scale),
            round(self._ep_r_discovery, 2),
            _m(self._ep_r_not_visible),
        ])
        self._csv_file.flush()

    # ── Logging ────────────────────────────────────────────────────────
    def _log(self):
        elapsed = time.time() - self.start_time
        ts = self.num_timesteps
        total = self.locals.get(
            'total_timesteps',
            getattr(self.model, '_total_timesteps', 0))
        pct = 100 * ts / max(total, 1)
        fps = ts / max(elapsed, 1)
        mean_r = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0

        # VecNormalize ratio check: compare normalised tracking vs hover
        # Raw tracking ≈ 6.05/step, raw hover ≈ 0.55/step → ratio ≈ 11×
        # After normalisation, ratio should remain ≥ 5× to preserve signal.
        ratio_str = ""
        vec_env = getattr(self.model, 'env', None)
        if vec_env is not None and hasattr(vec_env, 'ret_rms'):
            ret_std = max(float(np.sqrt(vec_env.ret_rms.var)), 1e-8)
            norm_tracking = 6.05 / ret_std
            norm_hover = 0.55 / ret_std
            ratio = norm_tracking / max(norm_hover, 1e-8)
            ratio_str = f" ratio={ratio:.1f}x"
            if ratio < 5.0:
                ratio_str += " [!LOW]"

        print(f"  [{pct:5.1f}%] Step {ts:>7,}/{total:,} | "
              f"Ep={self.episode_count} | R={mean_r:7.1f} | "
              f"Phase={self.current_phase} Speed={self.current_speed:.3f} | "
              f"DR={self._dr_level:.2f} | "
              f"{fps:.0f} fps{ratio_str}")

    # ── Recording ─────────────────────────────────────────────────────
    def _record(self):
        print(f"\n  Recording eval episode (chunk {self._chunk}, "
              f"phase {self.current_phase})...")
        self.video_recorder.start_episode(self._chunk)
        obs, info = self.env.reset()
        done = False
        step = 0
        total_r = 0
        while not done and step < 1000:
            obs_t = {k: v[np.newaxis, ...] for k, v in obs.items()}
            action, _ = self.model.predict(obs_t, deterministic=True)
            obs, r, term, trunc, info = self.env.step(action.squeeze())
            done = term or trunc
            total_r += r
            step += 1
            raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            fpv = raw._last_high_freq_image
            bird = None
            if hasattr(raw, '_bird_camera') and raw._bird_camera:
                ok, rgba = raw._bird_camera.get_image()
                if ok and rgba is not None:
                    bird = rgba[:, :, :3]
            t_info = info.get('target', {})
            overlay = {
                'Chunk': self._chunk,
                'Phase': self.current_phase,
                'Step': f"{step}/1000",
                'Timestep': f"{self.num_timesteps:,}",
                'Reward': round(total_r, 1),
                'Distance': t_info.get('distance_to_target', 0),
                'target': t_info,
                'visual_tracking': info.get('visual_tracking', {}),
            }
            self.video_recorder.capture_frame(fpv, bird, info=overlay)
        self.video_recorder.end_episode()
        print(f"  Done: {step} steps, R={total_r:.1f}")
        self.env.reset()

    # ── Save ──────────────────────────────────────────────────────────
    def save_metrics(self, extras=None):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        summary = {
            'total_episodes': self.episode_count,
            'final_mean_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'final_speed': self.current_speed,
            'final_phase': self.current_phase,
            'csv_path': str(self.csv_path),
        }
        if extras:
            summary.update(extras)
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train lemniscate follower v2 (multi-component reward)")
    p.add_argument('--timesteps', type=int, default=1_000_000)
    p.add_argument('--scale', type=float, default=5.0,
                   help="Lemniscate half-width (m)")
    p.add_argument('--initial-speed', type=float, default=0.02)
    p.add_argument('--max-speed', type=float, default=0.3)
    p.add_argument('--phase-b', type=float, default=0.30,
                   help="Progress fraction when phase B starts")
    p.add_argument('--phase-c', type=float, default=0.70,
                   help="Progress fraction when phase C starts")
    p.add_argument('--n-steps', type=int, default=4096,
                   help="PPO rollout buffer (larger than v1)")
    p.add_argument('--max-ep-steps', type=int, default=1000)
    p.add_argument('--output-dir', type=str,
                   default='./models/lemniscate_v2')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--record', action='store_true', default=False)
    p.add_argument('--record-interval', type=int, default=50)
    p.add_argument('--load-model', type=str, default=None,
                   help="Path to .zip to resume training")
    p.add_argument('--no-transfer', action='store_true',
                   help="Skip transfer learning from goal_controller")
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Training app
# ──────────────────────────────────────────────────────────────────────

class LemniscateV2App(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D scene...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # FPV camera
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Bird camera
        self.bird_camera = None
        if args.record:
            self.bird_camera = opencv_camera(self, 'bird_cam', 1)
            self.bird_camera.cam.reparentTo(self.render)
            self.bird_camera.cam.setPos(0, -12, 16)
            self.bird_camera.cam.lookAt(0, 0, 5)
            self.bird_camera.buffer.setActive(1)

        # Environment — Phase A starts with fixed target
        print("Creating environment (v2 reward)...")
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='fixed',           # Phase A: static
            target_range=3.0,
            target_speed=0.0,              # Phase A: no movement
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            enable_collisions=False,
            n=args.max_ep_steps,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            filming_mode=True,
            lemniscate_scale=args.scale,
            ideal_fraction=0.25,
            # ── v2 flags ──
            use_new_reward=True,
            initial_target_distance=2.0,
            constrained_init=True,
            init_pos_range=0.2,
            init_vel_range=0.1,
            init_ang_range=0.05,
        )
        self.env = Monitor(self.env)
        if self.bird_camera:
            self.env.unwrapped._bird_camera = self.bird_camera
        self.vec_env = DummyVecEnv([lambda: self.env])
        # Normalise rewards only (not observations — images would break).
        # gamma matches PPO's discount so the running variance tracks
        # discounted returns, keeping the value-function target scale ≈ 1.
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=False,
            norm_reward=True,
            gamma=0.99,
            clip_reward=10.0,
        )

        # PPO — transfer learning or fresh
        policy_kwargs = {
            'features_extractor_class': StateCameraExtractor,
            'features_extractor_kwargs': {
                'features_dim': 128,
                'camera_key': 'camera_high_freq',
            },
            'net_arch': dict(pi=[128, 64], vf=[128, 64]),
        }

        transfer_path = os.path.join(
            project_root, 'models', 'goal_controller', 'best_model.zip')

        if args.load_model and os.path.exists(args.load_model):
            print(f"Resuming from {args.load_model}")
            self.model = PPO.load(args.load_model, env=self.vec_env,
                                  device='auto')
        elif not args.no_transfer and os.path.exists(transfer_path):
            print(f"Transfer learning from {transfer_path}")
            self.model = PPO.load(transfer_path, env=self.vec_env,
                                  device='auto')
            # Freeze feature extractor during Phase A — only policy/value
            # heads adapt to the new reward.  Unfreezing happens in the
            # callback at Phase B with a dedicated lr=1e-5 param group.
            extractor = self.model.policy.features_extractor
            for p in extractor.parameters():
                p.requires_grad = False
            # Reset optimiser with only non-frozen params
            trainable = [p for p in self.model.policy.parameters()
                         if p.requires_grad]
            self.model.policy.optimizer = self.model.policy.optimizer.__class__(
                trainable, lr=3e-4,
            )
        else:
            print("Training from scratch")
            self.model = PPO(
                'MultiInputPolicy', self.vec_env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=args.n_steps,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                seed=args.seed,
            )

        total_params = sum(p.numel() for p in self.model.policy.parameters())
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_recorder = None
        if args.record:
            rec_dir = self.output_dir / 'recordings'
            self.video_recorder = EpisodeRecorder(
                output_dir=str(rec_dir), fps=30, resolution=(640, 360))

        print("\n" + "=" * 70)
        print("  LEMNISCATE v2 — MULTI-COMPONENT REWARD TRAINING")
        print("=" * 70)
        print(f"  Scale:        {args.scale}m (width={2*args.scale:.0f}m)")
        print(f"  Speed:        {args.initial_speed} -> {args.max_speed} (curriculum)")
        print(f"  Phases:       A[0-{args.phase_b:.0%}] B[{args.phase_b:.0%}-{args.phase_c:.0%}] C[{args.phase_c:.0%}-100%]")
        print(f"  Timesteps:    {args.timesteps:,}")
        print(f"  n_steps:      {args.n_steps}")
        print(f"  Episode max:  {args.max_ep_steps}")
        print(f"  Params:       {total_params:,}")
        print(f"  Transfer:     {'yes' if not args.no_transfer else 'no'}")
        print(f"  Output:       {args.output_dir}")
        print("=" * 70)

    def run_training(self):
        args = self.args

        render_cb = Panda3DRenderCallback(self)
        self.metrics_cb = LemniscateV2Callback(
            env=self.env,
            output_dir=str(self.output_dir),
            initial_speed=args.initial_speed,
            max_speed=args.max_speed,
            phase_b=args.phase_b,
            phase_c=args.phase_c,
            video_recorder=self.video_recorder,
            record_interval=args.record_interval,
        )

        print("\nStarting training...\n")
        start = time.time()

        self.model.learn(
            total_timesteps=args.timesteps,
            callback=[render_cb, self.metrics_cb],
            progress_bar=True,
        )

        elapsed = time.time() - start

        # Save model + VecNormalize stats
        model_path = self.output_dir / 'best_model'
        self.model.save(str(model_path))
        vecnorm_path = self.output_dir / 'vecnormalize.pkl'
        self.vec_env.save(str(vecnorm_path))

        self.metrics_cb.save_metrics(extras={
            'total_timesteps': args.timesteps,
            'training_time_seconds': elapsed,
            'lemniscate_scale': args.scale,
            'initial_speed': args.initial_speed,
            'max_speed': args.max_speed,
            'phase_b': args.phase_b,
            'phase_c': args.phase_c,
            'n_steps': args.n_steps,
            'transfer_learning': not args.no_transfer,
            'policy_params': sum(
                p.numel() for p in self.model.policy.parameters()),
        })

        if self.video_recorder and self.video_recorder.episode_files:
            self.video_recorder.compile_timelapse(
                "training_timelapse.mp4", max_frames_per_ep=150)

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Time:      {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Episodes:  {self.metrics_cb.episode_count}")
        print(f"  Model:     {model_path}.zip")
        print("=" * 70)

        self.vec_env.close()


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 320 240')
        loadPrcFileData('', 'undecorated true')
    app = LemniscateV2App(args)
    try:
        app.run_training()
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted.")
        if hasattr(app, 'model'):
            p = Path(args.output_dir) / 'interrupted_model'
            app.model.save(str(p))
            print(f"Model saved to {p}.zip")
        if hasattr(app, 'vec_env'):
            vn = Path(args.output_dir) / 'vecnormalize.pkl'
            app.vec_env.save(str(vn))
            print(f"VecNormalize saved to {vn}")
        if hasattr(app, 'metrics_cb'):
            app.metrics_cb.save_metrics()


if __name__ == "__main__":
    main()