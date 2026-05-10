#!/usr/bin/env python
"""
Test the v9.1 best model.

Loads:
  - models/hover_track_v9_1/best_model.zip      (SAC policy)
  - models/hover_track_v9_1/best_vec_normalize.pkl  (obs running stats)

Runs N deterministic eval episodes with seeds the training never saw,
records one video of a full 30-s flight, and prints final metrics with
a corrected survival check (independent of the off-by-one in the
training callback).

Usage:
    python scripts/test_hover_track_v9_1.py
    python scripts/test_hover_track_v9_1.py --n-episodes 20 --no-display
"""

import argparse
import json
import os
import sys
import time
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.utils.episode_recorder import EpisodeRecorder

# Reuse the wrapper from the training script so the MDP is identical
from scripts.train_hover_track_v9 import HoverTrackV9Wrapper


def parse_args():
    p = argparse.ArgumentParser(description="Test v9.1 best model")
    p.add_argument('--model-dir', type=str,
                   default='./models/hover_track_v9_1')
    p.add_argument('--model-name', type=str, default='best_model',
                   help="Filename (without .zip) inside model-dir.")
    p.add_argument('--vec-norm-name', type=str, default='best_vec_normalize',
                   help="Filename (without .pkl) of the VecNormalize stats.")
    p.add_argument('--n-episodes', type=int, default=10)
    p.add_argument('--max-ep-steps', type=int, default=3000)
    p.add_argument('--seed-base', type=int, default=2000,
                   help="Seeds: seed_base..seed_base+n_episodes-1. "
                        "Default 2000 differs from training (42) and from "
                        "the eval callback (1000-1004).")
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--video-fps', type=int, default=10)
    p.add_argument('--output-name', type=str, default='test_results')
    return p.parse_args()


class TestApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        self.model_dir = Path(args.model_dir)

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

        print(f"Creating env (max_ep_steps={args.max_ep_steps})...")
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
            spawn_height=1.5, jitter_xy=0.20,
            w_alive=0.10, w_jerk=0.20, w_invisible=1.0,
            z_min=0.5, z_max=3.0,
            invisible_term_steps=100, crash_penalty=2.0,
        )
        self.env = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # Load VecNormalize stats (required: model was trained on
        # normalized obs).
        vec_norm_path = self.model_dir / f"{args.vec_norm_name}.pkl"
        if not vec_norm_path.exists():
            raise FileNotFoundError(
                f"VecNormalize stats not found at {vec_norm_path}. "
                f"Cannot run test — the policy expects normalized obs.")
        self.vec_env = VecNormalize.load(str(vec_norm_path), self.vec_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        print(f"Loaded VecNormalize stats from {vec_norm_path.name}")

        # Load model
        model_path = self.model_dir / f"{args.model_name}.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = SAC.load(str(model_path), env=self.vec_env)
        print(f"Loaded SAC model from {model_path.name}")

        # Video recorder (one episode = one mp4)
        self.ep_recorder = None
        if not args.no_video:
            self.ep_recorder = EpisodeRecorder(
                output_dir=str(self.model_dir / 'test_recordings'),
                fps=args.video_fps, resolution=(640, 480),
            )
        self.frame_step = max(1, 100 // args.video_fps)

    def run(self):
        args = self.args
        max_steps = args.max_ep_steps
        results = []

        for i in range(args.n_episodes):
            seed = args.seed_base + i
            print(f"\n--- Episode {i+1}/{args.n_episodes} (seed={seed}) ---")
            obs, _ = self.raw_env.reset(seed=seed)
            self.taskMgr.step()

            if self.ep_recorder is not None:
                self.ep_recorder.start_episode(i + 1)

            step = 0
            visible = 0
            ep_reward = 0.0
            prev_act = None
            jerk_sum = 0.0
            jerk_n = 0
            term_reason = ''
            done = False

            while not done and step < max_steps:
                # Normalize obs (VecNormalize expects (n_envs, obs_dim)
                # and returns same shape; we pass single-env obs).
                obs_norm = self.vec_env.normalize_obs(
                    np.asarray(obs, dtype=np.float32))
                act, _ = self.model.predict(obs_norm, deterministic=True)

                obs, reward, term, trunc, info = self.raw_env.step(act)
                self.taskMgr.step()

                vt = info.get('visual_tracking', {}) or {}
                if vt.get('target_visible', False):
                    visible += 1
                if prev_act is not None:
                    jerk_sum += float(np.mean(np.abs(act - prev_act)))
                    jerk_n += 1
                prev_act = act
                ep_reward += float(reward)

                if (self.ep_recorder is not None
                        and step % self.frame_step == 0):
                    fpv_img = self.raw_env._last_high_freq_image
                    bird_img = None
                    ok, rgba = self.ext_camera.get_image()
                    if ok:
                        bird_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
                    overlay = {
                        'visual_tracking': vt,
                        'target': info.get('target', {}),
                        'Episode': i + 1,
                        'Step': step,
                        'Reward': round(float(reward), 2),
                    }
                    self.ep_recorder.capture_frame(fpv_img, bird_img, overlay)

                if info.get('v9_term_reason'):
                    term_reason = info['v9_term_reason']

                step += 1
                done = term or trunc

            if self.ep_recorder is not None:
                self.ep_recorder.end_episode()

            survived = int(step >= max_steps)
            vis_pct = visible / max(step, 1)
            jerk = jerk_sum / max(jerk_n, 1)
            res = {
                'episode': i + 1,
                'seed': seed,
                'steps': step,
                'survived': survived,
                'visibility': vis_pct,
                'mean_jerk': jerk,
                'reward': ep_reward,
                'term_reason': term_reason,
            }
            results.append(res)
            print(f"  steps={step:>4}  survived={survived}  "
                  f"vis={vis_pct:.2%}  jerk={jerk:.4f}  "
                  f"reward={ep_reward:.1f}  term={term_reason or '(none)'}")

        # ── Aggregate ──
        steps_arr = np.array([r['steps'] for r in results])
        surv_arr = np.array([r['survived'] for r in results])
        vis_arr = np.array([r['visibility'] for r in results])
        jerk_arr = np.array([r['mean_jerk'] for r in results])
        rew_arr = np.array([r['reward'] for r in results])

        summary = {
            'n_episodes': args.n_episodes,
            'survival_rate': float(surv_arr.mean()),
            'mean_steps': float(steps_arr.mean()),
            'std_steps': float(steps_arr.std()),
            'min_steps': int(steps_arr.min()),
            'max_steps': int(steps_arr.max()),
            'mean_visibility': float(vis_arr.mean()),
            'mean_jerk': float(jerk_arr.mean()),
            'mean_reward': float(rew_arr.mean()),
            'std_reward': float(rew_arr.std()),
            'episodes': results,
        }

        out_path = self.model_dir / f"{args.output_name}.json"
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        print(f"  Episodes:           {args.n_episodes}")
        print(f"  Survival rate:      {summary['survival_rate']:.2%}  "
              f"({int(surv_arr.sum())}/{args.n_episodes} eps reached "
              f"{max_steps} steps)")
        print(f"  Mean steps:         {summary['mean_steps']:.1f} ± "
              f"{summary['std_steps']:.1f}")
        print(f"  Range steps:        [{summary['min_steps']}, "
              f"{summary['max_steps']}]")
        print(f"  Mean visibility:    {summary['mean_visibility']:.2%}")
        print(f"  Mean jerk:          {summary['mean_jerk']:.4f}")
        print(f"  Mean reward/ep:     {summary['mean_reward']:.1f} ± "
              f"{summary['std_reward']:.1f}")
        print(f"  Output:             {out_path}")
        if self.ep_recorder is not None:
            print(f"  Videos:             "
                  f"{self.model_dir / 'test_recordings'}")
        print("=" * 70)

        self.vec_env.close()


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    app = TestApp(args)
    app.run()


if __name__ == "__main__":
    main()
