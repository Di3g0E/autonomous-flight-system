#!/usr/bin/env python
"""
Test offline para modelos v10. A diferencia de test_hover_track_v9_1.py
(que fuerza target_speed=0), este script admite --target-speed FLOAT y
lo aplica al wrapper en cada reset, permitiendo medir robustez a
distintas velocidades del target en lemniscata.

Carga el best_model + best_vec_normalize, ejecuta N episodios con seeds
nunca vistos por el callback (default 2000-2009), graba opcionalmente
vídeos y guarda métricas en JSON.

Usage:
    # Validar v10.4 best a target_speed=0.05 (régimen del best step)
    python scripts/test_hover_track_v10.py --no-display --model-dir ./models/hover_track_v10_4 --target-speed 0.05

    # Estrés a 0.10 m/s con 10 seeds nuevos
    python scripts/test_hover_track_v10.py --no-display --model-dir ./models/hover_track_v10_4 --target-speed 0.10 --output-name test_at_010
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: F401  must precede SB3 to force miniconda torch
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

# v10 wrapper supports use_curriculum_speed → reset uses self._curriculum_target_speed
from scripts.train_hover_track_v10 import HoverTrackV10Wrapper


def parse_args():
    p = argparse.ArgumentParser(description="v10 offline test with arbitrary target_speed")
    p.add_argument('--model-dir', type=str, required=True)
    p.add_argument('--model-name', type=str, default='best_model')
    p.add_argument('--vec-norm-name', type=str, default='best_vec_normalize')
    p.add_argument('--target-speed', type=float, default=0.0,
                   help="Target speed in m/s. Overrides any default in the wrapper.")
    p.add_argument('--n-episodes', type=int, default=10)
    p.add_argument('--max-ep-steps', type=int, default=3000)
    p.add_argument('--seed-base', type=int, default=2000)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--output-name', type=str, default='')
    return p.parse_args()


class TestApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        self.model_dir = Path(args.model_dir)
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

        # Build wrapper with use_curriculum_speed=True so reset() honors
        # self._curriculum_target_speed (set just below to args.target_speed).
        self.raw_env = HoverTrackV10Wrapper(
            panda3d_app=self, quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True, use_depth=False,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            use_target=True, target_mode='moving',
            target_speed=args.target_speed,  # base default; reset() will set _curriculum_target_speed
            target_radius=0.25,
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
            use_curriculum_speed=True,  # honor _curriculum_target_speed in reset()
        )
        # The curriculum value reset() will read every episode:
        self.raw_env._curriculum_target_speed = float(args.target_speed)

        # Load model + VecNormalize stats
        mon = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: mon])
        vp = self.model_dir / f"{args.vec_norm_name}.pkl"
        if not vp.exists():
            raise FileNotFoundError(f"VecNormalize not found: {vp}")
        self.vec_env = VecNormalize.load(str(vp), self.vec_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        self.vec_normalize = self.vec_env

        mp = self.model_dir / f"{args.model_name}.zip"
        if not mp.exists():
            raise FileNotFoundError(f"Model not found: {mp}")
        self.model = SAC.load(str(mp), env=self.vec_env, device='auto')
        print(f"Loaded model: {mp}")
        print(f"Loaded VecNormalize: {vp}")
        print(f"Target speed: {args.target_speed} m/s")

    def run_episodes(self):
        args = self.args
        results = []
        for i in range(args.n_episodes):
            seed = args.seed_base + i
            obs, _ = self.raw_env.reset(seed=seed)
            self.taskMgr.step()
            done = False
            step = 0
            visible = 0
            jerk_sum = 0.0
            jerk_n = 0
            prev_act = None
            ep_reward = 0.0
            term_reason = ''
            while not done and step < args.max_ep_steps:
                obs_in = self.vec_normalize.normalize_obs(
                    np.asarray(obs, dtype=np.float32))
                act, _ = self.model.predict(obs_in, deterministic=True)
                obs, r, term, trunc, info = self.raw_env.step(act)
                self.taskMgr.step()
                ep_reward += float(r)
                vt = info.get('visual_tracking', {}) or {}
                if vt.get('target_visible', False):
                    visible += 1
                if prev_act is not None:
                    jerk_sum += float(np.mean(np.abs(act - prev_act)))
                    jerk_n += 1
                prev_act = act
                step += 1
                term_reason = info.get('v9_term_reason', '') or term_reason
                done = term or trunc
            survived = int(step >= args.max_ep_steps)
            jerk = jerk_sum / max(jerk_n, 1)
            vis = visible / max(step, 1)
            print(f"--- Episode {i+1}/{args.n_episodes} (seed={seed}) ---")
            print(f"  steps={step:>5}  survived={survived}  vis={vis*100:.2f}%  "
                  f"jerk={jerk:.4f}  reward={ep_reward:.1f}  term={term_reason or '(none)'}")
            results.append({
                'episode': i + 1, 'seed': seed, 'steps': step, 'survived': survived,
                'visibility': vis, 'mean_jerk': jerk, 'reward': ep_reward,
                'term_reason': term_reason,
            })
        return results


def main():
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'window-type offscreen')
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')
    print("=" * 70)
    print(f"  TEST V10 — target_speed={args.target_speed} m/s")
    print("=" * 70)
    app = TestApp(args)
    t0 = time.time()
    results = app.run_episodes()
    elapsed = time.time() - t0
    steps = [r['steps'] for r in results]
    survs = [r['survived'] for r in results]
    viss = [r['visibility'] for r in results]
    jerks = [r['mean_jerk'] for r in results]
    rewards = [r['reward'] for r in results]
    summary = {
        'target_speed': args.target_speed,
        'n_episodes': len(results),
        'survival_rate': float(np.mean(survs)),
        'mean_steps': float(np.mean(steps)),
        'std_steps': float(np.std(steps)),
        'min_steps': int(np.min(steps)),
        'max_steps': int(np.max(steps)),
        'mean_visibility': float(np.mean(viss)),
        'mean_jerk': float(np.mean(jerks)),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'episodes': results,
    }
    out_name = args.output_name or f"test_results_speed_{args.target_speed:.2f}".replace('.', '_')
    out_path = Path(args.model_dir) / f"{out_name}.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print("=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    print(f"  Target speed:     {args.target_speed} m/s")
    print(f"  Episodes:         {summary['n_episodes']}")
    print(f"  Survival rate:    {summary['survival_rate']*100:.2f}%  "
          f"({sum(survs)}/{len(survs)} eps reached {args.max_ep_steps} steps)")
    print(f"  Mean steps:       {summary['mean_steps']:.1f} ± {summary['std_steps']:.1f}")
    print(f"  Range steps:      [{summary['min_steps']}, {summary['max_steps']}]")
    print(f"  Mean visibility:  {summary['mean_visibility']*100:.2f}%")
    print(f"  Mean jerk:        {summary['mean_jerk']:.4f}")
    print(f"  Mean reward/ep:   {summary['mean_reward']:.1f} ± {summary['std_reward']:.1f}")
    print(f"  Time:             {elapsed:.0f}s")
    print(f"  Output:           {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
