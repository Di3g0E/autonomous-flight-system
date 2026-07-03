#!/usr/bin/env python
"""
Test the spiral_follow_v2 PPO model in isolation (no target, no v10.4).

Loads the trained v2 spiral PPO model and runs it inside its exact
training environment (SpiralFollowEnvV2 on top of Panda3DQuadrotorEnv
with no camera and no target). Records a bird's-eye video that draws
two lines in different colors:
    * RED  — the spiral reference trajectory the drone should follow
    * CYAN — the contrail of the drone's actual XY position

A --spawn-xyz flag lets you verify the v2 model's position/altitude
invariance: spawn the drone anywhere and watch the spiral execute
relative to that point after the climb phase.

Usage:
    # Default near origin
    python scripts/test_spiral_follow_v2.py --no-display

    # Verify invariance: spawn 2 m away, 1.5 m up
    python scripts/test_spiral_follow_v2.py --no-display --spawn-xyz 2.0 -1.5 1.5
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: F401
from panda3d.core import Filename, loadPrcFile, loadPrcFileData

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from direct.showbase.ShowBase import ShowBase
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.envs.spiral_follow_env_v2 import SpiralFollowEnvV2


# Colors are RGB tuples (we write the frame with COLOR_RGB2BGR at the end).
COLOR_REF   = (255, 60, 60)    # bright red    — spiral reference
COLOR_TRAIL = (0, 255, 255)    # cyan          — drone contrail
COLOR_DRONE = (60, 255, 60)    # green         — drone marker
COLOR_TARGET_Z = (255, 200, 0)  # amber        — climb target altitude marker


def parse_args():
    p = argparse.ArgumentParser(
        description="Test spiral_follow_v2 PPO model with two-color overlay")
    p.add_argument('--model',
                   default='./models/spiral_follow_v2/best_model.zip')
    p.add_argument('--n-episodes', type=int, default=1)
    p.add_argument('--max-ep-steps', type=int, default=2000)
    p.add_argument('--omega-scale', type=float, default=1.0)
    p.add_argument('--omega', type=float, default=1.8)
    p.add_argument('--r-growth', type=float, default=0.12)
    p.add_argument('--climb-offset', type=float, default=0.8)
    p.add_argument('--climb-duration-steps', type=int, default=100)
    p.add_argument('--vision-radius', type=float, default=0.5)
    p.add_argument('--seed-base', type=int, default=2000)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--output-path', type=str, default='')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--frame-step', type=int, default=1)
    p.add_argument('--win-size', type=str, default='1920x1080')
    p.add_argument('--panel-size', type=str, default='1280x720')
    p.add_argument('--deterministic', type=int, default=1)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--use-vecnormalize', action='store_true', default=True)
    p.add_argument('--spawn-xyz', type=float, nargs=3, default=None,
                   metavar=('X', 'Y', 'Z'),
                   help="If given, force the drone to spawn at this absolute "
                        "(x, y, z). Verifies position/altitude invariance.")
    p.add_argument('--view-extent', type=float, default=4.0,
                   help="World half-width visible on each bird's-eye axis (m).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────

class SpiralV2TestApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)
        self.taskMgr.remove('Camera Movement')

        # Bird's-eye external camera. If we spawn far from origin, follow
        # the drone with the camera so the spiral stays in frame.
        cam_x, cam_y, cam_z = 0.0, -2.0, 14.0
        look_x, look_y, look_z = 0.0, 0.0, 1.4
        if args.spawn_xyz is not None:
            cam_x, cam_y = float(args.spawn_xyz[0]), float(args.spawn_xyz[1]) - 2.0
            look_x, look_y = float(args.spawn_xyz[0]), float(args.spawn_xyz[1])
            look_z = float(args.spawn_xyz[2])

        self.cam.reparentTo(self.render)
        self.cam.setPos(cam_x, cam_y, cam_z)
        self.cam.lookAt(look_x, look_y, look_z)

        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.cam.setPos(cam_x, cam_y, cam_z)
        self.ext_camera.cam.lookAt(look_x, look_y, look_z)
        self.ext_camera.buffer.setActive(1)

        # The orthographic centre used by the world->panel projection.
        # When spawning far from origin we shift it to the spawn XY so the
        # spiral stays centred on the panel.
        self.panel_center_x = float(args.spawn_xyz[0]) if args.spawn_xyz else 0.0
        self.panel_center_y = float(args.spawn_xyz[1]) if args.spawn_xyz else 0.0

        print(f"Creating env (no camera, no target, max_steps={args.max_ep_steps})...")
        self.base_env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=False,
            use_depth=False,
            use_target=False,
            enable_collisions=False,
            n=args.max_ep_steps,
            t_step=0.01,
            direct_control=1,
            filming_mode=True,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.1,
            init_vel_range=0.05,
            init_ang_range=0.03,
        )

        self.spiral_env = SpiralFollowEnvV2(
            self.base_env,
            omega=args.omega,
            r_growth=args.r_growth,
            climb_offset=args.climb_offset,
            climb_duration_steps=args.climb_duration_steps,
            vision_radius=args.vision_radius,
        )
        self.spiral_env.omega_scale = float(args.omega_scale)

        if args.use_vecnormalize:
            mon = Monitor(self.spiral_env)
            self.vec_env = DummyVecEnv([lambda: mon])
            vp = Path(args.model).parent / 'vecnormalize.pkl'
            if vp.exists():
                self.vec_env = VecNormalize.load(str(vp), self.vec_env)
                self.vec_env.training = False
                self.vec_env.norm_reward = False
                print(f"Loaded VecNormalize stats: {vp}")
            else:
                self.vec_env = VecNormalize(
                    self.vec_env, norm_obs=False,
                    norm_reward=False, gamma=0.99)
                print("WARNING: vecnormalize.pkl not found, using fresh VecNormalize")
            self.model = PPO.load(args.model, env=self.vec_env, device='auto')
        else:
            self.vec_env = None
            self.model = PPO.load(args.model, device='auto')
        print(f"Loaded spiral v2 PPO model: {args.model}")
        print(f"omega_scale: {args.omega_scale}  "
              f"climb_offset: {args.climb_offset}  "
              f"climb_duration_steps: {args.climb_duration_steps}")


# ──────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────

def world_to_panel(x_w, y_w, panel_w, panel_h, view_extent, cx_world, cy_world):
    """Project a world XY onto the bird's-eye panel using an orthographic
    projection centred at (cx_world, cy_world) with the given half-extent."""
    cx, cy = panel_w // 2, panel_h // 2
    px = int(cx + (x_w - cx_world) / view_extent * cx)
    py = int(cy - (y_w - cy_world) / view_extent * cy)
    return px, py


def draw_polyline(panel, world_pts, color, panel_w, panel_h, view_extent,
                  cx_world, cy_world, thickness=2):
    if len(world_pts) < 2:
        return
    pts = np.array([
        world_to_panel(p[0], p[1], panel_w, panel_h, view_extent,
                       cx_world, cy_world)
        for p in world_pts
    ], dtype=np.int32)
    cv2.polylines(panel, [pts], isClosed=False, color=color, thickness=thickness)


def draw_overlay(panel, info, panel_w, panel_h,
                 drone_pos, ref_pos, drone_history, ref_history,
                 view_extent, cx_world, cy_world,
                 climb_target_z):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(panel, "Bird's-eye + spiral v2 reference", (10, 28),
                font, 0.65, (255, 255, 255), 2)

    # Two coloured trajectories: reference (red) and drone trail (cyan).
    draw_polyline(panel, ref_history, COLOR_REF, panel_w, panel_h,
                  view_extent, cx_world, cy_world, thickness=2)
    draw_polyline(panel, drone_history, COLOR_TRAIL, panel_w, panel_h,
                  view_extent, cx_world, cy_world, thickness=2)

    # Current drone marker (green)
    dpx, dpy = world_to_panel(drone_pos[0], drone_pos[1],
                              panel_w, panel_h, view_extent,
                              cx_world, cy_world)
    cv2.circle(panel, (dpx, dpy), 8, COLOR_DRONE, -1)
    cv2.circle(panel, (dpx, dpy), 8, (255, 255, 255), 1)
    cv2.putText(panel, "drone", (dpx + 12, dpy - 6),
                font, 0.45, COLOR_DRONE, 1)

    # Current reference marker (red)
    rpx, rpy = world_to_panel(ref_pos[0], ref_pos[1],
                              panel_w, panel_h, view_extent,
                              cx_world, cy_world)
    cv2.circle(panel, (rpx, rpy), 7, COLOR_REF, -1)
    cv2.circle(panel, (rpx, rpy), 7, (255, 255, 255), 1)
    cv2.putText(panel, "ref", (rpx + 12, rpy - 6),
                font, 0.45, COLOR_REF, 1)

    # Drone-to-ref segment (faint grey) to visualise position error
    cv2.line(panel, (dpx, dpy), (rpx, rpy), (200, 200, 200), 1)

    # Legend (top-left)
    leg_y = 60
    rows_leg = [
        ('Reference (target)', COLOR_REF),
        ('Drone contrail',     COLOR_TRAIL),
        ('Drone (now)',        COLOR_DRONE),
    ]
    for label, color in rows_leg:
        cv2.rectangle(panel, (10, leg_y - 8), (28, leg_y + 8),
                      color, -1)
        cv2.putText(panel, label, (36, leg_y + 6),
                    font, 0.5, (255, 255, 255), 1)
        leg_y += 22

    # Telemetry (top-right)
    y = 60
    in_climb = info.get('InClimb', False)
    rows = [
        ('Phase',    'CLIMB' if in_climb else 'SPIRAL'),
        ('Step',     f"{info.get('Step', 0):>4}"),
        ('Time',     f"{info.get('Step', 0) * 0.01:>5.2f} s"),
        ('Drone XY', f"({drone_pos[0]:+.2f}, {drone_pos[1]:+.2f})"),
        ('Drone Z',  f"{drone_pos[2]:.3f} m"),
        ('Target Z', f"{climb_target_z:.3f} m"),
        ('Alt err',  f"{abs(drone_pos[2] - climb_target_z):.3f} m"),
        ('Ref r',    f"{info.get('SpiralR', 0):.3f} m"),
        ('Pos err',  f"{info.get('PosErr', 0):.3f} m"),
    ]
    x_text = panel_w - 320
    cv2.rectangle(panel, (x_text - 10, y - 30),
                  (panel_w - 5, y + len(rows) * 26),
                  (0, 0, 0), -1)
    cv2.putText(panel, "-- Telemetry --", (x_text, y - 8),
                font, 0.5, (255, 255, 255), 1)
    for k, v in rows:
        cv2.putText(panel, f"{k:>9}: {v}", (x_text, y), font, 0.5,
                    COLOR_DRONE, 1)
        y += 24


def get_bird_image(ext_camera, panel_w, panel_h):
    ok, rgba = ext_camera.get_image()
    if not ok:
        return np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    return cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)


# ──────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────

def force_spawn(app, xyz):
    """Override the drone's state immediately after reset() so it starts
    at an arbitrary absolute position."""
    s = app.base_env.base_env.state.copy()
    s[0] = float(xyz[0])
    s[2] = float(xyz[1])
    s[4] = float(xyz[2])
    s[1] = s[3] = s[5] = 0.0
    s[6] = 1.0
    s[7:10] = 0.0
    s[10:13] = 0.0
    app.base_env.base_env.state = s.copy()
    app.base_env.base_env.previous_state = s.copy()
    app.base_env._update_visualization()
    # Re-anchor the spiral reference and altitude target at the new spawn.
    app.spiral_env._center_x = float(xyz[0])
    app.spiral_env._center_y = float(xyz[1])
    app.spiral_env._z0 = float(xyz[2])
    app.spiral_env._target_z = float(xyz[2]) + app.spiral_env.climb_offset
    app.spiral_env._ref_x = float(xyz[0])
    app.spiral_env._ref_y = float(xyz[1])


def run_episode(app, args, seed, output_path, panel_w, panel_h):
    if app.vec_env is not None:
        obs = app.vec_env.reset()
    else:
        obs, _ = app.spiral_env.reset(seed=seed)

    if args.spawn_xyz is not None:
        force_spawn(app, args.spawn_xyz)

    app.taskMgr.step()

    # Anchor for the climb target altitude (z0 + climb_offset).
    z0 = float(app.spiral_env._z0)
    x0 = float(app.spiral_env._x0)
    y0 = float(app.spiral_env._y0)
    climb_target_z = z0 + args.climb_offset

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps,
                              (panel_w, panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter at {output_path}")

    drone_history = []
    ref_history = []
    pos_errors = []
    alt_errors = []
    max_r = 0.0
    cum_reward = 0.0
    step = 0
    deterministic = bool(args.deterministic)
    last_term_reason = ''

    while step < args.max_ep_steps:
        act, _ = app.model.predict(obs, deterministic=deterministic)
        if app.vec_env is not None:
            obs, rewards, dones, infos = app.vec_env.step(act)
            r = float(rewards[0])
            done = bool(dones[0])
            info = infos[0] if infos else {}
            term = info.get('TimeLimit.truncated') is False and done
            trunc = info.get('TimeLimit.truncated', False)
        else:
            single_act = act[0] if act.ndim > 1 else act
            obs, r, term, trunc, info = app.spiral_env.step(single_act)
            done = term or trunc
        app.taskMgr.step()
        cum_reward += float(r)
        step += 1

        st = app.base_env.base_env.state
        drone_pos = (float(st[0]), float(st[2]), float(st[4]))
        # Absolute-frame reference for plotting (rel + spawn anchor).
        ref_abs_x = x0 + float(app.spiral_env._ref_x_rel)
        ref_abs_y = y0 + float(app.spiral_env._ref_y_rel)
        ref_pos = (ref_abs_x, ref_abs_y)
        spiral_r = float(app.spiral_env._ref_r)
        pos_err = math.sqrt((drone_pos[0] - ref_pos[0]) ** 2 +
                            (drone_pos[1] - ref_pos[1]) ** 2)
        alt_err = abs(drone_pos[2] - climb_target_z)
        in_climb = app.spiral_env.spiral_step < app.spiral_env.climb_duration_steps

        pos_errors.append(pos_err)
        alt_errors.append(alt_err)
        drone_history.append((drone_pos[0], drone_pos[1]))
        # Only start tracing the reference once the spiral phase has begun;
        # during the climb the reference is a single point at (x0, y0).
        if in_climb:
            if not ref_history:
                ref_history.append((x0, y0))
        else:
            ref_history.append((ref_pos[0], ref_pos[1]))
        max_r = max(max_r, spiral_r)

        # Cap stored history length to bound memory.
        if len(drone_history) > 4000:
            drone_history = drone_history[-4000:]
        if len(ref_history) > 4000:
            ref_history = ref_history[-4000:]

        if args.verbose:
            act_disp = act[0] if hasattr(act, 'shape') and act.ndim > 1 else act
            phase = 'CLIMB' if in_climb else 'SPIRAL'
            print(f"  step={step:>4} [{phase}]  "
                  f"drone=({drone_pos[0]:+.2f},{drone_pos[1]:+.2f},{drone_pos[2]:.2f})  "
                  f"ref=({ref_pos[0]:+.2f},{ref_pos[1]:+.2f})  r={spiral_r:.2f}  "
                  f"pos_err={pos_err:.3f}  alt_err={alt_err:.3f}  "
                  f"act=[{act_disp[0]:+.2f},{act_disp[1]:+.2f},"
                  f"{act_disp[2]:+.2f},{act_disp[3]:+.2f}]  r={r:+.3f}")

        if step % args.frame_step == 0:
            bird = get_bird_image(app.ext_camera, panel_w, panel_h)
            info_overlay = {
                'Step': step,
                'SpiralR': spiral_r,
                'PosErr': pos_err,
                'InClimb': in_climb,
            }
            draw_overlay(bird, info_overlay, panel_w, panel_h,
                         drone_pos, ref_pos,
                         drone_history, ref_history,
                         view_extent=args.view_extent,
                         cx_world=app.panel_center_x,
                         cy_world=app.panel_center_y,
                         climb_target_z=climb_target_z)
            writer.write(cv2.cvtColor(bird, cv2.COLOR_RGB2BGR))

        if app.vec_env is not None:
            if done:
                last_term_reason = (f"VecEnv done at step {step} "
                                    f"(term={term}, trunc={trunc})")
                break
        else:
            if term or trunc:
                last_term_reason = f"term={term} trunc={trunc} at step {step}"
                break

    writer.release()
    return {
        'seed': seed,
        'steps': step,
        'mean_pos_error': float(np.mean(pos_errors)) if pos_errors else 0.0,
        'p95_pos_error': float(np.percentile(pos_errors, 95)) if pos_errors else 0.0,
        'mean_alt_error': float(np.mean(alt_errors)) if alt_errors else 0.0,
        'max_r_reached': max_r,
        'cum_reward': cum_reward,
        'video_path': str(output_path),
        'term_reason': last_term_reason,
        'climb_target_z': climb_target_z,
    }


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    win_w, win_h = (int(x) for x in args.win_size.lower().replace(' ', 'x').split('x'))
    panel_w, panel_h = (int(x) for x in args.panel_size.lower().split('x'))

    if args.no_display:
        loadPrcFileData('', 'window-type offscreen')
        loadPrcFileData('', f'win-size {win_w} {win_h}')
        loadPrcFileData('', 'undecorated true')
    else:
        loadPrcFileData('', f'win-size {win_w} {win_h}')

    app = SpiralV2TestApp(args)

    outdir = Path('./models/spiral_follow_v2/recordings')
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  SPIRAL FOLLOW V2 STANDALONE TEST  (omega_scale={args.omega_scale})")
    print("=" * 70)

    results = []
    for i in range(args.n_episodes):
        seed = args.seed_base + i
        if args.output_path and args.n_episodes == 1:
            video_path = Path(args.output_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            scale_tag = f"{args.omega_scale:.2f}".replace('.', '')
            xyz_tag = ''
            if args.spawn_xyz is not None:
                xyz_tag = ("_xyz_"
                           + "_".join(f"{v:+.1f}".replace('.', '').replace('+', 'p').replace('-', 'm')
                                      for v in args.spawn_xyz))
            video_path = outdir / (
                f"spiral_v2_seed{seed}_omega{scale_tag}{xyz_tag}.mp4")
        print(f"\n--- Episode {i + 1}/{args.n_episodes} (seed={seed}) ---")
        t0 = time.time()
        result = run_episode(app, args, seed, video_path, panel_w, panel_h)
        elapsed = time.time() - t0
        result['wall_clock_s'] = elapsed
        results.append(result)
        print(f"  Steps:           {result['steps']}/{args.max_ep_steps}")
        print(f"  Climb target Z:  {result['climb_target_z']:.3f} m")
        print(f"  Mean pos error:  {result['mean_pos_error']:.3f} m")
        print(f"  P95 pos error:   {result['p95_pos_error']:.3f} m")
        print(f"  Mean alt error:  {result['mean_alt_error']:.3f} m")
        print(f"  Max r reached:   {result['max_r_reached']:.3f} m")
        print(f"  Cum reward:      {result['cum_reward']:.1f}")
        if result['term_reason']:
            print(f"  Termination:     {result['term_reason']}")
        print(f"  Video:           {result['video_path']}")

    print("\n" + "=" * 70)
    print(f"  AGGREGATE  ({args.n_episodes} episodes)")
    print("=" * 70)
    if args.n_episodes >= 1:
        for k in ('mean_pos_error', 'p95_pos_error', 'mean_alt_error',
                  'max_r_reached', 'cum_reward'):
            vals = [r[k] for r in results]
            print(f"  {k:>20}: mean={np.mean(vals):.3f}  "
                  f"std={np.std(vals):.3f}")


if __name__ == "__main__":
    main()
