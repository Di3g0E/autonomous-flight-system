#!/usr/bin/env python
"""
Record demo videos of the v10.4 final TFG model.

Two modes:
  --mode tracking
    The drone spawns near the target and tracks it for ``--max-steps``.
    Demonstrates the v10.4 best model in the regime it was trained for
    (target_speed = 0.0..0.10 m/s).

  --mode full-flight
    The drone spawns far from the target. Uses ``SpiralSearchController``
    which drives an Archimedes spiral until the target enters its FOV;
    then transitions through HANDOFF to the v10.4 tracking policy.

Each video is recorded with two side-by-side panels:
    [ FPV (drone view) | Bird's-eye (external) ]
overlaid with step number, simulated time, reward, target visibility
and (full-flight only) the controller state TRACK / SEARCH / HANDOFF.

Default real-time playback (~100 Hz sim → 20 Hz video × frame_step=5).

Usage:
    # Tracking demo at target_speed=0.10 m/s, 30 s
    python scripts/record_v10_4_demo.py --no-display --mode tracking --target-speed 0.10

    # Full-flight demo (spiral search → tracking) at target_speed=0.05
    python scripts/record_v10_4_demo.py --no-display --mode full-flight --target-speed 0.05 --max-steps 5000
"""

import argparse
import math
import os
import sys
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

import torch  # noqa: F401  must precede SB3 to use miniconda torch
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
from src.agents.spiral_search_controller import SpiralSearchController
from src.agents.spiral_search_controller_v2 import SpiralSearchControllerV2
from scripts.train_hover_track_v10 import HoverTrackV10Wrapper


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Record v10.4 demo videos")
    p.add_argument('--mode', required=True, choices=['tracking', 'full-flight'])
    p.add_argument('--model', default='./models/hover_track_v10_4/best_model_TFG.zip')
    p.add_argument('--vec-norm',
                   default='./models/hover_track_v10_4/best_vec_normalize_TFG.pkl')
    p.add_argument('--spiral-model',
                   default='./models/spiral_follow/best_model.zip')
    p.add_argument('--use-v2', action='store_true',
                   help="Use SpiralSearchControllerV2 (relative-frame, "
                        "climb-then-spiral). Default --spiral-model is v1; "
                        "with --use-v2 you usually want "
                        "--spiral-model ./models/spiral_follow_v2/best_model.zip")
    p.add_argument('--climb-offset', type=float, default=0.8,
                   help="v2 only: metres to climb before spiral starts.")
    p.add_argument('--climb-duration-steps', type=int, default=100,
                   help="v2 only: sim steps reserved for the climb phase "
                        "(100 = 1.0 s).")
    p.add_argument('--target-speed', type=float, default=0.10,
                   help="Lemniscate speed in m/s. 0.10 demonstrates the "
                        "best model's operational regime.")
    p.add_argument('--seed', type=int, default=2007,
                   help="Seed 2007 reached 3000 steps perfectly in offline test.")
    p.add_argument('--max-steps', type=int, default=0,
                   help="0 = auto (3000 tracking, 6000 full-flight)")
    p.add_argument('--spawn-offset', type=float, default=1.0,
                   help="full-flight only: XY distance from target at spawn (m). "
                        "1.0 = target found in ~2 s; >1.5 risks BB violation.")
    p.add_argument('--scenario', choices=['target-below', 'target-outside'],
                   default=None,
                   help="full-flight only. Overrides spawn positions:\n"
                        "  target-below   : target directly under the drone "
                        "(same vertical axis) → tracking activates immediately.\n"
                        "  target-outside : target far lateral from the drone, "
                        "outside the downward camera's FOV → search/spiral "
                        "activates, then HANDOFF when found.")
    p.add_argument('--hover-z', type=float, default=1.5,
                   help="Drone spawn altitude for --scenario (m). v2 spiral "
                        "handles any altitude; 1.5 keeps the camera coverage "
                        "reasonable.")
    p.add_argument('--scenario-offset', type=float, default=2.5,
                   help="--scenario target-outside: lateral XY distance from "
                        "target where the drone spawns (m). 2.5 m is well "
                        "outside the downward camera FOV at hover_z=1.5.")
    p.add_argument('--central-fov-radius', type=float, default=0.0,
                   help="Stricter visibility for SEARCH→TRACK transition: "
                        "only treat target as visible if its centroid is "
                        "within a circle of this normalised radius (image "
                        "coords in [-1, 1]). 0.0 disables the filter "
                        "(default = raw visibility). Try 0.30–0.50.")
    p.add_argument('--central-fov-frames', type=int, default=3,
                   help="Consecutive frames the centroid must be inside the "
                        "central radius before TRACK activates. Avoids flicker.")
    p.add_argument('--central-fov-timeout', type=int, default=0,
                   help="Fallback: if the target is visible but never centred "
                        "for this many consecutive steps, force the transition "
                        "anyway. 0 disables the fallback (default).")
    p.add_argument('--spiral-restart-cooldown', type=int, default=200,
                   help="Min steps between spiral restarts. The spiral "
                        "restarts at the drone's current XYZ when the target "
                        "becomes raw-visible during SEARCH but the central "
                        "filter blocks TRACK — so the drone re-spirals around "
                        "its current position instead of drifting further on "
                        "the previous arc. 0 disables restarts.")
    p.add_argument('--invisible-threshold', type=int, default=5,
                   help="full-flight: steps without target before SEARCH activates")
    p.add_argument('--handoff-steps', type=int, default=50,
                   help="full-flight: linear blending steps from spiral→RL when "
                        "target found. Longer = smoother (default 50, was 15).")
    p.add_argument('--no-velocity-damp', action='store_true',
                   help="full-flight: disable velocity damping at SEARCH→HANDOFF "
                        "transition (NOT recommended, exists for ablation only).")
    p.add_argument('--output-path', type=str, default='')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--frame-step', type=int, default=5,
                   help="capture every Nth sim step. fps*frame_step≈100 → real-time")
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--panel-size', type=str, default='1280x720',
                   help="Per-panel size for the output video. Total frame "
                        "is 2× wide. Default 1280x720 → 2560x720 video (HD).")
    p.add_argument('--win-size', type=str, default='1920x1080',
                   help="Panda3D offscreen window/camera buffer size. Default "
                        "1920x1080 → opencv_camera returns 960x540 source.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────

STATE_COLORS = {
    'track':   (50, 255, 50),    # green
    'search':  (255, 165, 0),    # orange
    'handoff': (255, 255, 0),    # yellow
}


def draw_fpv(panel, info, panel_w, panel_h, central_fov_radius=0.0,
             centred_now=False):
    """Add labels, crosshair and target marker on top of FPV image."""
    cv2.putText(panel, "FPV (drone view)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # Crosshair
    cx, cy = panel_w // 2, panel_h // 2
    cv2.line(panel, (cx - 12, cy), (cx + 12, cy), (0, 255, 0), 1)
    cv2.line(panel, (cx, cy - 12), (cx, cy + 12), (0, 255, 0), 1)
    # Central-FOV gating circle (ellipse to honour the wide panel).
    if central_fov_radius > 0.0:
        ax = int(central_fov_radius * panel_w / 2)
        ay = int(central_fov_radius * panel_h / 2)
        color = (60, 255, 60) if centred_now else (255, 200, 0)
        cv2.ellipse(panel, (cx, cy), (ax, ay), 0, 0, 360, color, 2)
        cv2.putText(panel, f"central r={central_fov_radius:.2f}",
                    (cx - ax, cy - ay - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    # Target dot
    vt = info.get('visual_tracking', {})
    if vt.get('target_visible', False):
        # cx,cy in obs are normalised [-1, 1]. Place dot accordingly.
        ncx = vt.get('cx_norm', 0.0)
        ncy = vt.get('cy_norm', 0.0)
        px = int(panel_w / 2 + ncx * panel_w / 2)
        py = int(panel_h / 2 + ncy * panel_h / 2)
        cv2.circle(panel, (px, py), 9, (255, 0, 0), -1)
        cv2.circle(panel, (px, py), 9, (255, 255, 255), 2)


def draw_bird(panel, info, panel_w, panel_h, controller_state=None):
    """Add structured telemetry overlay to bird's-eye panel."""
    cv2.putText(panel, "Bird's-eye", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    y = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = [
        ('Step',    f"{info.get('Step', 0)}"),
        ('Time',    f"{info.get('Step', 0) * 0.01:.2f} s"),
        ('Speed',   f"{info.get('TargetSpeed', 0):.2f} m/s"),
        ('Reward',  f"{info.get('CumReward', 0):.1f}"),
    ]
    for k, v in rows:
        cv2.putText(panel, f"{k}: {v}", (10, y), font, 0.50, (0, 255, 0), 1)
        y += 22

    vt = info.get('visual_tracking', {})
    visible = vt.get('target_visible', False)
    color = (0, 255, 0) if visible else (200, 80, 80)
    cv2.putText(panel, f"Visible: {'YES' if visible else 'NO'}",
                (10, y), font, 0.55, color, 2)
    y += 26

    # Controller state badge (full-flight)
    if controller_state is not None:
        c = STATE_COLORS.get(controller_state, (255, 255, 255))
        cv2.rectangle(panel, (0, panel_h - 50), (panel_w, panel_h),
                      (0, 0, 0), -1)
        cv2.putText(panel, f"MODE: {controller_state.upper()}",
                    (10, panel_h - 20), font, 0.75, c, 2)


def get_bird_image(ext_camera, panel_w, panel_h):
    ok, rgba = ext_camera.get_image()
    if not ok:
        return np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    return cv2.resize(rgb, (panel_w, panel_h))


def get_fpv_image(fpv_camera, panel_w, panel_h):
    """Read the HD FPV buffer directly (instead of the env's 32×32 obs image),
    so the video shows what the drone "sees" at HD resolution. The env still
    uses its own 32×32 resize internally for the policy obs."""
    ok, rgba = fpv_camera.get_image()
    if not ok or rgba is None:
        return np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    return cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)


def enrich_info(info, obs_19d):
    """Copy normalised centroid into visual_tracking for the FPV target dot."""
    vt = info.setdefault('visual_tracking', {})
    if obs_19d is not None and len(obs_19d) >= 19:
        vt['cx_norm'] = float(obs_19d[13])
        vt['cy_norm'] = float(obs_19d[14])


# ──────────────────────────────────────────────────────────────────────
# App: holds Panda3D + env + tracking model + (optional) controller
# ──────────────────────────────────────────────────────────────────────

class DemoApp(ShowBase):
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

        # Default Panda3D camera position (also used for video bird's-eye)
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -10, 18)
        self.cam.lookAt(0, 0, 4)

        # FPV camera attached to drone
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.1)
        self.fpv_camera.cam.setHpr(0, -90, 0)
        self.fpv_camera.buffer.setActive(1)

        # External camera for bird's-eye video panel
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.cam.setPos(0, -10, 18)
        self.ext_camera.cam.lookAt(0, 0, 4)
        self.ext_camera.buffer.setActive(1)

        # Build env (always v10 wrapper so target_speed is configurable)
        max_steps = args.max_steps or (3000 if args.mode == 'tracking' else 6000)
        # Full-flight needs altitude termination disabled: the spiral controller
        # spawns the drone at hover_height (z=1.394) which equals target_z, so
        # z_rel=0 < z_min=0.5 would terminate immediately. Tracking demo keeps
        # the original bounds (it spawns ABOVE target as in training).
        if args.mode == 'full-flight':
            z_min, z_max = -3.0, 10.0
            invis_term = 99999  # never auto-truncate by invisibility (the demo IS the search)
            # Disable the env-level search timeout (default 1000 = 10 s).
            # Without this the spiral never has time to grow far enough to
            # reach the target — the env truncates at exactly step 1000.
            search_timeout = 999_999
        else:
            z_min, z_max = 0.5, 3.0
            invis_term = 100
            search_timeout = 1000  # default behaviour for tracking demo
        print(f"Creating env (max_ep_steps={max_steps}, "
              f"z_min={z_min}, z_max={z_max})...")
        self.raw_env = HoverTrackV10Wrapper(
            panda3d_app=self, quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True, use_depth=False,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            use_target=True, target_mode='moving',
            target_speed=args.target_speed, target_radius=0.25,
            lemniscate_scale=2.0,
            filming_mode=True, enable_collisions=False,
            n=max_steps, t_step=0.01, direct_control=1,
            centroid_obs=True, camera_down=True,
            hover_height=1.394,
            use_new_reward=True, constrained_init=True,
            init_pos_range=0.2, init_vel_range=0.10, init_ang_range=0.03,
            reward_version='v3.1',
            spawn_height=1.5, jitter_xy=0.20,
            w_alive=0.10, w_jerk=0.20, w_invisible=1.0,
            z_min=z_min, z_max=z_max,
            invisible_term_steps=invis_term,
            crash_penalty=2.0,
            use_curriculum_speed=True,
            search_timeout_steps=search_timeout,
        )
        self.raw_env._curriculum_target_speed = float(args.target_speed)
        self.max_steps = max_steps

        # Wrap for VecNormalize then load tracking model
        mon = Monitor(self.raw_env)
        self.vec_env = DummyVecEnv([lambda: mon])
        self.vec_env = VecNormalize.load(args.vec_norm, self.vec_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        self.tracking_model = SAC.load(args.model, env=self.vec_env, device='auto')
        print(f"Loaded tracking model: {args.model}")

        # Spiral controller (only full-flight)
        self.controller = None
        if args.mode == 'full-flight':
            if args.use_v2:
                self.controller = SpiralSearchControllerV2(
                    spiral_model_path=args.spiral_model,
                    omega=1.8, r_growth=0.12,
                    climb_offset=args.climb_offset,
                    climb_duration_steps=args.climb_duration_steps,
                    vision_radius=0.5,
                    invisible_threshold=args.invisible_threshold,
                    handoff_steps=args.handoff_steps,
                )
                print(f"Loaded spiral search model V2: {args.spiral_model}  "
                      f"(climb_offset={args.climb_offset} m, "
                      f"climb_steps={args.climb_duration_steps}, "
                      f"handoff_steps={args.handoff_steps}, "
                      f"velocity_damp={'OFF' if args.no_velocity_damp else 'ON'})")
            else:
                self.controller = SpiralSearchController(
                    spiral_model_path=args.spiral_model,
                    omega=1.8, r_growth=0.12,
                    hover_height=1.394, vision_radius=0.5,
                    invisible_threshold=args.invisible_threshold,
                    handoff_steps=args.handoff_steps,
                )
                print(f"Loaded spiral search model: {args.spiral_model}  "
                      f"(handoff_steps={args.handoff_steps}, "
                      f"velocity_damp={'OFF' if args.no_velocity_damp else 'ON'})")


# ──────────────────────────────────────────────────────────────────────
# Demo runners
# ──────────────────────────────────────────────────────────────────────

def open_writer(output_path, panel_w, panel_h, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps,
        (panel_w * 2, panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter at {output_path}")
    return writer


def run_tracking_demo(app, args, output_path, panel_w, panel_h):
    print(f"\n=== Tracking demo: target_speed={args.target_speed} m/s ===")
    obs, info = app.raw_env.reset(seed=args.seed)
    app.taskMgr.step()

    writer = open_writer(output_path, panel_w, panel_h, args.fps)

    step = 0
    visible = 0
    cum_reward = 0.0
    max_steps = app.max_steps

    while step < max_steps:
        obs_in = app.vec_env.normalize_obs(np.asarray(obs, dtype=np.float32))
        act, _ = app.tracking_model.predict(obs_in, deterministic=True)
        obs, r, term, trunc, info = app.raw_env.step(act)
        app.taskMgr.step()
        cum_reward += float(r)
        if info.get('visual_tracking', {}).get('target_visible', False):
            visible += 1
        step += 1

        if step % args.frame_step == 0:
            enrich_info(info, obs)
            info['Step'] = step
            info['CumReward'] = cum_reward
            info['TargetSpeed'] = args.target_speed
            fpv = get_fpv_image(app.fpv_camera, panel_w, panel_h)
            bird = get_bird_image(app.ext_camera, panel_w, panel_h)
            draw_fpv(fpv, info, panel_w, panel_h)
            draw_bird(bird, info, panel_w, panel_h)
            frame = np.hstack([fpv, bird])
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if term or trunc:
            break

    writer.release()
    vis_pct = 100 * visible / max(step, 1)
    print(f"  Steps run:        {step}/{max_steps}")
    print(f"  Visibility:       {vis_pct:.1f}%")
    print(f"  Cumulative reward {cum_reward:.1f}")
    print(f"  Saved video:      {output_path}")


def run_full_flight_demo(app, args, output_path, panel_w, panel_h):
    print(f"\n=== Full flight demo: spawn_offset={args.spawn_offset} m, "
          f"target_speed={args.target_speed} m/s ===")
    obs, info = app.raw_env.reset(seed=args.seed)
    app.taskMgr.step()

    # Decide drone and target positions.
    # Two scenarios are supported via --scenario; otherwise we fall back
    # to the legacy "offset from random target" placement.
    if args.scenario in ('target-below', 'target-outside'):
        # Start the lemniscate at phase=π/2 (lemniscate_point returns the
        # origin there) so the target begins at (0,0,0). The lemniscate
        # then advances at args.target_speed — pass 0.0 for a static
        # target, or e.g. 0.10 for a moving target along the ∞ curve.
        app.raw_env._lemniscate_phase = float(np.pi / 2)
        app.raw_env._target_time = 0.0
        app.raw_env.target_speed = float(args.target_speed)
        app.raw_env.target_pos = np.array([0.0, 0.0, 0.0])
        app.raw_env._update_target_marker_pos()

    if args.scenario == 'target-below':
        # Drone directly above the target; downward camera sees it on
        # frame 1 → controller stays in TRACK; no spiral.
        drone_x, drone_y = 0.0, 0.0
        drone_z = float(args.hover_z)
        print(f"Scenario: target-below  target=(0.00,0.00,0.00)  "
              f"drone=({drone_x:.2f},{drone_y:.2f},{drone_z:.2f})")
    elif args.scenario == 'target-outside':
        # Drone spawned laterally far so the downward camera does NOT
        # see the target at (0,0,0) → SEARCH activates after
        # --invisible-threshold and the v2 spiral runs.
        off = float(args.scenario_offset)
        drone_x = float(off)
        drone_y = float(off * 0.6)
        drone_z = float(args.hover_z)
        print(f"Scenario: target-outside  target=(0.00,0.00,0.00)  "
              f"drone=({drone_x:.2f},{drone_y:.2f},{drone_z:.2f})  "
              f"lateral={off:.2f} m")
    else:
        # Legacy v1 placement (offset from whatever target the wrapper
        # randomised). Used by full-flight runs that don't pass --scenario.
        target_pos = app.raw_env.target_pos
        drone_x = float(target_pos[0] + args.spawn_offset)
        drone_y = float(target_pos[1] + args.spawn_offset * 0.6)
        # v1 required absolute z=1.394; v2 doesn't, but we keep it as the
        # legacy default to preserve old behaviour when --use-v2 is off.
        drone_z = float(args.hover_z) if args.use_v2 else 1.394

    state = app.raw_env.base_env.state.copy()
    state[0] = drone_x
    state[2] = drone_y
    state[4] = drone_z
    state[1] = state[3] = state[5] = 0.0
    state[6] = 1.0
    state[7:10] = 0.0
    state[10:13] = 0.0
    app.raw_env.base_env.state = state.copy()
    app.raw_env.base_env.previous_state = state.copy()
    app.raw_env._update_visualization()
    app.graphicsEngine.renderFrame()
    app.raw_env._capture_camera_images(force_capture=True)
    obs = app.raw_env._build_observation(state.astype(np.float32))

    app.controller.reset()

    writer = open_writer(output_path, panel_w, panel_h, args.fps)

    step = 0
    cum_reward = 0.0
    state_counts = {'track': 0, 'search': 0, 'handoff': 0}
    first_lock_step = -1
    max_steps = app.max_steps
    last_visible = False
    prev_state = 'track'  # controller initial state

    # Strict-visibility filter for SEARCH→TRACK transition.
    # We only consider the target "visible enough to take over" if its
    # centroid lives inside a central circle in the image for several
    # consecutive frames. While in TRACK, raw visibility is used so the
    # controller can still detect target loss and fall back to SEARCH.
    central_streak = 0
    uncentred_visible_streak = 0
    central_filter_on = float(args.central_fov_radius) > 0.0

    # Spiral restart on sight: when the target becomes raw-visible during
    # SEARCH but the central filter still blocks TRACK, re-anchor the
    # spiral at the drone's current XYZ so the next arc spirals around
    # the new position instead of drifting further on the previous one.
    raw_visible_prev = False
    spiral_restart_step = -10_000  # far in the past
    spiral_restart_count = 0

    while step < max_steps:
        # ── Strict visibility for SEARCH→TRACK ────────────────────────
        # Use the centroid from the current obs (set by the previous
        # env.step) to decide if the target is centred enough. Only
        # apply the filter while the controller is in SEARCH/HANDOFF;
        # when in TRACK, raw visibility is used so the controller can
        # still detect target loss.
        state_before = app.controller.current_state
        ctrl = app.controller

        is_central_now = False
        if central_filter_on and last_visible:
            cx_norm = float(obs[13])
            cy_norm = float(obs[14])
            centroid_dist = math.sqrt(cx_norm * cx_norm + cy_norm * cy_norm)
            is_central_now = centroid_dist < float(args.central_fov_radius)

        if is_central_now:
            central_streak += 1
            uncentred_visible_streak = 0
        else:
            central_streak = 0
            uncentred_visible_streak = (
                uncentred_visible_streak + 1 if last_visible else 0)

        if central_filter_on and state_before in ('search', 'handoff'):
            strict_ok = central_streak >= int(args.central_fov_frames)
            timeout_ok = (int(args.central_fov_timeout) > 0
                          and uncentred_visible_streak
                          >= int(args.central_fov_timeout))
            visible_for_ctrl = bool(strict_ok or timeout_ok)
        else:
            visible_for_ctrl = bool(last_visible)

        # ── Spiral restart on sight ───────────────────────────────────
        # If the target just appeared in raw FOV during SEARCH but the
        # central filter still blocks TRACK, re-anchor the spiral at
        # the drone's current XYZ. With a cooldown so we don't restart
        # every time the target flashes in/out.
        raw_visible_now = bool(last_visible)
        just_acquired = raw_visible_now and not raw_visible_prev
        raw_visible_prev = raw_visible_now

        cooldown_ok = (
            int(args.spiral_restart_cooldown) > 0
            and (step - spiral_restart_step)
            >= int(args.spiral_restart_cooldown)
        )
        if (state_before == 'search'
                and just_acquired
                and not visible_for_ctrl
                and cooldown_ok):
            st = app.raw_env.base_env.state
            app.controller._reset_spiral(
                float(st[0]), float(st[2]), float(st[4]))
            spiral_restart_step = step
            spiral_restart_count += 1
            print(f"  [Step {step}] SPIRAL RESTART #{spiral_restart_count} "
                  f"at ({st[0]:+.2f},{st[2]:+.2f},{st[4]:+.2f}) "
                  f"— target visible but not centred")

        # ── Pre-action damping ────────────────────────────────────────
        damp_now = False
        damp_label = ''
        if not args.no_velocity_damp:
            if state_before == 'search' and visible_for_ctrl:
                damp_now = True
                damp_label = ('SEARCH→TRACK  ' if ctrl.handoff_steps <= 0
                              else 'SEARCH→HANDOFF')
            elif state_before == 'handoff':
                if visible_for_ctrl:
                    if ctrl._handoff_step + 1 >= ctrl.handoff_steps:
                        damp_now = True
                        damp_label = 'HANDOFF→TRACK '
                else:
                    damp_now = True
                    damp_label = 'HANDOFF→SEARCH'

        if damp_now:
            s = app.raw_env.base_env.state.copy()
            s[1] = s[3] = s[5] = 0.0    # linear vels (vx, vy, vz)
            s[10:13] = 0.0               # angular vels (wx, wy, wz)
            s[6] = 1.0                   # quaternion identity (level pose)
            s[7:10] = 0.0
            app.raw_env.base_env.state = s.copy()
            app.raw_env.base_env.previous_state = s.copy()
            # Rebuild the visualisation and re-capture the camera so the
            # observation that v10.4 will see reflects the clean state.
            app.raw_env._update_visualization()
            app.graphicsEngine.renderFrame()
            app.raw_env._capture_camera_images(force_capture=True)
            obs = app.raw_env._build_observation(s.astype(np.float32))
            print(f"  [Step {step}] {damp_label}: vels=0, quat=identity "
                  f"(pre-action damp)")

        # Normalise obs (now possibly rebuilt from the damped state) and
        # ask the controller for an action. Pass the FILTERED visibility
        # so the controller transitions SEARCH→TRACK only when the
        # centroid is centred (per --central-fov-radius). The raw
        # visibility is still used while in TRACK so target loss is
        # detected immediately.
        obs_in = app.vec_env.normalize_obs(np.asarray(obs, dtype=np.float32))
        act = app.controller.get_action(
            obs_in, visible_for_ctrl, app.tracking_model, app.raw_env)
        state_after = app.controller.current_state

        obs, r, term, trunc, info = app.raw_env.step(act)
        app.taskMgr.step()
        cum_reward += float(r)
        last_visible = info.get('visual_tracking', {}).get('target_visible', False)
        cs = app.controller.current_state
        state_counts[cs] = state_counts.get(cs, 0) + 1
        # Real lock-on = transition from SEARCH (or HANDOFF) into TRACK after
        # the spiral phase. The controller starts in TRACK initially, so we
        # only count it if we previously visited SEARCH.
        if cs == 'track' and prev_state in ('search', 'handoff') and first_lock_step < 0:
            first_lock_step = step
        prev_state = cs
        step += 1

        if step % args.frame_step == 0:
            enrich_info(info, obs)
            info['Step'] = step
            info['CumReward'] = cum_reward
            info['TargetSpeed'] = args.target_speed
            fpv = get_fpv_image(app.fpv_camera, panel_w, panel_h)
            bird = get_bird_image(app.ext_camera, panel_w, panel_h)
            draw_fpv(fpv, info, panel_w, panel_h,
                     central_fov_radius=float(args.central_fov_radius),
                     centred_now=is_central_now)
            draw_bird(bird, info, panel_w, panel_h, controller_state=cs)
            frame = np.hstack([fpv, bird])
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if term or trunc:
            break

    writer.release()
    print(f"  Total steps:      {step}")
    print(f"  Search:           {state_counts.get('search', 0)} steps "
          f"({state_counts.get('search', 0) * 0.01:.1f} s)")
    print(f"  Handoff:          {state_counts.get('handoff', 0)} steps "
          f"({state_counts.get('handoff', 0) * 0.01:.1f} s)")
    print(f"  Track:            {state_counts.get('track', 0)} steps "
          f"({state_counts.get('track', 0) * 0.01:.1f} s)")
    if first_lock_step > 0:
        print(f"  First lock-on at: step {first_lock_step} "
              f"(~{first_lock_step * 0.01:.1f} s)")
    else:
        print(f"  First lock-on:    NEVER (target not found in {step} steps)")
    print(f"  Spiral restarts:  {spiral_restart_count}")
    print(f"  Cumulative reward {cum_reward:.1f}")
    print(f"  Saved video:      {output_path}")


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Parse --win-size (accepts "WxH" or "W H")
    win_str = args.win_size.lower().replace(' ', 'x')
    win_w, win_h = (int(x) for x in win_str.split('x'))
    if args.no_display:
        loadPrcFileData('', 'window-type offscreen')
        loadPrcFileData('', f'win-size {win_w} {win_h}')
        loadPrcFileData('', 'undecorated true')
    else:
        loadPrcFileData('', f'win-size {win_w} {win_h}')

    panel_w, panel_h = (int(x) for x in args.panel_size.lower().split('x'))

    # Default output path
    if not args.output_path:
        outdir = Path('./models/hover_track_v10_4/recordings')
        outdir.mkdir(parents=True, exist_ok=True)
        speed_tag = f"{args.target_speed:.2f}".replace('.', '')
        if args.mode == 'tracking':
            args.output_path = outdir / f"demo_tracking_speed_{speed_tag}.mp4"
        else:
            scn = f"_{args.scenario}" if args.scenario else ""
            v2_tag = "_v2" if args.use_v2 else ""
            cfov_tag = (f"_cfov{int(round(args.central_fov_radius * 100)):02d}"
                        if args.central_fov_radius > 0.0 else "")
            args.output_path = outdir / (
                f"demo_full_flight_speed_{speed_tag}"
                f"{scn}{v2_tag}{cfov_tag}.mp4")
    else:
        args.output_path = Path(args.output_path)

    app = DemoApp(args)
    if args.mode == 'tracking':
        run_tracking_demo(app, args, args.output_path, panel_w, panel_h)
    else:
        run_full_flight_demo(app, args, args.output_path, panel_w, panel_h)


if __name__ == "__main__":
    main()
