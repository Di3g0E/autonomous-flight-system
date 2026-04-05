#!/usr/bin/env python
"""
Spiral-to-hover video test — only two RL models, no PD intermediary.

The target is placed statically below the drone but offset in XY so it
is outside the camera FOV.  Two models alternate:

  SEARCH (PPO spiral)  ←→  TRACK (SAC v2)

  - Target invisible → SEARCH: PPO spiral from current position.
  - Target visible   → TRACK:  SAC v2 stabilises over it.

No PD controller, no BRAKE, no HANDOFF — raw model-to-model switch.

Output:  experiments/spiral_to_hover/

Usage:
    python tests/test_spiral_to_hover_video.py
    python tests/test_spiral_to_hover_video.py --duration 30 --offset 2.0
"""

import argparse
import math
import os
import sys
import numpy as np
import cv2
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: F401  — must precede Panda3D to avoid DLL conflicts

from panda3d.core import Filename, loadPrcFile
from direct.showbase.ShowBase import ShowBase

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from stable_baselines3 import SAC, PPO

from src.simulation.world_setup import world_setup, quad_setup
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


# ── defaults ───────────────────────────────────────────────────────────
DURATION_S = 30
FPS = 30
PANEL_SIZE = 480
LABEL_H = 40
SEP = 3
HOVER_MODEL = './models/hover_track_v2/best_model.zip'
SPIRAL_MODEL = './models/spiral_follow/best_model.zip'
TARGET_OFFSET = 1.5
HOVER_HEIGHT = 1.394


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--hover-model', default=HOVER_MODEL)
    p.add_argument('--spiral-model', default=SPIRAL_MODEL)
    p.add_argument('--duration', type=int, default=DURATION_S)
    p.add_argument('--fps', type=int, default=FPS)
    p.add_argument('--panel-size', type=int, default=PANEL_SIZE)
    p.add_argument('--offset', type=float, default=TARGET_OFFSET)
    p.add_argument('--hover-height', type=float, default=HOVER_HEIGHT)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Two-state controller: SEARCH (PPO spiral) ←→ TRACK (SAC v2)
# ═══════════════════════════════════════════════════════════════════════

class SpiralToTrackController:

    SEARCH = 'search'
    TRACK  = 'track'

    def __init__(self, spiral_model_path,
                 spiral_hover_height=1.39, omega=1.8, r_growth=0.12,
                 vision_radius=0.5, invisible_threshold=20):

        self.spiral_model = PPO.load(str(spiral_model_path))
        self.spi_hh = spiral_hover_height
        self.omega = omega
        self.r_growth = r_growth
        self.vr = vision_radius
        self.K = invisible_threshold
        self.reset()

    def reset(self):
        self._st = self.SEARCH
        self._inv = 0
        # spiral reference
        self._sp = 0
        self._th = 0.0
        self._cx = self._cy = 0.0
        self._rx = self._ry = 0.0
        self._rvx = self._rvy = 0.0
        self._spiral_started = False

    # ── spiral reference (mirrors SpiralFollowEnv) ─────────────────────

    def _reset_spiral(self, x, y):
        self._sp = 0; self._th = 0.0
        self._cx, self._cy = x, y
        r0 = 0.05
        self._rx = x + r0; self._ry = y
        self._rvx = self.r_growth; self._rvy = r0 * self.omega
        self._spiral_started = True

    def _advance_spiral(self, dt):
        self._sp += 1
        t = self._sp * dt
        r = self.r_growth * t + 0.05
        ab = 0.70 * 9.82 * math.sin(0.25)
        w = min(self.omega, math.sqrt(ab / max(r, 0.05)))
        self._th += w * dt
        c, s = math.cos(self._th), math.sin(self._th)
        dr = self.r_growth
        self._rx = self._cx + r * c
        self._ry = self._cy + r * s
        self._rvx = dr * c - r * w * s
        self._rvy = dr * s + r * w * c

    def _spiral_obs(self, state):
        dx = (self._rx - state[0]) / self.vr
        dy = (self._ry - state[2]) / self.vr
        vm = max(math.sqrt(self._rvx**2 + self._rvy**2), 1e-3)
        dz = (self.spi_hh - state[4]) / max(self.spi_hh, 0.1)
        ref = np.array([dx, dy, self._rvx/vm, self._rvy/vm, dz],
                       dtype=np.float32)
        return np.concatenate([state.astype(np.float32), ref])

    # ── public API ─────────────────────────────────────────────────────

    def get_action(self, obs19, target_visible, sac_model, env):
        state = env.base_env.state.copy()
        dt = env.base_env.t_step

        # ── transitions ───────────────────────────────────────────────
        if self._st == self.SEARCH:
            if target_visible:
                self._st = self.TRACK
                self._inv = 0

        elif self._st == self.TRACK:
            if target_visible:
                self._inv = 0
            else:
                self._inv += 1
                if self._inv >= self.K:
                    self._reset_spiral(state[0], state[2])
                    self._st = self.SEARCH

        # ── actions ───────────────────────────────────────────────────
        if self._st == self.SEARCH:
            if not self._spiral_started:
                self._reset_spiral(state[0], state[2])
            self._advance_spiral(dt)
            action, _ = self.spiral_model.predict(
                self._spiral_obs(state), deterministic=True)
            return action

        # TRACK
        action, _ = sac_model.predict(obs19, deterministic=True)
        return action

    @property
    def current_state(self):
        return self._st


# ═══════════════════════════════════════════════════════════════════════
#  Video overlays
# ═══════════════════════════════════════════════════════════════════════

STATE_COLOURS = {
    'search': (0, 165, 255),   # orange
    'track':  (0, 255, 0),     # green
}
STATE_LABELS = {
    'search': 'SPIRAL SEARCH (PPO)',
    'track':  'TRACKING (SAC v2)',
}


# ═══════════════════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════════════════

class SpiralToHoverVideoApp(ShowBase):

    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        self.duration_s = args.duration
        self.fps = args.fps
        self.ps = args.panel_size
        self.hh = args.hover_height

        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()
        self.output_dir = Path(project_root) / "experiments" / "spiral_to_hover"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── scene ──────────────────────────────────────────────────────
        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)

        # FPV camera — pointing down
        self.fpv_cam = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_cam.cam.reparentTo(self.quad_model)
        self.fpv_cam.cam.setPos(0, 0, 0.01)
        self.fpv_cam.cam.lookAt(0, 0, 0)
        self.fpv_cam.buffer.setActive(1)

        # external camera
        self.ext_cam = opencv_camera(self, 'ext_cam', 1)
        self.ext_cam.cam.reparentTo(self.render)
        self.ext_cam.buffer.setActive(1)

        # ── environment ────────────────────────────────────────────────
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True, use_depth=False,
            use_target=True, target_mode='fixed',
            target_range=5.0, target_speed=0.0,
            camera_high_freq_obj=self.fpv_cam,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            enable_collisions=False,
            n=args.duration * 100 + 200,
            t_step=0.01, direct_control=1,
            target_radius=0.25,
            filming_mode=True,
            camera_down=True,
            hover_height=self.hh,
            centroid_obs=True,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.05,
            init_vel_range=0.05,
            init_ang_range=0.02,
        )

        # ── models ─────────────────────────────────────────────────────
        for p, n in [(args.hover_model, "SAC v2"),
                     (args.spiral_model, "PPO spiral")]:
            if not os.path.exists(p):
                print(f"ERROR: {n} not found: {p}"); sys.exit(1)

        self.sac = SAC.load(args.hover_model, env=None)
        print(f"SAC v2 model:    {args.hover_model}")

        self.ctrl = SpiralToTrackController(
            spiral_model_path=args.spiral_model,
        )
        print(f"Spiral model:    {args.spiral_model}")

        self.taskMgr.doMethodLater(0.5, self._run, 'run')

    # ── video helpers ──────────────────────────────────────────────────

    def _panel(self, img, title, colour=(0, 255, 255)):
        ps = self.ps
        r = cv2.resize(img, (ps, ps))
        p = np.zeros((ps + LABEL_H, ps, 3), np.uint8)
        p[LABEL_H:] = r
        cv2.putText(p, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, colour, 2)
        return p

    def _detection(self, rgb32, st):
        h, w = rgb32.shape[:2]
        UP = self.ps // w
        bgr = cv2.cvtColor(rgb32, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))
        pc = int(np.sum(mask > 0))
        frac = pc / (h * w)

        ann = cv2.resize(bgr, (self.ps, self.ps),
                         interpolation=cv2.INTER_NEAREST)
        mu = cv2.resize(mask, (self.ps, self.ps),
                        interpolation=cv2.INTER_NEAREST)
        ov = np.zeros_like(ann); ov[mu > 0] = (0, 255, 0)
        ann = cv2.addWeighted(ann, 0.7, ov, 0.5, 0)

        c = self.ps // 2
        cv2.line(ann, (c-20, c), (c+20, c), (255, 255, 255), 1)
        cv2.line(ann, (c, c-20), (c, c+20), (255, 255, 255), 1)

        if pc > 2:
            ys, xs = np.where(mask > 0)
            cx, cy = float(np.mean(xs)) * UP, float(np.mean(ys)) * UP
            cv2.circle(ann, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            cv2.circle(ann, (int(cx), int(cy)), 8, (255, 255, 255), 2)
            xmn, xmx = int(np.min(xs))*UP, int(np.max(xs)+1)*UP
            ymn, ymx = int(np.min(ys))*UP, int(np.max(ys)+1)*UP
            cv2.rectangle(ann, (xmn, ymn), (xmx, ymx), (0, 255, 255), 2)
            cxn = (np.mean(xs) - w/2) / (w/2)
            cyn = (np.mean(ys) - h/2) / (h/2)
            txt = f"cx={cxn:+.2f} cy={cyn:+.2f} frac={frac:.3f}"
        else:
            txt = "NOT DETECTED"

        cv2.putText(ann, txt, (8, self.ps - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        sc = STATE_COLOURS.get(st, (200, 200, 200))
        sl = STATE_LABELS.get(st, st)
        cv2.putText(ann, sl, (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
        return ann

    def _badge(self, frame, st, step):
        sc = STATE_COLOURS.get(st, (200, 200, 200))
        sl = STATE_LABELS.get(st, st)
        cv2.putText(frame, sl, (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
        cv2.putText(frame, f"t={step*0.01:.1f}s", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame

    # ── main loop ──────────────────────────────────────────────────────

    def _run(self, task):
        ps = self.ps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w1 = cv2.VideoWriter(str(self.output_dir / '1_raw_camera.mp4'),
                             fourcc, self.fps, (ps, ps))
        w2 = cv2.VideoWriter(str(self.output_dir / '2_rl_input.mp4'),
                             fourcc, self.fps, (ps, ps))
        w3 = cv2.VideoWriter(str(self.output_dir / '3_hsv_detection.mp4'),
                             fourcc, self.fps, (ps, ps))
        w4 = cv2.VideoWriter(str(self.output_dir / '4_external_view.mp4'),
                             fourcc, self.fps, (ps, ps))
        qw = 2 * ps + SEP
        qh = 2 * (ps + LABEL_H) + SEP
        wq = cv2.VideoWriter(str(self.output_dir / 'quad_view.mp4'),
                             fourcc, self.fps, (qw, qh))
        writers = [w1, w2, w3, w4, wq]

        # ── reset + place target ───────────────────────────────────────
        obs, info = self.env.reset()
        drone = self.env.base_env.state[0:5:2].copy()

        # Target: below drone at hover_height, offset in X so it is
        # completely outside the downward camera FOV (~0.56 m half-width)
        off = self.args.offset
        tx = drone[0] + off
        ty = drone[1]
        tz = drone[2] - self.hh

        self.env.target_pos = np.array([tx, ty, tz])
        self.env._update_target_marker_pos()
        self.ctrl.reset()

        print(f"\n  Drone:   ({drone[0]:.2f}, {drone[1]:.2f}, {drone[2]:.2f})")
        print(f"  Target:  ({tx:.2f}, {ty:.2f}, {tz:.2f})  "
              f"[offset={off:.1f} m]")

        # ── fixed external camera ──────────────────────────────────────
        max_r = 0.12 * self.duration_s
        vr = max(max_r, off) + 1.0
        mx = (drone[0] + tx) / 2
        my = (drone[1] + ty) / 2
        lz = drone[2] + 5
        cd = vr * 2.5
        self.ext_cam.cam.setPos(mx - cd * 0.6, my - cd * 0.6,
                                lz + cd * 0.55)
        self.ext_cam.cam.lookAt(float(mx), float(my), float(lz - 1))

        # ── warm-up (neutral action, just for Panda3D buffers) ─────────
        neutral = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        for _ in range(15):
            obs, _, _, _, info = self.env.step(neutral)
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()

        # ── recording ──────────────────────────────────────────────────
        total = self.duration_s * 100
        fi = max(1, 100 // self.fps)
        nf = 0
        prev = self.ctrl.current_state
        miles = {}

        print(f"\n  Recording {self.duration_s}s  "
              f"({total} steps, {self.fps} fps)...\n")

        for step in range(total):
            vt = info.get('visual_tracking', {})
            vis = vt.get('target_visible', False)

            action = self.ctrl.get_action(obs, vis, self.sac, self.env)
            cs = self.ctrl.current_state

            if cs != prev:
                vl = math.sqrt(self.env.base_env.state[1]**2
                               + self.env.base_env.state[3]**2)
                print(f"  [{step*0.01:6.2f}s] {prev:10s} -> {cs:10s}"
                      f"  v_lat={vl:.3f} m/s")
                miles.setdefault(cs, step)
                prev = cs

            obs, _, term, trunc, info = self.env.step(action)

            if term or trunc:
                obs, info = self.env.reset()
                self.env.target_pos = np.array([tx, ty, tz])
                self.env._update_target_marker_pos()
                self.ctrl.reset()

            self.taskMgr.step()

            if step % fi != 0:
                continue

            rgb32 = self.env._last_high_freq_image
            if rgb32 is None:
                continue
            ok1, fpv_rgba = self.fpv_cam.get_image()
            ok2, ext_rgba = self.ext_cam.get_image()
            if not ok1 or not ok2:
                continue

            raw = cv2.resize(cv2.cvtColor(fpv_rgba, cv2.COLOR_RGBA2BGR),
                             (ps, ps))
            self._badge(raw, cs, step)
            w1.write(raw)

            rl = cv2.resize(cv2.cvtColor(rgb32, cv2.COLOR_RGB2BGR),
                            (ps, ps), interpolation=cv2.INTER_NEAREST)
            self._badge(rl, cs, step)
            w2.write(rl)

            det = self._detection(rgb32, cs)
            w3.write(det)

            ext = cv2.resize(cv2.cvtColor(ext_rgba, cv2.COLOR_RGBA2BGR),
                             (ps, ps))
            self._badge(ext, cs, step)
            w4.write(ext)

            p1 = self._panel(raw, "1. Raw Camera")
            p2 = self._panel(rl, "2. RL Input (32x32)", (255, 0, 255))
            p3 = self._panel(det, "3. HSV Detection", (0, 255, 0))
            p4 = self._panel(ext, "4. External View")
            ph = ps + LABEL_H
            sv = np.full((ph, SEP, 3), 255, np.uint8)
            sh = np.full((SEP, 2 * ps + SEP, 3), 255, np.uint8)
            wq.write(np.vstack([np.hstack([p1, sv, p2]),
                                sh,
                                np.hstack([p3, sv, p4])]))
            nf += 1

        for w in writers:
            w.release()

        # ── report ─────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  SPIRAL -> HOVER (PPO + SAC v2, direct switch)")
        print(f"{'='*60}")
        print(f"  Duration:  {self.duration_s}s   Frames: {nf}")
        print(f"  Offset:    {off:.1f} m")
        for s, n in sorted(miles.items(), key=lambda x: x[1]):
            print(f"  {s:14s}  step {n:5d}  (t={n*0.01:.2f}s)")
        print(f"\n  Saved to {self.output_dir}/")
        print(f"{'='*60}\n")

        self.userExit()
        return task.done


if __name__ == "__main__":
    args = parse_args()
    app = SpiralToHoverVideoApp(args)
    app.run()
