#!/usr/bin/env python
"""
Hover-track v3/v3.1 video evaluation — records quad-view per difficulty tier.

For each tier (easy / medium / hard) the script records five synchronised
videos plus a 2×2 quad-view:

  1. raw_camera.mp4       — high-res downward camera image.
  2. rl_input.mp4         — 32×32 image the policy sees (upscaled).
  3. hsv_detection.mp4    — HSV magenta mask + centroid + bounding box.
  4. external_view.mp4    — third-person view of drone + target.
  5. quad_view.mp4        — 2×2 combined grid with live metrics overlay.

Supports both v3 and v3.1 models via --reward-version flag.

Output:
    experiments/hover_track_v3_video/<tier>/   (v3)
    experiments/hover_track_v3_1_video/<tier>/ (v3.1)

Usage:
    # v3 model (850k checkpoint)
    python tests/test_hover_track_v3_video.py

    # v3.1 model (fine-tuned)
    python tests/test_hover_track_v3_video.py --reward-version v3.1

    python tests/test_hover_track_v3_video.py --checkpoint 900000
    python tests/test_hover_track_v3_video.py --duration 10 --fps 30 --panel-size 540
    python tests/test_hover_track_v3_video.py --tiers easy hard
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: F401

from panda3d.core import Filename, loadPrcFile
from direct.showbase.ShowBase import ShowBase

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from stable_baselines3 import SAC

from src.simulation.world_setup import world_setup, quad_setup
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv

# ── Defaults ──────────────────────────────────────────────────────────
DURATION_S = 10
FPS = 30
PANEL_SIZE = 480
LABEL_H = 40
SEP = 3
DEFAULT_CHECKPOINT = 850_000
CKPT_DIR = './models/hover_track_v3/checkpoints'

TIERS = {
    'easy':   {'offset': 0.2, 'vel': 0.10, 'ang': 0.05},
    'medium': {'offset': 0.6, 'vel': 0.25, 'ang': 0.10},
    'hard':   {'offset': 1.0, 'vel': 0.35, 'ang': 0.15},
}

TIER_COLORS = {
    'easy':   (0, 200, 0),     # green
    'medium': (0, 180, 255),   # orange
    'hard':   (0, 0, 255),     # red
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Record quad-view videos of hover-track v3/v3.1 per tier")
    p.add_argument('--reward-version', type=str, default='v3',
                   choices=['v3', 'v3.1'],
                   help="Reward version: v3 (default) or v3.1")
    p.add_argument('--checkpoint', type=int, default=None,
                   help="Checkpoint step (default: 850k for v3, best_model for v3.1)")
    p.add_argument('--model-path', type=str, default=None,
                   help="Direct path to model .zip (overrides --checkpoint)")
    p.add_argument('--checkpoint-dir', type=str, default=None,
                   help="Checkpoint directory (auto-set from reward-version)")
    p.add_argument('--duration', type=int, default=DURATION_S,
                   help="Episode duration in seconds (default: 10)")
    p.add_argument('--fps', type=int, default=FPS,
                   help="Video frame rate (default: 30)")
    p.add_argument('--panel-size', type=int, default=PANEL_SIZE,
                   help="Resolution per panel in pixels (default: 480)")
    p.add_argument('--tiers', nargs='+', default=['easy', 'medium', 'hard'],
                   choices=['easy', 'medium', 'hard'],
                   help="Difficulty tiers to record (default: all)")
    p.add_argument('--output-dir', type=str, default=None,
                   help="Output directory (auto-set from reward-version)")
    args = p.parse_args()

    # Auto-configure paths based on reward version
    if args.reward_version == 'v3.1':
        if args.checkpoint_dir is None:
            args.checkpoint_dir = './models/hover_track_v3_1/checkpoints'
        if args.checkpoint is None and args.model_path is None:
            args.model_path = './models/hover_track_v3_1/best_model.zip'
        if args.output_dir is None:
            args.output_dir = './experiments/hover_track_v3_1_video'
    else:
        if args.checkpoint_dir is None:
            args.checkpoint_dir = CKPT_DIR
        if args.checkpoint is None and args.model_path is None:
            args.checkpoint = DEFAULT_CHECKPOINT
        if args.output_dir is None:
            args.output_dir = './experiments/hover_track_v3_video'

    return args


def find_checkpoint(ckpt_dir, step):
    """Find checkpoint file, falling back to closest available."""
    ckpt_dir = Path(ckpt_dir)
    exact = ckpt_dir / f'model_{step}_steps.zip'
    if exact.exists():
        return exact

    available = {}
    for f in ckpt_dir.glob('model_*_steps.zip'):
        try:
            s = int(f.stem.split('_')[1])
            available[s] = f
        except (ValueError, IndexError):
            pass

    if not available:
        print(f"ERROR: No checkpoints in {ckpt_dir}")
        sys.exit(1)

    closest = min(available.keys(), key=lambda x: abs(x - step))
    print(f"  Note: {step} not found, using closest: {closest}")
    return available[closest]


class HoverTrackV3VideoApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Load 3D world ──
        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)

        # ── FPV camera (downward) ──
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, 0.01)
        self.fpv_camera.cam.lookAt(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # ── External camera ──
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.buffer.setActive(1)

        # ── Environment (v3 or v3.1 reward) ──
        self.reward_version = args.reward_version
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='fixed',
            target_range=3.0,
            target_speed=0.0,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            enable_collisions=False,
            n=args.duration * 100 + 50,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            filming_mode=True,
            camera_down=True,
            hover_height=1.394,
            centroid_obs=True,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.3,
            init_vel_range=0.10,
            init_ang_range=0.05,
            reward_version=args.reward_version,
        )

        # ── Load model ──
        if args.model_path:
            model_path = Path(args.model_path)
            if not model_path.exists():
                print(f"ERROR: Model not found: {model_path}")
                sys.exit(1)
            self.ckpt_step = 'best'
        else:
            model_path = find_checkpoint(args.checkpoint_dir, args.checkpoint)
            self.ckpt_step = args.checkpoint
        self.model = SAC.load(str(model_path), env=None)
        print(f"Model loaded: {model_path.name}  "
              f"(reward: {args.reward_version})")

        # Warm-up Panda3D
        for _ in range(5):
            self.graphicsEngine.renderFrame()

        self.taskMgr.doMethodLater(0.5, self._run_all_tiers, 'run')

    # ──────────────────────────────────────────────────────────────────
    def _make_panel(self, img_bgr, title, title_color=(0, 255, 255)):
        PS = self.args.panel_size
        resized = cv2.resize(img_bgr, (PS, PS))
        panel = np.zeros((PS + LABEL_H, PS, 3), dtype=np.uint8)
        panel[LABEL_H:, :] = resized
        cv2.putText(panel, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, title_color, 2)
        return panel

    # ──────────────────────────────────────────────────────────────────
    def _build_detection_frame(self, img_32_rgb):
        h, w = img_32_rgb.shape[:2]
        UP = self.args.panel_size // w

        img_bgr = cv2.cvtColor(img_32_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))

        pixel_count = int(np.sum(mask > 0))
        total_pixels = h * w
        frac = pixel_count / total_pixels

        annotated = cv2.resize(img_bgr, (self.args.panel_size, self.args.panel_size),
                               interpolation=cv2.INTER_NEAREST)
        mask_up = cv2.resize(mask, (self.args.panel_size, self.args.panel_size),
                             interpolation=cv2.INTER_NEAREST)

        overlay = np.zeros_like(annotated)
        overlay[mask_up > 0] = (0, 255, 0)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.5, 0)

        cx_img, cy_img = self.args.panel_size // 2, self.args.panel_size // 2
        cv2.line(annotated, (cx_img - 20, cy_img), (cx_img + 20, cy_img),
                 (255, 255, 255), 1)
        cv2.line(annotated, (cx_img, cy_img - 20), (cx_img, cy_img + 20),
                 (255, 255, 255), 1)

        visible = pixel_count > 2
        if visible:
            ys, xs = np.where(mask > 0)
            cent_x = float(np.mean(xs)) * UP
            cent_y = float(np.mean(ys)) * UP

            cv2.circle(annotated, (int(cent_x), int(cent_y)), 8,
                       (0, 0, 255), -1)
            cv2.circle(annotated, (int(cent_x), int(cent_y)), 8,
                       (255, 255, 255), 2)

            x_min, x_max = int(np.min(xs)) * UP, int(np.max(xs) + 1) * UP
            y_min, y_max = int(np.min(ys)) * UP, int(np.max(ys) + 1) * UP
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max),
                          (0, 255, 255), 2)

            cx_norm = (np.mean(xs) - w / 2) / (w / 2)
            cy_norm = (np.mean(ys) - h / 2) / (h / 2)
            info_text = f"cx={cx_norm:+.2f} cy={cy_norm:+.2f} frac={frac:.3f}"
        else:
            info_text = "NOT DETECTED"

        cv2.putText(annotated, info_text, (8, self.args.panel_size - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        return annotated

    # ──────────────────────────────────────────────────────────────────
    def _add_metrics_overlay(self, quad_frame, tier_name, step, total_steps,
                             cumul_reward, vt, tier_cfg):
        """Draw live metrics on the quad-view frame."""
        PS = self.args.panel_size
        h = quad_frame.shape[0]

        # Background band at the bottom
        band_h = 50
        y0 = h - band_h
        cv2.rectangle(quad_frame, (0, y0), (quad_frame.shape[1], h),
                      (30, 30, 30), -1)

        tier_col = TIER_COLORS.get(tier_name, (255, 255, 255))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Tier + progress
        pct = 100 * step / max(total_steps, 1)
        t_sec = step / 100.0
        label = (f"TIER: {tier_name.upper()}  |  "
                 f"t={t_sec:.1f}s  ({pct:.0f}%)  |  "
                 f"offset={tier_cfg['offset']}m  "
                 f"vel={tier_cfg['vel']}m/s  "
                 f"ang={tier_cfg['ang']}rad")
        cv2.putText(quad_frame, label, (10, y0 + 20),
                    font, 0.50, tier_col, 1)

        # Metrics
        vis = vt.get('target_visible', False)
        vis_txt = "YES" if vis else "NO"
        vis_col = (0, 255, 0) if vis else (0, 0, 255)
        cd = vt.get('centering_dist', -1)
        frac = vt.get('target_fraction', 0)
        r_stab = vt.get('r_stability', 0)
        r_cent = vt.get('r_centering', 0)
        r_scl = vt.get('r_scale', 0)

        metrics = (f"R_total={cumul_reward:.0f}  |  "
                   f"vis={vis_txt}  cent={cd:.3f}  frac={frac:.3f}  |  "
                   f"r_stab={r_stab:.2f}  r_cent={r_cent:.2f}  "
                   f"r_scale={r_scl:.2f}")
        cv2.putText(quad_frame, metrics, (10, y0 + 42),
                    font, 0.45, (200, 200, 200), 1)

        # Visibility indicator dot
        cv2.circle(quad_frame, (quad_frame.shape[1] - 25, y0 + 15),
                   10, vis_col, -1)

        return quad_frame

    # ──────────────────────────────────────────────────────────────────
    def _record_tier(self, tier_name, tier_cfg):
        """Record one episode for a given tier. Returns episode summary."""
        args = self.args
        PS = args.panel_size

        tier_dir = self.output_dir / tier_name
        tier_dir.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w1 = cv2.VideoWriter(str(tier_dir / '1_raw_camera.mp4'),
                             fourcc, args.fps, (PS, PS))
        w2 = cv2.VideoWriter(str(tier_dir / '2_rl_input.mp4'),
                             fourcc, args.fps, (PS, PS))
        w3 = cv2.VideoWriter(str(tier_dir / '3_hsv_detection.mp4'),
                             fourcc, args.fps, (PS, PS))
        w4 = cv2.VideoWriter(str(tier_dir / '4_external_view.mp4'),
                             fourcc, args.fps, (PS, PS))

        quad_w = 2 * PS + SEP
        quad_h = 2 * (PS + LABEL_H) + SEP + 50  # +50 for metrics band
        w_quad = cv2.VideoWriter(str(tier_dir / 'quad_view.mp4'),
                                 fourcc, args.fps, (quad_w, quad_h))

        writers = [w1, w2, w3, w4, w_quad]

        # ── Reset + configure tier ──
        self.env.init_vel_range = tier_cfg['vel']
        self.env.init_ang_range = tier_cfg['ang']
        self.env.stabilization_only = False

        obs, info = self.env.reset()

        # Apply target offset
        drone_pos = self.env.base_env.state[0:5:2].copy()
        angle = np.random.uniform(0, 2 * np.pi)
        off = tier_cfg['offset']
        dx = off * np.cos(angle)
        dy = off * np.sin(angle)
        self.env.target_pos = np.array([
            drone_pos[0] + dx,
            drone_pos[1] + dy,
            drone_pos[2] - self.env.hover_height,
        ])
        self.env._update_target_marker_pos()

        # Re-render + rebuild observation
        self.graphicsEngine.renderFrame()
        self.env._capture_camera_images(force_capture=True)
        state = self.env.base_env.state.astype(np.float32)
        obs = self.env._build_observation(state)

        # Position external camera
        mx = (drone_pos[0] + self.env.target_pos[0]) / 2
        my = (drone_pos[1] + self.env.target_pos[1]) / 2
        drone_z = drone_pos[2] + 5
        target_z = self.env.target_pos[2] + 5
        mid_z = (drone_z + target_z) / 2
        cam_dist = max(off * 3, 4.0)
        self.ext_camera.cam.setPos(
            mx - cam_dist * 0.6,
            my - cam_dist * 0.6,
            mid_z + cam_dist * 0.4)
        self.ext_camera.cam.lookAt(float(mx), float(my), float(mid_z - 0.5))

        print(f"\n  [{tier_name.upper()}] offset={off}m  "
              f"vel={tier_cfg['vel']}m/s  ang={tier_cfg['ang']}rad")
        print(f"  Target at ({dx:+.2f}, {dy:+.2f}) from drone")

        # Warm-up
        neutral = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        for _ in range(15):
            obs, _, _, _, info = self.env.step(neutral)
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()

        # ── Recording loop ──
        total_steps = args.duration * 100
        frame_interval = max(1, 100 // args.fps)

        frames_written = 0
        cumul_reward = 0.0
        visible_steps = 0
        centering_dists = []
        fractions = []
        terminated_early = False

        for step in range(total_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            cumul_reward += reward

            vt = info.get('visual_tracking', {})
            if vt.get('target_visible', False):
                visible_steps += 1
                centering_dists.append(vt.get('centering_dist', 0))
                fractions.append(vt.get('target_fraction', 0))

            if terminated or truncated:
                terminated_early = True
                # Reset preserving same target position
                obs, info = self.env.reset()
                self.env.target_pos = np.array([
                    drone_pos[0] + dx,
                    drone_pos[1] + dy,
                    drone_pos[2] - self.env.hover_height,
                ])
                self.env._update_target_marker_pos()
                self.graphicsEngine.renderFrame()
                self.env._capture_camera_images(force_capture=True)
                state = self.env.base_env.state.astype(np.float32)
                obs = self.env._build_observation(state)

            self.taskMgr.step()

            if step % frame_interval != 0:
                continue

            # Grab images
            fpv_32_rgb = self.env._last_high_freq_image
            if fpv_32_rgb is None:
                continue
            ok_fpv, fpv_hires_rgba = self.fpv_camera.get_image()
            ok_ext, ext_rgba = self.ext_camera.get_image()
            if not ok_fpv or not ok_ext:
                continue

            # Frame 1: raw camera
            raw_bgr = cv2.cvtColor(fpv_hires_rgba, cv2.COLOR_RGBA2BGR)
            raw_resized = cv2.resize(raw_bgr, (PS, PS))
            w1.write(raw_resized)

            # Frame 2: RL input 32x32 upscaled
            rl_bgr = cv2.resize(
                cv2.cvtColor(fpv_32_rgb, cv2.COLOR_RGB2BGR),
                (PS, PS), interpolation=cv2.INTER_NEAREST)
            w2.write(rl_bgr)

            # Frame 3: HSV detection
            det_bgr = self._build_detection_frame(fpv_32_rgb)
            w3.write(det_bgr)

            # Frame 4: external view
            ext_bgr = cv2.cvtColor(ext_rgba, cv2.COLOR_RGBA2BGR)
            ext_resized = cv2.resize(ext_bgr, (PS, PS))
            w4.write(ext_resized)

            # Quad-view with metrics band
            p1 = self._make_panel(raw_resized,
                                  "1. Raw Camera", (0, 255, 255))
            p2 = self._make_panel(rl_bgr,
                                  "2. RL Input (32x32)", (255, 0, 255))
            p3 = self._make_panel(det_bgr,
                                  "3. HSV Detection", (0, 255, 0))
            p4 = self._make_panel(ext_resized,
                                  "4. External View", (0, 255, 255))

            ph = PS + LABEL_H
            sep_v = np.full((ph, SEP, 3), 255, dtype=np.uint8)
            sep_h = np.full((SEP, 2 * PS + SEP, 3), 255, dtype=np.uint8)

            top_row = np.hstack([p1, sep_v, p2])
            bot_row = np.hstack([p3, sep_v, p4])
            quad_frame = np.vstack([top_row, sep_h, bot_row])

            # Metrics band
            band = np.zeros((50, quad_frame.shape[1], 3), dtype=np.uint8)
            quad_frame = np.vstack([quad_frame, band])
            quad_frame = self._add_metrics_overlay(
                quad_frame, tier_name, step, total_steps,
                cumul_reward, vt, tier_cfg)

            w_quad.write(quad_frame)
            frames_written += 1

        for w in writers:
            w.release()

        _m = lambda lst: float(np.mean(lst)) if lst else 0.0
        vis_pct = 100 * visible_steps / max(total_steps, 1)
        summary = {
            'tier': tier_name,
            'cumul_reward': round(cumul_reward, 2),
            'visibility_pct': round(vis_pct, 1),
            'mean_centering': round(_m(centering_dists), 4),
            'mean_fraction': round(_m(fractions), 4),
            'terminated_early': terminated_early,
            'frames': frames_written,
        }

        tag = "EARLY TERM" if terminated_early else "OK"
        print(f"  Result: R={cumul_reward:.0f}  vis={vis_pct:.1f}%  "
              f"cent={_m(centering_dists):.3f}  [{tag}]")
        print(f"  Saved to {tier_dir}/")

        return summary

    # ──────────────────────────────────────────────────────────────────
    def _run_all_tiers(self, task):
        """Record videos for each requested tier sequentially."""
        args = self.args
        tiers_to_run = args.tiers

        print(f"\n{'='*65}")
        print(f"  HOVER-TRACK {self.reward_version.upper()} VIDEO TEST  "
              f"—  Model: {self.ckpt_step}")
        print(f"{'='*65}")
        print(f"  Duration:    {args.duration}s  |  FPS: {args.fps}  "
              f"|  Resolution: {args.panel_size}x{args.panel_size}")
        print(f"  Reward:      {self.reward_version}")
        print(f"  Tiers:       {', '.join(tiers_to_run)}")

        summaries = []
        for tier_name in tiers_to_run:
            cfg = TIERS[tier_name]
            summary = self._record_tier(tier_name, cfg)
            summaries.append(summary)

        # ── Final report ──
        print(f"\n{'='*65}")
        print(f"  RESULTS SUMMARY  —  {self.reward_version.upper()}  "
              f"Model: {self.ckpt_step}")
        print(f"{'='*65}")
        print(f"  {'Tier':<8} | {'Reward':>8} | {'Vis%':>6} | "
              f"{'Center':>7} | {'Frac':>7} | {'Status':<10}")
        print(f"  {'-'*58}")
        for s in summaries:
            tag = "EARLY" if s['terminated_early'] else "OK"
            print(f"  {s['tier']:<8} | "
                  f"{s['cumul_reward']:>8.0f} | "
                  f"{s['visibility_pct']:>5.1f}% | "
                  f"{s['mean_centering']:>7.3f} | "
                  f"{s['mean_fraction']:>7.4f} | "
                  f"{tag:<10}")
        print(f"\n  Output: {self.output_dir}/")
        for tier_name in tiers_to_run:
            print(f"    {tier_name}/quad_view.mp4")
        print(f"{'='*65}\n")

        self.userExit()
        return task.done


if __name__ == "__main__":
    args = parse_args()
    app = HoverTrackV3VideoApp(args)
    app.run()
