#!/usr/bin/env python
"""
Hover-track v2 stabilisation video test.

Evaluates the SAC v2 model (trained with curriculum + offset targets)
under challenging post-spiral conditions and records four synchronised
videos plus a 2×2 quad-view:

  1. raw_camera.mp4       — imagen cruda de alta resolución que captura
                            la cámara del dron (apuntando hacia abajo).
  2. rl_input.mp4         — la imagen 32×32 que el dron usa para decidir
                            sus acciones (entrada a la red neuronal).
  3. hsv_detection.mp4    — máscara HSV magenta + centroide: la imagen
                            que el dron usa para reconocer el objetivo y
                            calcular su centro en la imagen.
  4. external_view.mp4    — vista exterior donde se ven el dron, la
                            esfera y el entorno 3D.
  5. quad_view.mp4        — cuadrícula 2×2 con los cuatro vídeos.

Output:
    experiments/hover_track_v2/

Usage:
    python tests/test_hover_track_v2_video.py
    python tests/test_hover_track_v2_video.py --duration 10 --fps 30
    python tests/test_hover_track_v2_video.py --target-offset 0.8
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Project root
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import torch BEFORE Panda3D to avoid DLL conflicts on Windows
import torch  # noqa: F401

# Panda3D imports
from panda3d.core import Filename, loadPrcFile
from direct.showbase.ShowBase import ShowBase

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from stable_baselines3 import SAC

from src.simulation.world_setup import world_setup, quad_setup
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


# ── Defaults ──────────────────────────────────────────────────────────
DURATION_S = 5          # seconds of simulation
FPS = 30                # output video frame-rate
PANEL_SIZE = 480        # pixel side for each video
LABEL_H = 40            # height reserved for the title bar
SEP = 3                 # separator thickness (white lines)
MODEL_PATH = './models/hover_track_v2/best_model.zip'
TARGET_OFFSET = 0.6     # metres — offset the target from the drone


def parse_args():
    p = argparse.ArgumentParser(
        description="Record 4 videos of hover-track v2 stabilisation test")
    p.add_argument('--model-path', type=str, default=MODEL_PATH,
                   help="Path to trained SAC hover-track v2 model")
    p.add_argument('--duration', type=int, default=DURATION_S,
                   help="Duration in seconds (default: 5)")
    p.add_argument('--fps', type=int, default=FPS,
                   help="Video frame rate (default: 30)")
    p.add_argument('--panel-size', type=int, default=PANEL_SIZE,
                   help="Video resolution (square, default: 480)")
    p.add_argument('--target-offset', type=float, default=TARGET_OFFSET,
                   help="Target XY offset in metres (default: 0.6)")
    p.add_argument('--init-vel', type=float, default=0.30,
                   help="Initial lateral velocity (default: 0.30 m/s)")
    p.add_argument('--init-ang', type=float, default=0.12,
                   help="Initial tilt in radians (default: 0.12)")
    return p.parse_args()


class HoverTrackV2VideoApp(ShowBase):
    def __init__(self, model_path, duration_s, fps, panel_size,
                 target_offset, init_vel, init_ang):
        ShowBase.__init__(self)

        self.duration_s = duration_s
        self.fps = fps
        self.panel_size = panel_size
        self.target_offset = target_offset
        self.init_vel = init_vel
        self.init_ang = init_ang

        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        # Output directory
        self.output_dir = Path(project_root) / "experiments" / "hover_track_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Load 3D world and drone model ──
        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)

        # ── 2. Downward camera (attached to the drone, pointing down) ──
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, 0.01)
        self.fpv_camera.cam.lookAt(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # ── 3. External camera (third-person view) ──
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.buffer.setActive(1)

        # ── 4. Create environment (centroid_obs for 19-D SAC input) ──
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
            n=duration_s * 100 + 50,
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
            init_vel_range=init_vel,
            init_ang_range=init_ang,
        )

        # ── 5. Load trained SAC hover-track v2 model ──
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)
        self.model = SAC.load(model_path, env=None)
        print(f"Hover-track v2 model loaded: {model_path}")

        # Schedule recording after the scene has settled
        self.taskMgr.doMethodLater(0.5, self._record_videos, 'record')

    # ──────────────────────────────────────────────────────────────────
    def _make_panel(self, img_bgr, title, title_color=(0, 255, 255)):
        """Resize to panel_size and add a title bar on top."""
        PS = self.panel_size
        resized = cv2.resize(img_bgr, (PS, PS))
        panel = np.zeros((PS + LABEL_H, PS, 3), dtype=np.uint8)
        panel[LABEL_H:, :] = resized
        cv2.putText(panel, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, title_color, 2)
        return panel

    # ──────────────────────────────────────────────────────────────────
    def _build_detection_frame(self, img_32_rgb):
        """
        Reproduce the HSV magenta detection pipeline and return an
        annotated BGR frame at panel_size resolution.
        """
        h, w = img_32_rgb.shape[:2]
        UP = self.panel_size // w

        # Same conversion the env uses: RGB -> BGR -> HSV
        img_bgr = cv2.cvtColor(img_32_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))

        pixel_count = int(np.sum(mask > 0))
        total_pixels = h * w
        frac = pixel_count / total_pixels

        # Upscale for visibility
        annotated = cv2.resize(img_bgr, (self.panel_size, self.panel_size),
                               interpolation=cv2.INTER_NEAREST)
        mask_up = cv2.resize(mask, (self.panel_size, self.panel_size),
                             interpolation=cv2.INTER_NEAREST)

        # Green overlay on detected magenta pixels
        overlay = np.zeros_like(annotated)
        overlay[mask_up > 0] = (0, 255, 0)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.5, 0)

        # Centre crosshair (white)
        cx_img, cy_img = self.panel_size // 2, self.panel_size // 2
        cv2.line(annotated, (cx_img - 20, cy_img), (cx_img + 20, cy_img),
                 (255, 255, 255), 1)
        cv2.line(annotated, (cx_img, cy_img - 20), (cx_img, cy_img + 20),
                 (255, 255, 255), 1)

        visible = pixel_count > 2
        if visible:
            ys, xs = np.where(mask > 0)
            cent_x = float(np.mean(xs)) * UP
            cent_y = float(np.mean(ys)) * UP

            # Red dot at centroid
            cv2.circle(annotated, (int(cent_x), int(cent_y)), 8,
                       (0, 0, 255), -1)
            cv2.circle(annotated, (int(cent_x), int(cent_y)), 8,
                       (255, 255, 255), 2)

            # Bounding box (cyan)
            x_min, x_max = int(np.min(xs)) * UP, int(np.max(xs) + 1) * UP
            y_min, y_max = int(np.min(ys)) * UP, int(np.max(ys) + 1) * UP
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max),
                          (0, 255, 255), 2)

            # Normalised centroid text
            cx_norm = (np.mean(xs) - w / 2) / (w / 2)
            cy_norm = (np.mean(ys) - h / 2) / (h / 2)
            info_text = f"cx={cx_norm:+.2f} cy={cy_norm:+.2f} frac={frac:.3f}"
        else:
            info_text = "NOT DETECTED"

        # Text overlay at the bottom
        cv2.putText(annotated, info_text, (8, self.panel_size - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        return annotated

    # ──────────────────────────────────────────────────────────────────
    def _record_videos(self, task):
        """Reset env with offset target, record four synchronised videos."""

        PS = self.panel_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        w1 = cv2.VideoWriter(str(self.output_dir / '1_raw_camera.mp4'),
                             fourcc, self.fps, (PS, PS))
        w2 = cv2.VideoWriter(str(self.output_dir / '2_rl_input.mp4'),
                             fourcc, self.fps, (PS, PS))
        w3 = cv2.VideoWriter(str(self.output_dir / '3_hsv_detection.mp4'),
                             fourcc, self.fps, (PS, PS))
        w4 = cv2.VideoWriter(str(self.output_dir / '4_external_view.mp4'),
                             fourcc, self.fps, (PS, PS))

        # Quad-view: 2x2 grid with title bars and separators
        quad_w = 2 * PS + SEP
        quad_h = 2 * (PS + LABEL_H) + SEP
        w_quad = cv2.VideoWriter(str(self.output_dir / 'quad_view.mp4'),
                                 fourcc, self.fps, (quad_w, quad_h))

        writers = [w1, w2, w3, w4, w_quad]

        # ── Reset environment ──
        obs, info = self.env.reset()

        # Apply target offset (simulating a target that is not centred)
        drone_pos = self.env.base_env.state[0:5:2]
        angle = np.random.uniform(0, 2 * np.pi)
        dx = self.target_offset * np.cos(angle)
        dy = self.target_offset * np.sin(angle)
        self.env.target_pos = np.array([
            drone_pos[0] + dx,
            drone_pos[1] + dy,
            drone_pos[2] - self.env.hover_height,
        ])
        self.env._update_target_marker_pos()

        print(f"  Target offset: ({dx:+.2f}, {dy:+.2f}) m  "
              f"(total: {self.target_offset:.2f} m)")

        # Position external camera to frame drone + target + surroundings
        mx = (drone_pos[0] + self.env.target_pos[0]) / 2
        my = (drone_pos[1] + self.env.target_pos[1]) / 2
        drone_viz_z = drone_pos[2] + 5
        target_viz_z = self.env.target_pos[2] + 5
        mid_z = (drone_viz_z + target_viz_z) / 2
        cam_dist = max(self.target_offset * 3, 4.0)
        self.ext_camera.cam.setPos(
            mx - cam_dist * 0.6,
            my - cam_dist * 0.6,
            mid_z + cam_dist * 0.4,
        )
        self.ext_camera.cam.lookAt(float(mx), float(my), float(mid_z - 0.5))

        # Warm-up so Panda3D buffers settle
        neutral_action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        for _ in range(15):
            obs, _, _, _, info = self.env.step(neutral_action)
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()

        # ── Recording loop ──
        total_steps = self.duration_s * 100          # dt=0.01 -> 100 steps/s
        frame_interval = max(1, 100 // self.fps)     # steps between frames

        print(f"\nRecording {self.duration_s}s hover-track v2 test "
              f"({total_steps} steps, {self.fps} fps)...")

        frames_written = 0
        cumul_reward = 0.0
        visible_steps = 0

        for step in range(total_steps):
            # Model predicts actions from the 19-D centroid observation
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            cumul_reward += reward

            vt = info.get('visual_tracking', {})
            if vt.get('target_visible', False):
                visible_steps += 1

            # If the episode ends early, reset keeping the same target
            if terminated or truncated:
                obs, info = self.env.reset()
                self.env.target_pos = np.array([
                    drone_pos[0] + dx,
                    drone_pos[1] + dy,
                    drone_pos[2] - self.env.hover_height,
                ])
                self.env._update_target_marker_pos()

            self.taskMgr.step()

            # Only write a frame every frame_interval steps
            if step % frame_interval != 0:
                continue

            # Grab images
            fpv_32_rgb = self.env._last_high_freq_image
            if fpv_32_rgb is None:
                continue

            success_fpv, fpv_hires_rgba = self.fpv_camera.get_image()
            success_ext, ext_rgba = self.ext_camera.get_image()
            if not success_fpv or not success_ext:
                continue

            # ── Frame 1: raw camera (high-res) ──
            raw_bgr = cv2.cvtColor(fpv_hires_rgba, cv2.COLOR_RGBA2BGR)
            raw_resized = cv2.resize(raw_bgr, (PS, PS))
            w1.write(raw_resized)

            # ── Frame 2: RL input 32x32 (nearest-neighbour upscale) ──
            rl_bgr = cv2.resize(
                cv2.cvtColor(fpv_32_rgb, cv2.COLOR_RGB2BGR),
                (PS, PS),
                interpolation=cv2.INTER_NEAREST,
            )
            w2.write(rl_bgr)

            # ── Frame 3: HSV detection + centroid ──
            detection_bgr = self._build_detection_frame(fpv_32_rgb)
            w3.write(detection_bgr)

            # ── Frame 4: external view ──
            ext_bgr = cv2.cvtColor(ext_rgba, cv2.COLOR_RGBA2BGR)
            ext_resized = cv2.resize(ext_bgr, (PS, PS))
            w4.write(ext_resized)

            # ── Quad-view: 2x2 grid with titles ──
            p1 = self._make_panel(raw_resized,
                                  "1. Raw Camera", (0, 255, 255))
            p2 = self._make_panel(rl_bgr,
                                  "2. RL Input (32x32)", (255, 0, 255))
            p3 = self._make_panel(detection_bgr,
                                  "3. HSV Detection + Centroid", (0, 255, 0))
            p4 = self._make_panel(ext_resized,
                                  "4. External View", (0, 255, 255))

            ph = PS + LABEL_H  # panel height with title bar
            sep_v = np.full((ph, SEP, 3), 255, dtype=np.uint8)
            sep_h = np.full((SEP, 2 * PS + SEP, 3), 255, dtype=np.uint8)

            top_row = np.hstack([p1, sep_v, p2])
            bot_row = np.hstack([p3, sep_v, p4])
            quad_frame = np.vstack([top_row, sep_h, bot_row])
            w_quad.write(quad_frame)

            frames_written += 1

        # ── Release ──
        for w in writers:
            w.release()

        # ── Console report ──
        vis_pct = 100 * visible_steps / max(total_steps, 1)
        print(f"\n{'='*60}")
        print(f"  HOVER-TRACK v2 STABILISATION TEST")
        print(f"{'='*60}")
        print(f"  Duration:        {self.duration_s}s")
        print(f"  Frames:          {frames_written}")
        print(f"  FPS:             {self.fps}")
        print(f"  Resolution:      {PS}x{PS}")
        print(f"  Target offset:   {self.target_offset:.2f} m")
        print(f"  Init velocity:   {self.init_vel:.2f} m/s")
        print(f"  Init tilt:       {self.init_ang:.2f} rad")
        print(f"  Cumul. reward:   {cumul_reward:.2f}")
        print(f"  Visibility:      {vis_pct:.1f}%")
        print(f"  Drone pos:       ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, "
              f"{drone_pos[2]:.2f})")
        print(f"  Target pos:      ({self.env.target_pos[0]:.2f}, "
              f"{self.env.target_pos[1]:.2f}, "
              f"{self.env.target_pos[2]:.2f})")
        print(f"\n  Saved to {self.output_dir}/")
        print(f"    quad_view.mp4         - 2x2 combined view")
        print(f"    1_raw_camera.mp4      - High-res downward camera")
        print(f"    2_rl_input.mp4        - 32x32 RL observation")
        print(f"    3_hsv_detection.mp4   - Magenta mask + centroid")
        print(f"    4_external_view.mp4   - Third-person view")
        print(f"{'='*60}\n")

        self.userExit()
        return task.done


if __name__ == "__main__":
    args = parse_args()
    app = HoverTrackV2VideoApp(
        model_path=args.model_path,
        duration_s=args.duration,
        fps=args.fps,
        panel_size=args.panel_size,
        target_offset=args.target_offset,
        init_vel=args.init_vel,
        init_ang=args.init_ang,
    )
    app.run()
