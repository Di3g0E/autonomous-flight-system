#!/usr/bin/env python
"""
Static hover video test: SAC model keeps the drone above the target.

Uses the trained hover-track SAC model (centroid_obs=True, 19-D) to
actively stabilise the drone over a magenta sphere while recording four
synchronised videos:

  1. raw_camera.mp4       — imagen cruda que captura la camara del dron
                            (alta resolucion).
  2. rl_input.mp4         — la imagen 32x32 que el dron usa para decidir
                            sus acciones (lo que entra a la red neuronal).
  3. hsv_detection.mp4    — mascara HSV magenta + centroide: la imagen que
                            el dron usa para reconocer el objetivo y calcular
                            su centro en la imagen.
  4. external_view.mp4    — vista exterior donde se ven el dron, la esfera
                            y el entorno 3D.

Output:
    experiments/static_dual_view/1_raw_camera.mp4
    experiments/static_dual_view/2_rl_input.mp4
    experiments/static_dual_view/3_hsv_detection.mp4
    experiments/static_dual_view/4_external_view.mp4

Usage:
    python tests/test_static_hover_video.py
    python tests/test_static_hover_video.py --duration 10 --fps 30
    python tests/test_static_hover_video.py --model-path ./models/hover_track/best_model.zip
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
MODEL_PATH = './models/hover_track/best_model.zip'


def parse_args():
    p = argparse.ArgumentParser(
        description="Record 4 videos of the drone hovering over a target")
    p.add_argument('--model-path', type=str, default=MODEL_PATH,
                   help="Path to trained SAC hover-track model")
    p.add_argument('--duration', type=int, default=DURATION_S,
                   help="Duration in seconds (default: 5)")
    p.add_argument('--fps', type=int, default=FPS,
                   help="Video frame rate (default: 30)")
    p.add_argument('--panel-size', type=int, default=PANEL_SIZE,
                   help="Video resolution (square, default: 480)")
    return p.parse_args()


class StaticHoverVideoApp(ShowBase):
    def __init__(self, model_path, duration_s, fps, panel_size):
        ShowBase.__init__(self)

        self.duration_s = duration_s
        self.fps = fps
        self.panel_size = panel_size

        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        # Output directory
        self.output_dir = Path(project_root) / "experiments" / "static_dual_view"
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

        # ── 4. Create the environment (centroid_obs for 19-D SAC input) ──
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
            init_pos_range=0.2,
            init_vel_range=0.1,
            init_ang_range=0.05,
        )

        # ── 5. Load trained SAC hover-track model ──
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)
        self.model = SAC.load(model_path, env=None)
        print(f"Hover-track model loaded: {model_path}")

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
        """Reset env, place target, record four synchronised videos."""

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

        # Place target directly below the drone
        drone_pos = self.env.base_env.state[0:5:2]
        self.env.target_pos = np.array([
            drone_pos[0],
            drone_pos[1],
            drone_pos[2] - self.env.hover_height,
        ])
        self.env._update_target_marker_pos()

        # Position external camera to frame drone + sphere + surroundings
        drone_viz_z = drone_pos[2] + 5
        target_viz_z = self.env.target_pos[2] + 5
        mid_z = (drone_viz_z + target_viz_z) / 2
        self.ext_camera.cam.setPos(
            drone_pos[0] - 4,
            drone_pos[1] - 4,
            mid_z + 1,
        )
        self.ext_camera.cam.lookAt(
            float(drone_pos[0]),
            float(drone_pos[1]),
            float(mid_z),
        )

        # Warm-up so Panda3D buffers settle
        neutral_action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        for _ in range(15):
            obs, _, _, _, info = self.env.step(neutral_action)
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()

        # ── Recording loop ──
        total_steps = self.duration_s * 100          # dt=0.01 -> 100 steps/s
        frame_interval = max(1, 100 // self.fps)     # steps between frames

        print(f"\nRecording {self.duration_s}s hover test "
              f"({total_steps} steps, {self.fps} fps)...")

        frames_written = 0
        for step in range(total_steps):
            # Model predicts actions from the 19-D centroid observation
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)

            # If the episode ends early, reset keeping the same target
            if terminated or truncated:
                obs, info = self.env.reset()
                self.env.target_pos = np.array([
                    drone_pos[0],
                    drone_pos[1],
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
            p1 = self._make_panel(raw_resized, "1. Raw Camera", (0, 255, 255))
            p2 = self._make_panel(rl_bgr, "2. RL Input (32x32)", (255, 0, 255))
            p3 = self._make_panel(detection_bgr,
                                  "3. HSV Detection + Centroid", (0, 255, 0))
            p4 = self._make_panel(ext_resized, "4. External View", (0, 255, 255))

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
        print(f"\n{'='*60}")
        print(f"  STATIC HOVER VIDEO TEST  (SAC hover-track model)")
        print(f"{'='*60}")
        print(f"  Duration:     {self.duration_s}s")
        print(f"  Frames:       {frames_written}")
        print(f"  FPS:          {self.fps}")
        print(f"  Resolution:   {PS}x{PS}")
        print(f"  Drone pos:    ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, "
              f"{drone_pos[2]:.2f})")
        print(f"  Target pos:   ({self.env.target_pos[0]:.2f}, "
              f"{self.env.target_pos[1]:.2f}, "
              f"{self.env.target_pos[2]:.2f})")
        print(f"  Height diff:  {self.env.hover_height:.3f} m")
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
    app = StaticHoverVideoApp(
        model_path=args.model_path,
        duration_s=args.duration,
        fps=args.fps,
        panel_size=args.panel_size,
    )
    app.run()
