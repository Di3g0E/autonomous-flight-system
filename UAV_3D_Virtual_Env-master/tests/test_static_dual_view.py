#!/usr/bin/env python
"""
Static quad-panel test: full vision pipeline of the drone.

Generates a static scene with the drone hovering and a magenta sphere
directly below it. Saves a single image with four panels:

  1. RAW CAMERA      — imagen cruda que captura la camara del dron (alta res).
  2. RL INPUT (32x32) — la imagen 32x32 que el dron usa para decidir sus
                         acciones (lo que realmente entra a la red neuronal).
  3. HSV DETECTION    — mascara HSV magenta + centroide: la imagen que el
                         dron usa para reconocer el objetivo y calcular su
                         centro en la imagen.
  4. EXTERNAL VIEW    — vista exterior donde se ven el dron, la esfera y
                         el entorno 3D.

Output:
    experiments/static_dual_view/quad_view.png   (4 paneles combinados)
    experiments/static_dual_view/1_raw_camera.png
    experiments/static_dual_view/2_rl_input_32x32.png
    experiments/static_dual_view/3_hsv_detection.png
    experiments/static_dual_view/4_external_view.png

Usage:
    python tests/test_static_dual_view.py
"""

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

from src.simulation.world_setup import world_setup, quad_setup
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


PANEL_SIZE = 480   # each panel will be PANEL_SIZE x PANEL_SIZE
LABEL_H = 40       # height reserved for the title bar


class StaticDualViewApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

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
        self.fpv_camera.cam.lookAt(0, 0, 0)   # looks straight down
        self.fpv_camera.buffer.setActive(1)

        # ── 3. External camera (third-person view) ──
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.buffer.setActive(1)

        # ── 4. Create the environment with target below the drone ──
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
            enable_collisions=False,
            n=100,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            filming_mode=True,
            camera_down=True,
            hover_height=1.394,
        )

        # Schedule capture after the scene has settled
        self.taskMgr.doMethodLater(0.5, self._capture_views, 'capture')

    # ─────────────────────────────────────────────────────────────────
    def _make_panel(self, img_bgr, title, title_color=(0, 255, 255)):
        """Resize image to PANEL_SIZE and add a title bar on top."""
        resized = cv2.resize(img_bgr, (PANEL_SIZE, PANEL_SIZE))
        panel = np.zeros((PANEL_SIZE + LABEL_H, PANEL_SIZE, 3), dtype=np.uint8)
        panel[LABEL_H:, :] = resized
        cv2.putText(panel, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, title_color, 2)
        return panel

    # ─────────────────────────────────────────────────────────────────
    def _build_detection_image(self, img_32_rgb):
        """
        Reproduce the exact HSV magenta detection pipeline used by the
        environment and draw the result on an annotated image.
        """
        h, w = img_32_rgb.shape[:2]

        # Same conversion the env uses: RGB -> BGR -> HSV
        img_bgr = cv2.cvtColor(img_32_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))

        pixel_count = int(np.sum(mask > 0))
        total_pixels = h * w
        frac = pixel_count / total_pixels

        # Upscale for visibility
        UP = PANEL_SIZE // w  # 480 / 32 = 15
        annotated = cv2.resize(img_bgr, (PANEL_SIZE, PANEL_SIZE),
                               interpolation=cv2.INTER_NEAREST)
        mask_up = cv2.resize(mask, (PANEL_SIZE, PANEL_SIZE),
                             interpolation=cv2.INTER_NEAREST)

        # Green overlay on detected magenta pixels
        overlay = np.zeros_like(annotated)
        overlay[mask_up > 0] = (0, 255, 0)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.5, 0)

        # Centre crosshair (white)
        cx_img, cy_img = PANEL_SIZE // 2, PANEL_SIZE // 2
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
            # Bounding box
            x_min, x_max = int(np.min(xs)) * UP, int(np.max(xs) + 1) * UP
            y_min, y_max = int(np.min(ys)) * UP, int(np.max(ys) + 1) * UP
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max),
                          (0, 255, 255), 2)
            # Normalised centroid (what the RL agent receives)
            cx_norm = (np.mean(xs) - w / 2) / (w / 2)
            cy_norm = (np.mean(ys) - h / 2) / (h / 2)
            info_text = f"cx={cx_norm:+.2f} cy={cy_norm:+.2f} frac={frac:.3f}"
        else:
            info_text = "NOT DETECTED"

        # Text overlay at the bottom
        cv2.putText(annotated, info_text, (8, PANEL_SIZE - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        return annotated, pixel_count, visible

    # ─────────────────────────────────────────────────────────────────
    def _capture_views(self, task):
        """Reset env, place target, render, capture all four views."""

        # Reset environment (spawns the target sphere)
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

        # Run hover steps so Panda3D buffers settle
        hover_action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        for _ in range(15):
            self.env.step(hover_action)
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()
        self.graphicsEngine.renderFrame()

        # ── Capture images ──
        fpv_32_rgb = self.env._last_high_freq_image          # (32,32,3) RGB
        success_fpv, fpv_hires_rgba = self.fpv_camera.get_image()
        success_ext, ext_rgba = self.ext_camera.get_image()

        if not success_fpv or not success_ext:
            print("ERROR: Failed to capture one or both cameras.")
            self.userExit()
            return task.done

        # ─── Panel 1: Raw camera image (high resolution) ───
        raw_bgr = cv2.cvtColor(fpv_hires_rgba, cv2.COLOR_RGBA2BGR)
        panel_1 = self._make_panel(raw_bgr,
                                   "1. Raw Camera",
                                   (0, 255, 255))

        # ─── Panel 2: RL input 32x32 (nearest-neighbour upscale) ───
        rl_bgr = cv2.resize(
            cv2.cvtColor(fpv_32_rgb, cv2.COLOR_RGB2BGR),
            (PANEL_SIZE, PANEL_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        panel_2 = self._make_panel(rl_bgr,
                                   "2. RL Input (32x32)",
                                   (255, 0, 255))

        # ─── Panel 3: HSV detection + centroid ───
        detection_bgr, px_count, visible = self._build_detection_image(fpv_32_rgb)
        panel_3 = self._make_panel(detection_bgr,
                                   "3. HSV Detection + Centroid",
                                   (0, 255, 0))

        # ─── Panel 4: External view ───
        ext_bgr = cv2.cvtColor(ext_rgba, cv2.COLOR_RGBA2BGR)
        panel_4 = self._make_panel(ext_bgr,
                                   "4. External View",
                                   (0, 255, 255))

        # ─── Combine 4 panels in a 2x2 grid ───
        sep_v = np.full((PANEL_SIZE + LABEL_H, 3, 3), 255, dtype=np.uint8)
        sep_h = np.full((3, 2 * PANEL_SIZE + 3, 3), 255, dtype=np.uint8)

        top_row = np.hstack([panel_1, sep_v, panel_2])
        bot_row = np.hstack([panel_3, sep_v, panel_4])
        combined = np.vstack([top_row, sep_h, bot_row])

        # ─── Save ───
        cv2.imwrite(str(self.output_dir / "quad_view.png"), combined)
        cv2.imwrite(str(self.output_dir / "1_raw_camera.png"), raw_bgr)
        cv2.imwrite(str(self.output_dir / "2_rl_input_32x32.png"),
                    cv2.cvtColor(fpv_32_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.output_dir / "3_hsv_detection.png"), detection_bgr)
        cv2.imwrite(str(self.output_dir / "4_external_view.png"), ext_bgr)

        # ─── Console report ───
        print(f"\n{'='*60}")
        print(f"  STATIC QUAD-VIEW TEST")
        print(f"{'='*60}")
        print(f"  Drone pos:    ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, "
              f"{drone_pos[2]:.2f})")
        print(f"  Target pos:   ({self.env.target_pos[0]:.2f}, "
              f"{self.env.target_pos[1]:.2f}, "
              f"{self.env.target_pos[2]:.2f})")
        print(f"  Height diff:  {self.env.hover_height:.3f} m")
        print(f"  Magenta px:   {px_count} / {32*32}  "
              f"({px_count/(32*32)*100:.1f}%)")
        print(f"  Detected:     {'YES' if visible else 'NO'}")
        print(f"\n  Saved to {self.output_dir}/")
        print(f"    quad_view.png         - 2x2 combined view")
        print(f"    1_raw_camera.png      - High-res downward camera")
        print(f"    2_rl_input_32x32.png  - 32x32 RL observation")
        print(f"    3_hsv_detection.png   - Magenta mask + centroid")
        print(f"    4_external_view.png   - Third-person view")
        print(f"{'='*60}\n")

        self.userExit()
        return task.done


if __name__ == "__main__":
    app = StaticDualViewApp()
    app.run()
