#!/usr/bin/env python
"""
Diagnostic test for target sphere colour detection.

Places the neon-green sphere at 3 m in front of the drone and saves
annotated images showing exactly what the HSV filter sees:

  1. raw_fpv.png          – raw FPV image as captured (32×32 upscaled)
  2. hsv_channels.png     – H, S, V channels side-by-side
  3. mask.png             – binary mask from inRange filter
  4. annotated.png        – FPV with detected pixels highlighted, centroid
                            marked, and centre crosshair
  5. colour_check.png     – tries RGBA and BGRA interpretations so you can
                            see which one is correct

Also prints pixel statistics to the terminal.

Usage:
    python tests/test_detection_debug.py
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from panda3d.core import Filename, loadPrcFile
from direct.showbase.ShowBase import ShowBase

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


class DetectionDebugApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # Aerial view in main window
        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -8, 10)
        self.cam.lookAt(0, 0, 5)

        # FPV camera (on the drone)
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Environment — fixed target at 3 m in front of the drone
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
            lemniscate_scale=2.5,
        )

    def run(self):
        out_dir = Path('experiments/detection_debug')
        out_dir.mkdir(parents=True, exist_ok=True)

        # Reset env (places target randomly)
        obs, info = self.env.reset()

        # Override: place the target exactly 3 m in front of the drone
        drone_pos = self.env.base_env.state[0:5:2]
        # Drone faces along +Y in Panda3D by default at start
        self.env.target_pos = np.array([
            drone_pos[0],
            drone_pos[1] + 3.0,
            drone_pos[2],
        ])
        self.env._update_target_marker_pos()

        # Render a few frames so Panda3D buffers settle
        for _ in range(10):
            hover = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            self.env.step(hover)
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()

        # ── Capture raw 4-channel image directly from buffer ──
        tex = self.fpv_camera.buffer.getTexture()
        raw_data = tex.getRamImage()
        raw_4ch = np.frombuffer(raw_data, np.uint8).reshape(
            tex.getYSize(), tex.getXSize(), 4)
        raw_4ch = cv2.flip(raw_4ch, 0)

        # ── Also get the image via the normal pipeline ──
        fpv_32 = self.env._last_high_freq_image  # (32, 32, 3) "RGB"

        print(f"\n{'='*60}")
        print(f"  DETECTION DIAGNOSTIC")
        print(f"{'='*60}")
        print(f"\nDrone pos:   ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})")
        print(f"Target pos:  ({self.env.target_pos[0]:.2f}, "
              f"{self.env.target_pos[1]:.2f}, {self.env.target_pos[2]:.2f})")
        print(f"Distance:    3.00 m")

        # ── 1. Check raw pixel format ──
        print(f"\n--- Raw buffer (centre pixel) ---")
        cy, cx = raw_4ch.shape[0] // 2, raw_4ch.shape[1] // 2
        print(f"  Shape: {raw_4ch.shape}")
        print(f"  Centre pixel [ch0, ch1, ch2, ch3]: {raw_4ch[cy, cx]}")

        # ── 2. Try both RGBA and BGRA interpretations ──
        rgba_rgb = raw_4ch[:, :, :3]                            # assume RGBA → RGB
        bgra_rgb = raw_4ch[:, :, [2, 1, 0]]                    # assume BGRA → RGB

        # Find bright green pixels in both
        for label, img_rgb in [("RGBA→RGB", rgba_rgb), ("BGRA→RGB", bgra_rgb)]:
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
            count = int(np.sum(mask > 0))
            print(f"  {label}: green pixels detected = {count}")

        # ── 3. Pipeline image (32×32) analysis ──
        print(f"\n--- Pipeline image (32×32, used by reward) ---")
        print(f"  Shape: {fpv_32.shape}, dtype: {fpv_32.dtype}")

        # The pipeline: img assumed RGB → BGR → HSV
        img_bgr = cv2.cvtColor(fpv_32, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Current filter: magenta
        mask_magenta = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))
        magenta_count = int(np.sum(mask_magenta > 0))
        total = fpv_32.shape[0] * fpv_32.shape[1]

        # Check all hue ranges to see what IS detected
        mask_red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask_orange = cv2.inRange(hsv, (5, 100, 100), (25, 255, 255))
        mask_yellow = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
        mask_green = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
        mask_cyan = cv2.inRange(hsv, (80, 100, 100), (100, 255, 255))
        mask_blue = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        mask_red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))

        print(f"\n  Pixel counts by colour range (HSV):")
        print(f"    Red (0-10):       {int(np.sum(mask_red1 > 0))}")
        print(f"    Orange (5-25):    {int(np.sum(mask_orange > 0))}")
        print(f"    Yellow (20-35):   {int(np.sum(mask_yellow > 0))}")
        print(f"    Green (35-85):    {int(np.sum(mask_green > 0))}")
        print(f"    Cyan (80-100):    {int(np.sum(mask_cyan > 0))}")
        print(f"    Blue (100-130):   {int(np.sum(mask_blue > 0))}")
        print(f"    MAGENTA (140-170):{magenta_count}  <-- current filter")
        print(f"    Red2 (170-180):   {int(np.sum(mask_red2 > 0))}")
        print(f"    Total pixels:     {total}")
        print(f"    Magenta fraction: {magenta_count/total*100:.1f}%")

        visible = magenta_count > 2
        print(f"\n  Target visible (>2 magenta px): {visible}")

        # ── 4. Save diagnostic images ──
        UP = 16  # upscale factor for visibility

        # 4a. Raw FPV (upscaled)
        fpv_up = cv2.resize(fpv_32, (32 * UP, 32 * UP),
                            interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / 'raw_fpv.png'),
                    cv2.cvtColor(fpv_up, cv2.COLOR_RGB2BGR))

        # 4b. HSV channels
        hsv_up = cv2.resize(hsv, (32 * UP, 32 * UP),
                            interpolation=cv2.INTER_NEAREST)
        h_ch = cv2.applyColorMap(hsv_up[:, :, 0], cv2.COLORMAP_HSV)
        s_ch = cv2.cvtColor(hsv_up[:, :, 1], cv2.COLOR_GRAY2BGR)
        v_ch = cv2.cvtColor(hsv_up[:, :, 2], cv2.COLOR_GRAY2BGR)

        labels = [("H (hue)", h_ch), ("S (sat)", s_ch), ("V (val)", v_ch)]
        for name, ch in labels:
            cv2.putText(ch, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        hsv_combined = np.hstack([h_ch, s_ch, v_ch])
        cv2.imwrite(str(out_dir / 'hsv_channels.png'), hsv_combined)

        # 4c. Magenta mask (upscaled)
        mask_up = cv2.resize(mask_magenta, (32 * UP, 32 * UP),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / 'mask_magenta.png'), mask_up)

        # 4d. Annotated FPV
        annotated = fpv_up.copy()
        # Highlight detected pixels in bright green overlay
        overlay = np.zeros_like(annotated)
        overlay[mask_up > 0] = (0, 255, 0)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.5, 0)

        # Draw image centre crosshair (white)
        img_cx, img_cy = 32 * UP // 2, 32 * UP // 2
        cv2.line(annotated, (img_cx - 20, img_cy), (img_cx + 20, img_cy),
                 (255, 255, 255), 2)
        cv2.line(annotated, (img_cx, img_cy - 20), (img_cx, img_cy + 20),
                 (255, 255, 255), 2)

        if visible:
            ys, xs = np.where(mask_magenta > 0)
            cent_x = float(np.mean(xs)) * UP
            cent_y = float(np.mean(ys)) * UP
            # Red dot for centroid
            cv2.circle(annotated, (int(cent_x), int(cent_y)), 8,
                       (255, 0, 0), -1)
            cv2.circle(annotated, (int(cent_x), int(cent_y)), 8,
                       (255, 255, 255), 2)
            # Bounding box around detected area
            x_min, x_max = int(np.min(xs)) * UP, int(np.max(xs) + 1) * UP
            y_min, y_max = int(np.min(ys)) * UP, int(np.max(ys) + 1) * UP
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max),
                          (0, 255, 255), 2)

            print(f"  Centroid (px):      ({np.mean(xs):.1f}, {np.mean(ys):.1f})")
            print(f"  Bounding box:       x[{np.min(xs)}-{np.max(xs)}] "
                  f"y[{np.min(ys)}-{np.max(ys)}]")

        cv2.putText(annotated, f"Magenta px: {magenta_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        orange_count = int(np.sum(mask_orange > 0))
        cv2.putText(annotated, f"Orange px: {orange_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        status = "DETECTED" if visible else "NOT DETECTED"
        color = (0, 255, 0) if visible else (0, 0, 255)
        cv2.putText(annotated, status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imwrite(str(out_dir / 'annotated.png'),
                    cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        # 4e. Colour format check (RGBA vs BGRA)
        raw_small = cv2.resize(raw_4ch, (32, 32),
                               interpolation=cv2.INTER_AREA)
        rgba_img = cv2.resize(raw_small[:, :, :3], (256, 256),
                              interpolation=cv2.INTER_NEAREST)
        bgra_img = cv2.resize(raw_small[:, :, [2, 1, 0]], (256, 256),
                              interpolation=cv2.INTER_NEAREST)
        cv2.putText(rgba_img, "RGBA (ch 0,1,2)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bgra_img, "BGRA (ch 2,1,0)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        combined = np.hstack([
            cv2.cvtColor(rgba_img, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(bgra_img, cv2.COLOR_RGB2BGR)])
        cv2.imwrite(str(out_dir / 'colour_check.png'), combined)

        print(f"\n--- Output saved to {out_dir}/ ---")
        print(f"  raw_fpv.png       – what the FPV camera captures")
        print(f"  hsv_channels.png  – H, S, V channels")
        print(f"  mask_magenta.png    – binary mask (magenta filter)")
        print(f"  annotated.png     – FPV + detected area + centroid")
        print(f"  colour_check.png  – RGBA vs BGRA interpretation")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    app = DetectionDebugApp()
    app.run()