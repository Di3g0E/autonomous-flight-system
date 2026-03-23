#!/usr/bin/env python
"""
5-second video diagnostic of the magenta sphere detection.

The drone hovers while the magenta sphere sits at 3 m in front.
Records a side-by-side video:
  LEFT  – raw FPV (upscaled)
  RIGHT – annotated FPV with HSV mask overlay, centroid, bounding box,
          and live pixel counts

Usage:
    python tests/test_detection_video.py
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


DURATION_S = 5
FPS = 30
PANEL_W, PANEL_H = 512, 512


class DetectionVideoApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # Aerial main window
        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -8, 10)
        self.cam.lookAt(0, 0, 5)

        # FPV camera
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Environment — fixed target
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
            n=DURATION_S * 100 + 50,
            t_step=0.01,
            direct_control=1,
            target_radius=0.25,
            filming_mode=True,
            lemniscate_scale=2.5,
        )

    def run(self):
        out_dir = Path('experiments/detection_debug')
        out_dir.mkdir(parents=True, exist_ok=True)

        obs, info = self.env.reset()

        # Place target 3 m in front of the drone
        drone_pos = self.env.base_env.state[0:5:2]
        self.env.target_pos = np.array([
            drone_pos[0], drone_pos[1] + 3.0, drone_pos[2]])
        self.env._update_target_marker_pos()

        # Warm-up
        for _ in range(10):
            self.env.step(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()

        # Video writer
        video_path = str(out_dir / 'detection_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, FPS,
                                 (PANEL_W * 2, PANEL_H))

        total_steps = DURATION_S * 100  # dt=0.01 → 100 steps/s
        frame_interval = 100 // FPS     # steps between video frames

        print(f"\nRecording {DURATION_S}s detection video ({total_steps} steps)...")

        for step in range(total_steps):
            hover = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = self.env.step(hover)
            if terminated or truncated:
                obs, info = self.env.reset()
                self.env.target_pos = np.array([
                    drone_pos[0], drone_pos[1] + 3.0, drone_pos[2]])
                self.env._update_target_marker_pos()

            self.taskMgr.step()

            # Only write a frame every frame_interval steps
            if step % frame_interval != 0:
                continue

            fpv_32 = self.env._last_high_freq_image
            if fpv_32 is None:
                continue

            # ── HSV detection (same pipeline as reward function) ──
            img_bgr = cv2.cvtColor(fpv_32, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))
            mag_count = int(np.sum(mask > 0))
            total_px = 32 * 32
            visible = mag_count > 2

            # ── Left panel: raw FPV ──
            left = cv2.resize(fpv_32, (PANEL_W, PANEL_H),
                              interpolation=cv2.INTER_NEAREST)
            cv2.putText(left, "RAW FPV (32x32)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ── Right panel: annotated ──
            right = cv2.resize(fpv_32, (PANEL_W, PANEL_H),
                               interpolation=cv2.INTER_NEAREST)
            mask_up = cv2.resize(mask, (PANEL_W, PANEL_H),
                                 interpolation=cv2.INTER_NEAREST)

            # Magenta overlay on detected pixels
            overlay = np.zeros_like(right)
            overlay[mask_up > 0] = (255, 0, 255)
            right = cv2.addWeighted(right, 0.6, overlay, 0.5, 0)

            # Centre crosshair (white)
            cx, cy = PANEL_W // 2, PANEL_H // 2
            cv2.line(right, (cx - 20, cy), (cx + 20, cy),
                     (255, 255, 255), 2)
            cv2.line(right, (cx, cy - 20), (cx, cy + 20),
                     (255, 255, 255), 2)

            if visible:
                ys, xs = np.where(mask > 0)
                cent_x = float(np.mean(xs)) / 32 * PANEL_W
                cent_y = float(np.mean(ys)) / 32 * PANEL_H
                # Centroid (red dot)
                cv2.circle(right, (int(cent_x), int(cent_y)), 8,
                           (255, 0, 0), -1)
                cv2.circle(right, (int(cent_x), int(cent_y)), 8,
                           (255, 255, 255), 2)
                # Bounding box (cyan)
                scale = PANEL_W / 32
                x1 = int(np.min(xs) * scale)
                x2 = int((np.max(xs) + 1) * scale)
                y1 = int(np.min(ys) * scale)
                y2 = int((np.max(ys) + 1) * scale)
                cv2.rectangle(right, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Text overlay
            cv2.putText(right, "HSV DETECTION", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            status = "DETECTED" if visible else "NOT DETECTED"
            s_color = (0, 255, 0) if visible else (0, 0, 255)
            cv2.putText(right, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)
            cv2.putText(right, f"Magenta px: {mag_count} ({mag_count/total_px*100:.1f}%)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 255), 2)

            target_info = info.get('target', {})
            dist = target_info.get('distance_to_target', 0)
            cv2.putText(right, f"Dist: {dist:.2f}m", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            vt = info.get('visual_tracking', {})
            cr = vt.get('centering_reward', 0)
            sr = vt.get('scale_reward', 0)
            cv2.putText(right, f"Center: {cr:.1f}/3  Scale: {sr:.1f}/3",
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

            # Combine and write
            frame = np.hstack([left, right])
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"\nDone! Video saved to {video_path}")


if __name__ == "__main__":
    app = DetectionVideoApp()
    app.run()