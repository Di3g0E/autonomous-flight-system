#!/usr/bin/env python
"""
Capture reference images from the different cameras used in the project.
The images are saved to data/camera_ref/ for visual reference.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
import cv2

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# IMPORTANT: Import torch BEFORE Panda3D to avoid DLL conflicts on Windows
import torch

# Panda3D imports
from panda3d.core import Filename, loadPrcFile
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

# Load Panda3D config
loadPrcFile(os.path.join(project_root, 'config', 'conf.prc'))

# Project imports
from src.simulation.world_setup import world_setup, quad_setup
from src.vision.img_2_cv import opencv_camera

class CameraTestApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        self.mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)
        ).getFullpath()
        
        # Output directory
        self.output_dir = Path(project_root) / "data" / "camera_ref"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Setup World
        print("Loading 3D world...")
        world_setup(self, self.render, self.mydir)
        quad_setup(self, self.render, self.mydir)
        
        # 2. Setup Cameras
        print("Initializing cameras...")
        
        # FPV Camera (Forward looking, used in RL/Depth training)
        self.fpv_cam = opencv_camera(self, 'fpv_reference', 1)
        self.fpv_cam.cam.reparentTo(self.quad_model)
        self.fpv_cam.cam.setPos(0, 0.3, -0.05)
        self.fpv_cam.cam.setHpr(0, 0, 0)
        self.fpv_cam.buffer.setActive(1)
        
        # Downward Camera 1 (Close-up, used in CV/Calibration)
        self.down_cam_1 = opencv_camera(self, 'downward_close', 1)
        self.down_cam_1.cam.reparentTo(self.quad_model)
        self.down_cam_1.cam.setPos(0, 0, 0.01)
        self.down_cam_1.cam.lookAt(0, 0, 0)
        self.down_cam_1.buffer.setActive(1)
        
        # Downward Camera 2 (1m distance)
        self.down_cam_2 = opencv_camera(self, 'downward_1m', 1)
        self.down_cam_2.cam.reparentTo(self.quad_model)
        self.down_cam_2.cam.setPos(0, 0, 1)
        self.down_cam_2.cam.lookAt(0, 0, 0)
        self.down_cam_2.buffer.setActive(1)
        
        # 3. Schedule Capture Task
        # We wait a few frames to ensure the scene is rendered and textures are loaded
        self.taskMgr.doMethodLater(0.5, self.capture_task, 'capture_task')
        print(f"Capturing reference images to {self.output_dir}...")

    def capture_task(self, task):
        # Force render frame
        self.graphicsEngine.renderFrame()
        
        cameras = [
            (self.fpv_cam, "fpv_forward.png"),
            (self.down_cam_1, "downward_close.png"),
            (self.down_cam_2, "downward_1m.png")
        ]
        
        for cam_obj, filename in cameras:
            success, img = cam_obj.get_image()
            if success and img is not None:
                # img is RGBA, convert to BGR for OpenCV save
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                save_path = self.output_dir / filename
                cv2.imwrite(str(save_path), img_bgr)
                print(f"  ✓ Saved {filename}")
            else:
                print(f"  ✗ Failed to capture {filename}")
                
        print("\nCapture complete. You can now close the window.")
        # Automatically exit after capture if desired, or keep open
        # self.userExit()
        return task.done

if __name__ == "__main__":
    app = CameraTestApp()
    app.run()
