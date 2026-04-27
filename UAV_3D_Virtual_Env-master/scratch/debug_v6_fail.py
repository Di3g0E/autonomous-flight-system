import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = str(Path(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from scripts.train_hover_track_v4 import MovingTargetV4Wrapper
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData

class DebugApp(ShowBase):
    def __init__(self):
        loadPrcFileData('', 'window-type offscreen')
        ShowBase.__init__(self)
        
        # Setup scene like in the env
        from src.simulation.world_setup import world_setup, quad_setup
        world_setup(self, self.render, project_root)
        quad_setup(self, self.render, project_root)
        
        from src.simulation.opencv_camera import opencv_camera
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.1)
        self.fpv_camera.cam.setHpr(0, -90, 0)
        self.fpv_camera.buffer.setActive(1)

        self.env = MovingTargetV4Wrapper(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            use_target=True,
            target_mode='moving',
            target_speed=0.10,
            centroid_obs=True,
            camera_down=True,
            hover_height=1.394,
            constrained_init=True,
            enable_collisions=False,
            reward_version='v3.1'
        )

    def run_debug(self):
        print("\n--- DEBUG START ---")
        obs, info = self.env.reset()
        print(f"Reset done.")
        print(f"Drone Pos: {self.env.base_env.state[0:5:2]}")
        print(f"Target Pos: {self.env.target_pos}")
        print(f"Initial Obs (extras): {obs[-6:]}")
        
        for i in range(20):
            action = np.zeros(4) # Hover action (approx)
            obs, reward, term, trunc, info = self.env.step(action)
            vt = info.get('visual_tracking', {})
            print(f"Step {i+1}: z={self.env.base_env.state[4]:.3f} | vis={vt.get('target_visible')} | pixels={vt.get('target_pixels')} | term={term}")
            if term or trunc:
                print(f"TERMINATED at step {i+1}")
                break

if __name__ == "__main__":
    app = DebugApp()
    app.run_debug()
