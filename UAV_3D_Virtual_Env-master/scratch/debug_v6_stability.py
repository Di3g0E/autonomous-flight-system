import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = str(Path(os.getcwd())).replace("\\", "/")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from scripts.train_hover_track_v4 import MovingTargetV4Wrapper
from stable_baselines3 import SAC
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData

class StabilityDebugApp(ShowBase):
    def __init__(self, model_path):
        loadPrcFileData('', 'window-type offscreen')
        ShowBase.__init__(self)
        
        # Setup scene
        from src.simulation.world_setup import world_setup, quad_setup
        world_setup(self, self.render, ".")
        quad_setup(self, self.render, ".")
        
        from src.vision.img_2_cv import opencv_camera
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
        
        print(f"Loading model: {model_path}")
        self.model = SAC.load(model_path)

    def run_debug(self):
        print("\n--- STABILITY DEBUG START ---")
        obs, info = self.env.reset()
        
        state = self.env.base_env.state
        print(f"Initial State: z={state[4]:.3f}, roll={self.env.base_env.ang[0]:.3f}, pitch={self.env.base_env.ang[1]:.3f}")
        
        for i in range(30):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = self.env.step(action)
            
            state = self.env.base_env.state
            ang = self.env.base_env.ang
            
            print(f"Step {i+1:02d}: z={state[4]:.3f} | r={ang[0]:.3f} p={ang[1]:.3f} y={ang[2]:.3f} | act={action} | term={term}")
            
            if term or trunc:
                print(f"TERMINATED/TRUNCATED at step {i+1}")
                # Check why
                cond_x = np.concatenate((state[0:6], ang, state[-3:]))
                for idx, (val, limit) in enumerate(zip(np.abs(cond_x), self.env.base_env.bb_cond)):
                    if val >= limit:
                        names = ['x','vx','y','vy','z','vz','roll','pitch','yaw','wx','wy','wz']
                        print(f"  LIMIT EXCEEDED: {names[idx]} = {val:.3f} (limit {limit})")
                break

if __name__ == "__main__":
    model_file = "models/hover_track_v6/checkpoints/model_400000_steps.zip"
    if not os.path.exists(model_file):
        print(f"Error: Model not found at {model_file}")
        sys.exit(1)
        
    app = StabilityDebugApp(model_file)
    app.run_debug()
