"""
Integration Example: Panda3D Quadrotor with Collision Detection

This script shows how to integrate the collision detection system with
an actual Panda3D application.

This is a complete working example that can be used as a template for
integrating collision detection into your main application.
"""

import sys
import os
from panda3d.core import Filename, loadPrcFile
from direct.showbase.ShowBase import ShowBase

# Load Panda3D configuration
# loadPrcFile('./config/conf.prc')  # Uncomment if you have a config file

from environment.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from models.world_setup import world_setup, quad_setup


class QuadrotorCollisionDemo(ShowBase):
    """
    Demo application showing quadrotor with collision detection.
    """
    
    def __init__(self):
        ShowBase.__init__(self)
        
        # Get current directory
        mydir = os.path.abspath(sys.path[0])
        mydir = Filename.fromOsSpecific(mydir).getFullpath()
        
        # Setup world and quadrotor models
        print("Setting up 3D world...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        
        # Create Panda3D-integrated environment
        print("Creating collision-enabled environment...")
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            t_step=0.01,
            n=3000,
            direct_control=1,
            T=1,
            render_mode='human',
            enable_collisions=True,
            collision_radius=0.3,
            collision_penalty=-100.0
        )
        
        # Add some obstacles to the environment
        print("Adding obstacles...")
        self._setup_obstacles()
        
        # Enable collision visualization for debugging
        self.env.enable_collision_debug()
        
        # Initialize episode
        self.observation, self.info = self.env.reset(seed=42)
        self.episode_reward = 0
        self.step_count = 0
        self.collision_count = 0
        
        # Add task for simulation loop
        self.taskMgr.add(self.simulation_task, 'SimulationTask')
        
        print("\n" + "=" * 70)
        print("Collision Detection Demo Started")
        print("=" * 70)
        print(f"Collision detection: ENABLED")
        print(f"Collision radius: {self.env.collision_radius} m")
        print(f"Collision penalty: {self.env.collision_penalty}")
        print("\nPress ESC to exit")
        print("=" * 70 + "\n")
    
    def _setup_obstacles(self):
        """Setup collision obstacles in the environment."""
        
        # Add some box obstacles (walls)
        self.env.add_box_obstacle(
            position=(3, 0, 2.5),
            size=(0.5, 4, 5),
            name="wall_east"
        )
        
        self.env.add_box_obstacle(
            position=(-3, 0, 2.5),
            size=(0.5, 4, 5),
            name="wall_west"
        )
        
        self.env.add_box_obstacle(
            position=(0, 3, 2.5),
            size=(4, 0.5, 5),
            name="wall_north"
        )
        
        self.env.add_box_obstacle(
            position=(0, -3, 2.5),
            size=(4, 0.5, 5),
            name="wall_south"
        )
        
        # Add some sphere obstacles (pillars)
        self.env.add_sphere_obstacle(
            position=(1.5, 1.5, 2),
            radius=0.4,
            name="pillar_1"
        )
        
        self.env.add_sphere_obstacle(
            position=(-1.5, -1.5, 2),
            radius=0.4,
            name="pillar_2"
        )
        
        # Add collision to the scene model (ground/buildings)
        if hasattr(self, 'scene'):
            self.env.add_model_collision(self.scene, name="scene_collision")
        
        print(f"Added {self.env.obstacle_manager.get_obstacle_count()} obstacles")
    
    def simulation_task(self, task):
        """Main simulation loop."""
        
        # Simple random policy (replace with your RL agent)
        action = self.env.action_space.sample()
        
        # Execute step
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_reward += reward
        self.step_count += 1
        
        # Check for collision
        if info.get('collision_occurred', False):
            self.collision_count += 1
            collision_info = info.get('collision', {})
            
            print(f"\n{'!'*70}")
            print(f"COLLISION DETECTED at step {self.step_count}!")
            print(f"{'!'*70}")
            print(f"Collision object: {collision_info.get('collision_object', 'Unknown')}")
            print(f"Collision point: {collision_info.get('collision_point', 'N/A')}")
            print(f"Distance: {collision_info.get('distance_to_collision', 'N/A'):.3f} m")
            print(f"Reward penalty: {self.env.collision_penalty}")
            print(f"{'!'*70}\n")
        
        # Print progress
        if self.step_count % 50 == 0:
            position = self.observation[0:6:2]
            print(f"Step {self.step_count}: "
                  f"Pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), "
                  f"Reward={reward:.2f}, "
                  f"Collisions={self.collision_count}")
        
        # Check if episode is done
        if terminated or truncated:
            reason = "COLLISION" if info.get('collision_occurred') else \
                     "SUCCESS" if info.get('solved') else \
                     "TIME_LIMIT" if truncated else "BOUNDARY"
            
            print(f"\n{'='*70}")
            print(f"EPISODE ENDED: {reason}")
            print(f"{'='*70}")
            print(f"Total steps: {self.step_count}")
            print(f"Total reward: {self.episode_reward:.2f}")
            print(f"Total collisions: {self.collision_count}")
            print(f"Average reward: {self.episode_reward/self.step_count:.2f}")
            print(f"{'='*70}\n")
            
            # Reset for new episode
            self.observation, self.info = self.env.reset()
            self.episode_reward = 0
            self.step_count = 0
            self.collision_count = 0
        
        return task.cont
    
    def cleanup(self):
        """Cleanup on exit."""
        self.env.close()


# Main entry point
if __name__ == "__main__":
    try:
        app = QuadrotorCollisionDemo()
        app.run()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires:")
        print("  1. Panda3D to be installed")
        print("  2. 3D models in the models/ directory")
        print("  3. Proper configuration in config/conf.prc")
        print("\nFor headless training without Panda3D, use:")
        print("  python example_collision_detection.py")
