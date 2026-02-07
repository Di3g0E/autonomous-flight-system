"""
Example: Using the Panda3D Quadrotor Environment with Collision Detection

This script demonstrates how to use the collision detection system with the
quadrotor environment.
"""

import numpy as np
from environment.quadrotor_env import quad
from environment.panda3d_quadrotor_env import Panda3DQuadrotorEnv


def example_headless_training():
    """
    Example 1: Training without Panda3D (headless mode)
    
    This is the fastest mode for training RL agents, as it doesn't require
    any 3D rendering or collision detection.
    """
    print("=" * 70)
    print("Example 1: Headless Training (No Panda3D)")
    print("=" * 70)
    
    # Create pure physics environment (no Panda3D)
    env = quad(t_step=0.01, n=500, direct_control=1, T=1)
    
    print("\nEnvironment created (headless mode)")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a quick episode
    observation, info = env.reset(seed=42)
    episode_reward = 0
    
    for step in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode completed in {step+1} steps")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Final position: {observation[0:6:2]}")
    
    env.close()


def example_with_panda3d_no_collisions():
    """
    Example 2: Using Panda3D wrapper without collision detection
    
    This mode provides visualization but no collision detection.
    Useful for debugging and visualization during development.
    """
    print("\n" + "=" * 70)
    print("Example 2: Panda3D Wrapper (No Collision Detection)")
    print("=" * 70)
    
    # Create environment without Panda3D components (simulates headless with wrapper)
    env = Panda3DQuadrotorEnv(
        panda3d_app=None,  # No Panda3D app
        quad_model=None,
        render_node=None,
        t_step=0.01,
        n=500,
        direct_control=1,
        enable_collisions=False  # Explicitly disable collisions
    )
    
    print("\nEnvironment created (wrapper mode, no collisions)")
    print(f"Collisions enabled: {env.enable_collisions}")
    
    # Run episode
    observation, info = env.reset(seed=42)
    print(f"Collision info in reset: {info.get('collision', {})}")
    
    episode_reward = 0
    collision_count = 0
    
    for step in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if info.get('collision_occurred', False):
            collision_count += 1
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode completed in {step+1} steps")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Collisions detected: {collision_count}")
    
    env.close()


def example_programmatic_obstacles():
    """
    Example 3: Creating obstacles programmatically
    
    This demonstrates how to add obstacles to the environment without
    actually running Panda3D (for testing the API).
    """
    print("\n" + "=" * 70)
    print("Example 3: Programmatic Obstacle Creation (API Demo)")
    print("=" * 70)
    
    # Create environment
    env = Panda3DQuadrotorEnv(
        panda3d_app=None,
        quad_model=None,
        render_node=None,
        t_step=0.01,
        n=500,
        direct_control=1,
        enable_collisions=False
    )
    
    print("\nDemonstrating obstacle API (without actual Panda3D):")
    
    # These calls will return None without Panda3D, but demonstrate the API
    print("\n1. Adding box obstacle at (2, 2, 1) with size (1, 1, 2)")
    result = env.add_box_obstacle(position=(2, 2, 1), size=(1, 1, 2), name="wall_1")
    print(f"   Result: {result}")
    
    print("\n2. Adding sphere obstacle at (-2, -2, 1) with radius 0.5")
    result = env.add_sphere_obstacle(position=(-2, -2, 1), radius=0.5, name="sphere_1")
    print(f"   Result: {result}")
    
    print("\n3. Clearing all obstacles")
    env.clear_obstacles()
    print("   Obstacles cleared")
    
    print("\nNote: These operations require actual Panda3D components to work.")
    print("See the integration example for full functionality.")
    
    env.close()


def example_collision_configuration():
    """
    Example 4: Configuring collision detection parameters
    """
    print("\n" + "=" * 70)
    print("Example 4: Collision Detection Configuration")
    print("=" * 70)
    
    # Create environment with custom collision settings
    env = Panda3DQuadrotorEnv(
        panda3d_app=None,
        quad_model=None,
        render_node=None,
        t_step=0.01,
        n=500,
        direct_control=1,
        enable_collisions=False,
        collision_radius=0.5,  # Larger collision sphere
        collision_penalty=-200.0  # Larger penalty for collisions
    )
    
    print(f"\nCollision settings:")
    print(f"  Collision radius: {env.collision_radius} m")
    print(f"  Collision penalty: {env.collision_penalty}")
    print(f"  Collisions enabled: {env.enable_collisions}")
    
    # Change collision penalty dynamically
    env.set_collision_penalty(-150.0)
    print(f"\nUpdated collision penalty: {env.collision_penalty}")
    
    env.close()


def example_info_dict_structure():
    """
    Example 5: Understanding the info dictionary structure
    """
    print("\n" + "=" * 70)
    print("Example 5: Info Dictionary Structure")
    print("=" * 70)
    
    env = Panda3DQuadrotorEnv(
        panda3d_app=None,
        quad_model=None,
        render_node=None,
        t_step=0.01,
        n=500,
        direct_control=1,
        enable_collisions=False
    )
    
    observation, info = env.reset(seed=42)
    
    print("\nInfo dict after reset:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Take a step
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print("\nInfo dict after step:")
    for key, value in info.items():
        if key == 'collision':
            print(f"  {key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    env.close()


if __name__ == "__main__":
    # Run all examples
    example_headless_training()
    example_with_panda3d_no_collisions()
    example_programmatic_obstacles()
    example_collision_configuration()
    example_info_dict_structure()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("\nNext steps:")
    print("1. Integrate with actual Panda3D application (see integration example)")
    print("2. Add obstacles to your 3D scene")
    print("3. Train RL agents with collision avoidance")
    print("=" * 70)
