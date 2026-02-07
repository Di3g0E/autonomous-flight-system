"""
Example: Using the Gymnasium-compatible Quadrotor Environment

This script demonstrates how to use the quadrotor environment with the
standard Gymnasium API for reinforcement learning.
"""

import numpy as np
from environment.quadrotor_env import quad

def random_agent_example():
    """Example 1: Random agent interacting with the environment"""
    
    print("=" * 70)
    print("Example 1: Random Agent")
    print("=" * 70)
    
    # Create environment
    env = quad(t_step=0.01, n=500, euler=0, direct_control=1, T=1)
    
    # Reset environment
    observation, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {observation.shape}")
    print(f"Initial position (x, y, z): {observation[0:6:2]}")
    
    # Run episode with random actions
    episode_reward = 0
    steps = 0
    
    for step in range(100):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"Step {step}: Reward={reward:.4f}, Position={observation[0:6:2]}")
        
        # Check if episode is done
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"\nEpisode ended ({reason}) at step {steps}")
            break
    
    print(f"\nTotal episode reward: {episode_reward:.4f}")
    print(f"Average reward per step: {episode_reward/steps:.4f}")
    
    if info['solved']:
        print("SUCCESS: Target reached!")
    
    env.close()


def hover_controller_example():
    """Example 2: Simple hover controller (PD-like control)"""
    
    print("\n" + "=" * 70)
    print("Example 2: Simple Hover Controller")
    print("=" * 70)
    
    # Create environment
    env = quad(t_step=0.01, n=500, euler=0, direct_control=1, T=1)
    
    # Reset to a specific initial state (near origin)
    det_state = np.zeros(13)
    det_state[6] = 1.0  # quaternion q0 = 1 (identity rotation)
    det_state[0:6:2] = [0.5, 0.5, 1.0]  # small initial displacement
    
    observation, info = env.reset(options={'det_state': det_state})
    print(f"\nInitial position: {observation[0:6:2]}")
    
    # Simple proportional controller gains
    Kp_pos = 0.3
    Kd_pos = 0.2
    
    episode_reward = 0
    steps = 0
    
    for step in range(200):
        # Extract state components
        position = observation[0:6:2]  # x, y, z
        velocity = observation[1:6:2]  # vx, vy, vz
        
        # Simple hover control: try to reach origin with zero velocity
        target_position = np.array([0.0, 0.0, 0.0])
        target_velocity = np.array([0.0, 0.0, 0.0])
        
        # PD control for vertical thrust
        error_pos = target_position - position
        error_vel = target_velocity - velocity
        
        # Generate control action (simplified)
        thrust_correction = Kp_pos * error_pos[2] + Kd_pos * error_vel[2]
        roll_correction = Kp_pos * error_pos[1] + Kd_pos * error_vel[1]
        pitch_correction = Kp_pos * error_pos[0] + Kd_pos * error_vel[0]
        
        # Clip to action space bounds
        action = np.array([
            np.clip(thrust_correction, -1, 1),
            np.clip(roll_correction, -1, 1),
            np.clip(pitch_correction, -1, 1),
            0.0  # yaw control
        ])
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        # Print progress every 40 steps
        if step % 40 == 0:
            dist_to_origin = np.linalg.norm(position)
            print(f"Step {step}: Distance to origin={dist_to_origin:.4f}, Reward={reward:.4f}")
        
        # Check if episode is done
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"\nEpisode ended ({reason}) at step {steps}")
            break
    
    print(f"\nTotal episode reward: {episode_reward:.4f}")
    print(f"Final position: {observation[0:6:2]}")
    print(f"Final distance to origin: {np.linalg.norm(observation[0:6:2]):.4f}")
    
    if info['solved']:
        print("SUCCESS: Hover target reached!")
    
    env.close()


def multiple_episodes_example():
    """Example 3: Running multiple episodes and collecting statistics"""
    
    print("\n" + "=" * 70)
    print("Example 3: Multiple Episodes Statistics")
    print("=" * 70)
    
    # Create environment
    env = quad(t_step=0.01, n=300, euler=0, direct_control=1, T=1)
    
    num_episodes = 5
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        observation, info = env.reset(seed=episode)
        episode_reward = 0
        steps = 0
        
        for step in range(300):
            # Random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if info['solved']:
            success_count += 1
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Length={steps}, Solved={info['solved']}")
    
    print(f"\n--- Statistics over {num_episodes} episodes ---")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    
    env.close()


if __name__ == "__main__":
    # Run all examples
    random_agent_example()
    hover_controller_example()
    multiple_episodes_example()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
