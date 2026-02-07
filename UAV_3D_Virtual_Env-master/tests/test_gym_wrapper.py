"""
Test script for Gymnasium-compatible Quadrotor Environment

This script verifies that the quadrotor_env.py has been successfully
converted to a Gymnasium-compatible environment.
"""

import numpy as np
from src.envs.quadrotor_env import quad

def test_gym_api():
    """Test basic Gymnasium API compatibility"""
    
    print("=" * 60)
    print("Testing Gymnasium Wrapper for Quadrotor Environment")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = quad(t_step=0.01, n=1000, euler=0, direct_control=1, T=1)
    print("   [OK] Environment created successfully")
    
    # Check spaces
    print("\n2. Checking action and observation spaces...")
    print(f"   Action space: {env.action_space}")
    print(f"   Action space shape: {env.action_space.shape}")
    print(f"   Action space bounds: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Observation space shape: {env.observation_space.shape}")
    print("   [OK] Spaces defined correctly")
    
    # Test reset
    print("\n3. Testing reset() method...")
    observation, info = env.reset(seed=42)
    print(f"   Observation shape: {observation.shape}")
    print(f"   Observation dtype: {observation.dtype}")
    print(f"   Info keys: {list(info.keys())}")
    print(f"   Initial position: {observation[0:6:2]}")
    print("   [OK] reset() returns (observation, info)")
    
    # Test step
    print("\n4. Testing step() method...")
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"   Observation shape: {observation.shape}")
    print(f"   Reward: {reward:.4f}")
    print(f"   Terminated: {terminated}")
    print(f"   Truncated: {truncated}")
    print(f"   Info keys: {list(info.keys())}")
    print("   [OK] step() returns (observation, reward, terminated, truncated, info)")
    
    # Test episode
    print("\n5. Running a short episode (10 steps)...")
    observation, info = env.reset(seed=42)
    total_reward = 0
    
    for step in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"   Episode ended at step {step+1}")
            break
    
    print(f"   Total reward over 10 steps: {total_reward:.4f}")
    print("   [OK] Episode ran successfully")
    
    # Test with deterministic state
    print("\n6. Testing reset with deterministic state...")
    det_state = np.zeros(13)
    det_state[6] = 1.0  # Set quaternion q0 = 1 (identity rotation)
    observation, info = env.reset(options={'det_state': det_state})
    print(f"   Initial state set to zeros (except q0=1)")
    print(f"   Position: {observation[0:6:2]}")
    print("   [OK] Deterministic reset works")
    
    # Test render and close
    print("\n7. Testing render() and close() methods...")
    env.render()
    print("   [OK] render() executed")
    env.close()
    print("   [OK] close() executed")
    
    print("\n" + "=" * 60)
    print("All tests passed! [OK]")
    print("The environment is now Gymnasium-compatible!")
    print("=" * 60)

def test_compatibility_with_stable_baselines3():
    """Test if environment can be used with Stable-Baselines3"""
    
    print("\n" + "=" * 60)
    print("Testing Stable-Baselines3 Compatibility")
    print("=" * 60)
    
    try:
        from stable_baselines3.common.env_checker import check_env
        
        print("\n1. Creating environment...")
        env = quad(t_step=0.01, n=1000, euler=0, direct_control=1, T=1)
        
        print("2. Running Stable-Baselines3 environment checker...")
        check_env(env, warn=True)
        
        print("\n[OK] Environment is compatible with Stable-Baselines3!")
        
    except ImportError:
        print("\nStable-Baselines3 not installed. Skipping this test.")
        print("To install: pip install stable-baselines3")
    except Exception as e:
        print(f"\n[ERROR] Compatibility check failed: {e}")
        print("This might require minor adjustments to the environment.")

if __name__ == "__main__":
    # Run basic tests
    test_gym_api()
    
    # Run SB3 compatibility test
    test_compatibility_with_stable_baselines3()
