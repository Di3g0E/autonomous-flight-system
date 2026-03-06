"""
Basic functionality test for camera-enabled environment.

This test verifies that the dual-camera system integrates correctly with the
Gymnasium environment and returns properly formatted observations.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


def test_backward_compatibility():
    """Test that environment works without camera (backward compatibility)."""
    print("\n" + "="*60)
    print("Test 1: Backward Compatibility (use_camera=False)")
    print("="*60)
    
    env = Panda3DQuadrotorEnv(use_camera=False, n=10)
    
    # Check observation space is Box (not Dict)
    assert env.observation_space.shape == (13,), f"Expected shape (13,), got {env.observation_space.shape}"
    print("[OK] Observation space is Box(13,)")
    
    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape == (13,), f"Expected observation shape (13,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
    print(f"[OK] Reset returns observation with shape {obs.shape} and dtype {obs.dtype}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (13,), f"Expected observation shape (13,), got {obs.shape}"
    print(f"[OK] Step returns observation with shape {obs.shape}")
    
    env.close()
    print("[OK] Backward compatibility test passed!\n")


def test_camera_observation():
    """Test that camera observations work correctly."""
    print("="*60)
    print("Test 2: Camera Observation (use_camera=True)")
    print("="*60)
    
    env = Panda3DQuadrotorEnv(
        use_camera=True,
        camera_high_freq_size=(64, 64),
        camera_low_freq_size=(320, 320),
        physics_steps_per_high_freq_capture=1,
        physics_steps_per_low_freq_capture=5,
        n=10
    )
    
    # Check observation space is Dict
    assert hasattr(env.observation_space, 'spaces'), "Expected Dict observation space"
    assert 'state' in env.observation_space.spaces, "Missing 'state' key"
    assert 'camera_high_freq' in env.observation_space.spaces, "Missing 'camera_high_freq' key"
    assert 'camera_low_freq' in env.observation_space.spaces, "Missing 'camera_low_freq' key"
    print("[OK] Observation space is Dict with correct keys")
    
    # Check shapes and dtypes
    state_space = env.observation_space['state']
    high_freq_space = env.observation_space['camera_high_freq']
    low_freq_space = env.observation_space['camera_low_freq']
    
    assert state_space.shape == (13,), f"Expected state shape (13,), got {state_space.shape}"
    assert high_freq_space.shape == (64, 64, 3), f"Expected camera_high_freq shape (64, 64, 3), got {high_freq_space.shape}"
    assert low_freq_space.shape == (320, 320, 3), f"Expected camera_low_freq shape (320, 320, 3), got {low_freq_space.shape}"
    
    assert high_freq_space.dtype == np.uint8, f"Expected uint8, got {high_freq_space.dtype}"
    assert low_freq_space.dtype == np.uint8, f"Expected uint8, got {low_freq_space.dtype}"
    print("[OK] Observation space shapes: state(13,), high_freq(64,64,3), low_freq(320,320,3)")
    print("[OK] Image dtypes are uint8 (memory efficient)")
    
    # Test reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict), f"Expected dict observation, got {type(obs)}"
    assert 'state' in obs and 'camera_high_freq' in obs and 'camera_low_freq' in obs
    print("[OK] Reset returns Dict observation")
    
    # Check actual observation shapes
    assert obs['state'].shape == (13,), f"Expected state shape (13,), got {obs['state'].shape}"
    assert obs['camera_high_freq'].shape == (64, 64, 3), f"Expected shape (64,64,3), got {obs['camera_high_freq'].shape}"
    assert obs['camera_low_freq'].shape == (320, 320, 3), f"Expected shape (320,320,3), got {obs['camera_low_freq'].shape}"
    print(f"[OK] Actual observation shapes match specification")
    
    # Check dtypes
    assert obs['camera_high_freq'].dtype == np.uint8, f"Expected uint8, got {obs['camera_high_freq'].dtype}"
    assert obs['camera_low_freq'].dtype == np.uint8, f"Expected uint8, got {obs['camera_low_freq'].dtype}"
    print("[OK] Image observations are uint8 (0-255 range)")
    
    # Check value ranges
    assert 0 <= obs['camera_high_freq'].min() <= 255, "Image values out of range"
    assert 0 <= obs['camera_high_freq'].max() <= 255, "Image values out of range"
    print(f"[OK] Image pixel values in valid range [0, 255]")
    
    # Test step with frame skip
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, dict), f"Step {i}: Expected dict, got {type(obs)}"
        assert obs['state'].shape == (13,), f"Step {i}: Wrong state shape"
        assert obs['camera_high_freq'].shape == (64, 64, 3), f"Step {i}: Wrong high-freq shape"
        assert obs['camera_low_freq'].shape == (320, 320, 3), f"Step {i}: Wrong low-freq shape"
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"[OK] Step returns correct Dict observations for 10 steps")
    
    env.close()
    print("[OK] Camera observation test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  CAMERA INTEGRATION FUNCTIONALITY TESTS")
    print("="*60 + "\n")
    
    try:
        test_backward_compatibility()
        test_camera_observation()
        
        print("="*60)
        print("  ALL TESTS PASSED!")
        print("="*60 + "\n")
        return 0
    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
