"""
Test suite for depth buffer extraction functionality.

This module tests the depth buffer extraction system added to the
Panda3D camera and environment observation space.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


class TestDepthExtractionHeadless:
    ""Test depth extraction in headless mode (without Panda3D)."""
    
    def test_depth_disabled_by_default(self):
        """Test that depth is disabled by default for backward compatibility."""
        env = Panda3DQuadrotorEnv(use_camera=False)
        
        assert not hasattr(env, 'use_depth') or env.use_depth == False
        assert env.observation_space.shape == (13,)
        
        env.close()
    
    def test_depth_requires_camera(self):
        """Test that use_depth=True without use_camera=True is ignored."""
        env = Panda3DQuadrotorEnv(use_camera=False, use_depth=True)
        
        # Depth should be disabled because camera is not enabled
        assert env.use_depth == False
        assert env.observation_space.shape == (13,)
        
        env.close()
    
    def test_observation_space_with_depth(self):
        """Test that observation space includes depth when enabled."""
        # Create env with camera and depth (headless mode)
        env = Panda3DQuadrotorEnv(
            use_camera=True,
            use_depth=True,
            camera_high_freq_size=(64, 64),
            camera_low_freq_size=(320, 320)
        )
        
        # Observation space should be a Dict
        from gymnasium import spaces
        assert isinstance(env.observation_space, spaces.Dict)
        
        # Check keys
        expected_keys = {"state", "camera_high_freq", "camera_low_freq", 
                        "depth_high_freq", "depth_low_freq"}
        assert set(env.observation_space.spaces.keys()) == expected_keys
        
        # Check depth shapes
        assert env.observation_space["depth_high_freq"].shape == (64, 64, 1)
        assert env.observation_space["depth_low_freq"].shape == (320, 320, 1)
        
        # Check depth dtypes
        assert env.observation_space["depth_high_freq"].dtype == np.float32
        assert env.observation_space["depth_low_freq"].dtype == np.float32
        
        env.close()
    
    def test_reset_with_depth(self):
        """Test that reset initializes depth placeholders."""
        env = Panda3DQuadrotorEnv(
            use_camera=True,
            use_depth=True,
            camera_high_freq_size=(64, 64)
        )
        
        obs, info = env.reset()
        
        # Check that depth keys exist
        assert "depth_high_freq" in obs
        assert "depth_low_freq" in obs
        
        #Check shapes
        assert obs["depth_high_freq"].shape == (64, 64, 1)
        assert obs["depth_low_freq"].shape == (320, 320, 1)
        
        # Check dtypes
        assert obs["depth_high_freq"].dtype == np.float32
        assert obs["depth_low_freq"].dtype == np.float32
        
        # In headless mode, depth should be all zeros (placeholder)
        assert np.all(obs["depth_high_freq"] == 0)
        assert np.all(obs["depth_low_freq"] == 0)
        
        env.close()
    
    def test_step_with_depth(self):
        """Test that step returns valid depth observations."""
        env = Panda3DQuadrotorEnv(
            use_camera=True,
            use_depth=True
        )
        
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Depth should still be present
        assert "depth_high_freq" in obs
        assert "depth_low_freq" in obs
        
        # Should be valid numpy arrays
        assert isinstance(obs["depth_high_freq"], np.ndarray)
        assert isinstance(obs["depth_low_freq"], np.ndarray)
        
        # No NaN or Inf in placeholders
        assert not np.any(np.isnan(obs["depth_high_freq"]))
        assert not np.any(np.isinf(obs["depth_high_freq"]))
        
        env.close()
    
    def test_depth_metric_vs_normalized(self):
        """Test depth_metric parameter."""
        env_normalized = Panda3DQuadrotorEnv(
            use_camera=True,
            use_depth=True,
            depth_metric=False
        )
        
        env_metric = Panda3DQuadrotorEnv(
            use_camera=True,
            use_depth=True,
            depth_metric=True
        )
        
        assert env_normalized.depth_metric == False
        assert env_metric.depth_metric == True
        
        env_normalized.close()
        env_metric.close()


class TestDepthObservationConsistency:
    """Test depth observation consistency across episodes."""
    
    def test_depth_placeholders_reset_properly(self):
        """Test that depth placeholders are properly reset between episodes."""
        env = Panda3DQuadrotorEnv(
            use_camera=True,
            use_depth=True
        )
        
        # First episode
        obs1, _ = env.reset(seed=42)
        depth1_initial = obs1["depth_high_freq"].copy()
        
        # Take some steps
        for _ in range(10):
            action = env.action_space.sample()
            obs1, _, _, _, _ = env.step(action)
        
        # Second episode with different seed
        obs2, _ = env.reset(seed=123)
        depth2_initial = obs2["depth_high_freq"].copy()
        
        # Initial depths should be the same (both placeholders)
        np.testing.assert_array_equal(depth1_initial, depth2_initial)
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
