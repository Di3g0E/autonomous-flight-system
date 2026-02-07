"""
Test Suite: Collision Detection System

This script runs comprehensive tests on the collision detection system
to verify all components work correctly.
"""

import numpy as np
from src.envs.quadrotor_env import quad
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


def test_base_environment():
    """Test 1: Base quadrotor environment (no Panda3D)"""
    print("=" * 70)
    print("TEST 1: Base Quadrotor Environment")
    print("=" * 70)
    
    try:
        env = quad(t_step=0.01, n=100, direct_control=1, T=1)
        
        # Test reset
        observation, info = env.reset(seed=42)
        assert observation.shape == (13,), "Observation shape incorrect"
        assert isinstance(info, dict), "Info should be a dict"
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (13,), "Step observation shape incorrect"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(terminated, bool), "Terminated should be bool"
        assert isinstance(truncated, bool), "Truncated should be bool"
        
        env.close()
        
        print("[PASS] Base environment works correctly")
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_wrapper_without_panda3d():
    """Test 2: Panda3D wrapper without Panda3D components"""
    print("\n" + "=" * 70)
    print("TEST 2: Wrapper Without Panda3D")
    print("=" * 70)
    
    try:
        env = Panda3DQuadrotorEnv(
            panda3d_app=None,
            quad_model=None,
            render_node=None,
            t_step=0.01,
            n=100,
            direct_control=1,
            enable_collisions=False
        )
        
        # Verify collisions are disabled
        assert env.enable_collisions == False, "Collisions should be disabled"
        
        # Test reset
        observation, info = env.reset(seed=42)
        assert 'collision' in info, "Info should contain collision key"
        assert info['collision'] == {}, "Collision info should be empty"
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert 'collision_occurred' in info, "Info should contain collision_occurred"
        assert info['collision_occurred'] == False, "No collision should occur"
        
        env.close()
        
        print("[PASS] Wrapper works without Panda3D")
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_obstacle_api():
    """Test 3: Obstacle management API"""
    print("\n" + "=" * 70)
    print("TEST 3: Obstacle Management API")
    print("=" * 70)
    
    try:
        env = Panda3DQuadrotorEnv(
            panda3d_app=None,
            quad_model=None,
            render_node=None,
            enable_collisions=False
        )
        
        # Test adding obstacles (should return None without Panda3D)
        result = env.add_box_obstacle((1, 1, 1), (0.5, 0.5, 0.5))
        assert result is None, "Should return None without Panda3D"
        
        result = env.add_sphere_obstacle((2, 2, 2), 0.3)
        assert result is None, "Should return None without Panda3D"
        
        # Test clearing obstacles
        env.clear_obstacles()  # Should not raise error
        
        env.close()
        
        print("[PASS] Obstacle API works correctly")
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_collision_configuration():
    """Test 4: Collision configuration parameters"""
    print("\n" + "=" * 70)
    print("TEST 4: Collision Configuration")
    print("=" * 70)
    
    try:
        env = Panda3DQuadrotorEnv(
            panda3d_app=None,
            quad_model=None,
            render_node=None,
            collision_radius=0.5,
            collision_penalty=-150.0
        )
        
        # Verify configuration
        assert env.collision_radius == 0.5, "Collision radius not set correctly"
        assert env.collision_penalty == -150.0, "Collision penalty not set correctly"
        
        # Test dynamic configuration
        env.set_collision_penalty(-200.0)
        assert env.collision_penalty == -200.0, "Penalty update failed"
        
        env.close()
        
        print("[PASS] Configuration works correctly")
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_info_dict_structure():
    """Test 5: Info dictionary structure"""
    print("\n" + "=" * 70)
    print("TEST 5: Info Dictionary Structure")
    print("=" * 70)
    
    try:
        env = Panda3DQuadrotorEnv(
            panda3d_app=None,
            quad_model=None,
            render_node=None,
            enable_collisions=False
        )
        
        # Test reset info
        observation, info = env.reset(seed=42)
        required_keys = ['solved', 'angular_position', 'collision']
        for key in required_keys:
            assert key in info, f"Missing key in reset info: {key}"
        
        # Test step info
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        required_keys = ['solved', 'angular_position', 'clipped_action', 
                        'timestep', 'collision', 'collision_occurred']
        for key in required_keys:
            assert key in info, f"Missing key in step info: {key}"
        
        # Verify collision info structure
        collision_info = info['collision']
        assert isinstance(collision_info, dict), "Collision info should be dict"
        
        env.close()
        
        print("[PASS] Info dict structure is correct")
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_episode_execution():
    """Test 6: Complete episode execution"""
    print("\n" + "=" * 70)
    print("TEST 6: Complete Episode Execution")
    print("=" * 70)
    
    try:
        env = Panda3DQuadrotorEnv(
            panda3d_app=None,
            quad_model=None,
            render_node=None,
            t_step=0.01,
            n=50,
            direct_control=1
        )
        
        observation, info = env.reset(seed=42)
        episode_reward = 0
        steps = 0
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        assert steps > 0, "No steps executed"
        assert isinstance(episode_reward, (int, float)), "Invalid reward type"
        
        env.close()
        
        print(f"[PASS] Episode executed successfully ({steps} steps, reward={episode_reward:.2f})")
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_gymnasium_compatibility():
    """Test 7: Gymnasium API compatibility"""
    print("\n" + "=" * 70)
    print("TEST 7: Gymnasium API Compatibility")
    print("=" * 70)
    
    try:
        env = Panda3DQuadrotorEnv(
            panda3d_app=None,
            quad_model=None,
            render_node=None
        )
        
        # Check required attributes
        assert hasattr(env, 'action_space'), "Missing action_space"
        assert hasattr(env, 'observation_space'), "Missing observation_space"
        assert hasattr(env, 'reset'), "Missing reset method"
        assert hasattr(env, 'step'), "Missing step method"
        assert hasattr(env, 'render'), "Missing render method"
        assert hasattr(env, 'close'), "Missing close method"
        
        # Check metadata
        assert hasattr(env, 'metadata'), "Missing metadata"
        assert 'render_modes' in env.metadata, "Missing render_modes in metadata"
        
        # Verify action and observation spaces
        assert env.action_space.shape == (4,), "Action space shape incorrect"
        assert env.observation_space.shape == (13,), "Observation space shape incorrect"
        
        env.close()
        
        print("[PASS] Gymnasium API compatibility verified")
        return True
        
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 70)
    print("COLLISION DETECTION SYSTEM - TEST SUITE")
    print("=" * 70 + "\n")
    
    tests = [
        test_base_environment,
        test_wrapper_without_panda3d,
        test_obstacle_api,
        test_collision_configuration,
        test_info_dict_structure,
        test_episode_execution,
        test_gymnasium_compatibility
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    print(f"Success rate: {100*passed/total:.1f}%")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! System is working correctly.")
    else:
        print(f"\n[WARNING] {total-passed} test(s) failed. Please review.")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
