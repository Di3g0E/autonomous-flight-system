"""
Test script for Camera Performance Benchmarking

This script measures FPS (frames per second) of the environment with and without
camera observation enabled. Use this to detect performance bottlenecks related to
camera capture and image processing.

Usage:
    python tests/test_camera_performance.py
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


def benchmark_env(use_camera=False, num_steps=1000, physics_steps_high=1, physics_steps_low=10):
    """
    Benchmark environment performance.
    
    Args:
        use_camera: Enable camera system
        num_steps: Number of steps to run
        physics_steps_high: Physics steps per high-freq camera capture
        physics_steps_low: Physics steps per low-freq camera capture
    
    Returns:
        float: FPS (frames per second)
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: use_camera={use_camera}")
    if use_camera:
        print(f"  - High-freq camera: every {physics_steps_high} step(s)")
        print(f"  - Low-freq camera: every {physics_steps_low} step(s)")
    print(f"  - Total steps: {num_steps}")
    print(f"{'='*60}\n")
    
    # Create environment
    # Note: For headless testing without Panda3D, camera_*_obj will be None
    # and cameras won't actually capture (will use placeholder black frames)
    env = Panda3DQuadrotorEnv(
        panda3d_app=None,  # Headless mode
        quad_model=None,
        render_node=None,
        t_step=0.01,
        n=num_steps,
        euler=0,
        direct_control=1,
        T=1,
        render_mode=None,
        enable_collisions=False,
        use_camera=use_camera,
        camera_high_freq_obj=None,  # Would need actual opencv_camera object
        camera_low_freq_obj=None,   # Would need actual opencv_camera object
        camera_high_freq_size=(64, 64),
        camera_low_freq_size=(320, 320),
        physics_steps_per_high_freq_capture=physics_steps_high,
        physics_steps_per_low_freq_capture=physics_steps_low
    )
    
    print("Running benchmark...")
    obs, _ = env.reset(seed=42)
    
    start = time.time()
    
    for i in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    elapsed = time.time() - start
    fps = num_steps / elapsed
    
    env.close()
    
    print(f"\n  [OK] Completed in {elapsed:.2f} seconds")
    print(f"  [OK] Average FPS: {fps:.2f}")
    
    if use_camera:
        # Calculate actual camera capture frequencies
        high_captures = num_steps // physics_steps_high
        low_captures = num_steps // physics_steps_low
        print(f"  [OK] High-freq captures: {high_captures}")
        print(f"  [OK] Low-freq captures: {low_captures}")
    
    return fps


def main():
    """Run performance benchmarks."""
    print("\n" + "="*60)
    print("  CAMERA PERFORMANCE BENCHMARK")
    print("="*60)
    
    num_steps = 1000
    
    # Test 1: Baseline (no camera)
    fps_nocam = benchmark_env(use_camera=False, num_steps=num_steps)
    
    # Test 2: With camera (default frequencies)
    fps_cam_default = benchmark_env(
        use_camera=True,
        num_steps=num_steps,
        physics_steps_high=1,
        physics_steps_low=10
    )
    
    # Test 3: With camera (reduced high-freq captures)
    fps_cam_reduced = benchmark_env(
        use_camera=True,
        num_steps=num_steps,
        physics_steps_high=4,
        physics_steps_low=10
    )
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\nNo camera:                  {fps_nocam:.2f} FPS (baseline)")
    print(f"Camera (default freq):      {fps_cam_default:.2f} FPS ({(fps_cam_default/fps_nocam)*100:.1f}% of baseline)")
    print(f"Camera (reduced high freq): {fps_cam_reduced:.2f} FPS ({(fps_cam_reduced/fps_nocam)*100:.1f}% of baseline)")
    
    # Analysis
    print("\n" + "-"*60)
    print("  ANALYSIS")
    print("-"*60)
    
    slowdown_default = ((fps_nocam - fps_cam_default) / fps_nocam) * 100
    slowdown_reduced = ((fps_nocam - fps_cam_reduced) / fps_nocam) * 100
    
    print(f"\nCamera overhead (default): {slowdown_default:.1f}% slowdown")
    print(f"Camera overhead (reduced): {slowdown_reduced:.1f}% slowdown")
    
    if fps_cam_default < 100:
        print("\n[!] WARNING: FPS with camera is below 100")
        print("   Consider:")
        print("   - Increasing physics_steps_per_high_freq_capture")
        print("   - Reducing camera resolution")
        print("   - Optimizing image capture method")
    elif fps_cam_default > 500:
        print("\n[OK] EXCELLENT: High FPS maintained with camera enabled")
    else:
        print("\n[OK] GOOD: Acceptable FPS for training with camera")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
