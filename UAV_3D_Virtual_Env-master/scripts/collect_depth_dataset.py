#!/usr/bin/env python
"""
Collect RGB-Depth dataset from the UAV simulation environment.

This script runs the simulation environment and collects paired RGB-Depth data
for training monocular depth prediction models.

Usage:
    python scripts/collect_depth_dataset.py --num-samples 10000 --output-dir ./data/depth_dataset
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.dataset.depth_dataset_collector import DepthDatasetCollector


def parse_args():
    parser = argparse.ArgumentParser(description="Collect RGB-Depth dataset from UAV simulation")
    
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of samples to collect')
    parser.add_argument('--output-dir', type=str, default='./data/depth_dataset',
                        help='Output directory for dataset')
    parser.add_argument('--camera', type=str, default='high_freq', choices=['high_freq', 'low_freq'],
                        help='Which camera to use for collection')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (0.0-1.0)')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test split ratio (0.0-1.0)')
    parser.add_argument('--depth-metric', action='store_true',
                        help='Use metric depth (meters) instead of normalized [0,1]')
    parser.add_argument('--random-actions', action='store_true',
                        help='Use random actions instead of hovering')
    parser.add_argument('--max-samples-per-file', type=int, default=1000,
                        help='Maximum samples per HDF5 file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def collect_dataset_headless(args):
    """
    Collect dataset in headless mode (no Panda3D visualization).
    
    This is the fastest way to collect data, but depth will be zero placeholders.
    Use this to test the pipeline before running with full Panda3D.
    """
    print("="*80)
    print("HEADLESS MODE COLLECTION (placeholders only)")
    print("="*80)
    print("NOTE: In headless mode, depth maps will be all zeros.")
    print("      This is useful for testing the collection pipeline.")
    print("      For real depth data, run with Panda3D environment.\n")
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create environment (headless)
    print(f"Creating environment (camera={args.camera})...")
    env = Panda3DQuadrotorEnv(
        use_camera=True,
        use_depth=True,
        depth_metric=args.depth_metric,
        camera_high_freq_size=(64, 64),
        camera_low_freq_size=(320, 320)
    )
    
    # Create dataset collector
    print(f"Initializing dataset collector...")
    print(f"  Output: {args.output_dir}")
    print(f"  Splits: train={1-args.val_split-args.test_split:.1%}, val={args.val_split:.1%}, test={args.test_split:.1%}\n")
    
    collector = DepthDatasetCollector(
        save_dir=args.output_dir,
        validation_split=args.val_split,
        test_split=args.test_split,
        max_samples_per_file=args.max_samples_per_file,
        camera_type=args.camera
    )
    
    # Collection loop
    print(f"Collecting {args.num_samples} samples...")
    obs, info = env.reset(seed=args.seed)
    
    samples_collected = 0
    episode_count = 0
    
    with tqdm(total=args.num_samples, desc="Collecting", unit="samples") as pbar:
        while samples_collected < args.num_samples:
            # Select action
            if args.random_actions:
                action = env.action_space.sample()
            else:
                # Hover action (neutral)
                action = np.array([0.25, 0.25, 0.25, 0.25])
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Extract RGB and depth based on camera choice
            if args.camera == 'high_freq':
                rgb = obs['camera_high_freq']
                depth = obs['depth_high_freq']
            else:  # low_freq
                rgb = obs['camera_low_freq']
                depth = obs['depth_low_freq']
            
            # Add sample to collector
            metadata = {
                'episode': episode_count,
                'step': samples_collected,
                'action': action.tolist(),
                'reward': float(reward),
                'state': obs['state'].tolist()
            }
            
            collector.add_sample(rgb, depth, metadata)
            samples_collected += 1
            pbar.update(1)
            
            # Reset if episode ended
            if terminated or truncated:
                obs, info = env.reset()
                episode_count += 1
    
    # Finalize dataset
    print("\nFinalizing dataset...")
    summary = collector.finalize()
    
    # Close environment
    env.close()
    
    return summary


def main():
    args = parse_args()
    
    print("\nRGB-Depth Dataset Collection Tool")
    print("="*80)
    print(f"Target samples: {args.num_samples}")
    print(f"Camera: {args.camera}")
    print(f"Depth format: {'metric (meters)' if args.depth_metric else 'normalized [0,1]'}")
    print(f"Action policy: {'random' if args.random_actions else 'hovering'}")
    print("="*80)
    print()
    
    # Run collection
    summary = collect_dataset_headless(args)
    
    print("\n✓ Dataset collection complete!")
    print(f"  Dataset saved to: {args.output_dir}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Summary file: {args.output_dir}/dataset_summary.json")
    print()


if __name__ == "__main__":
    main()
