#!/usr/bin/env python
"""
Visualize samples from collected depth dataset.

This script loads and displays RGB-Depth pairs from a collected dataset
to verify data quality.

Usage:
    python scripts/visualize_depth_samples.py --dataset-dir ./data/depth_dataset --num-samples 10
"""

import argparse
import sys
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.dataset.depth_dataset_collector import load_dataset_sample
from src.dataset.depth_visualization import (
    save_rgb_depth_pair,
    apply_colormap_to_depth,
    visualize_depth_statistics
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize depth dataset samples")
    
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Which split to visualize')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save visualizations (default: dataset_dir/visualizations)')
    parser.add_argument('--colormap', type=str, default='turbo',
                        help='Matplotlib colormap for depth')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show depth statistics')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    split_dir = dataset_dir / args.split
    
    # Check if dataset exists
    if not split_dir.exists():
        print(f"ERROR: Dataset split directory not found: {split_dir}")
        return
    
    # Get list of H5 files
    h5_files = sorted(list(split_dir.glob("*.h5")))
    
    if len(h5_files) == 0:
        print(f"ERROR: No HDF5 files found in {split_dir}")
        return
    
    print(f"Found {len(h5_files)} HDF5 files in {args.split} split")
    
    # Set up save directory
    if args.save_dir is None:
        save_dir = dataset_dir / "visualizations" / args.split
    else:
        save_dir = Path(args.save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {save_dir}\n")
    
    # Collect samples to visualize
    samples_to_viz = []
    samples_collected = 0
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            num_samples_in_file = f.attrs['num_samples']
            
            for idx in range(num_samples_in_file):
                if samples_collected >= args.num_samples:
                    break
                
                rgb, depth, metadata = load_dataset_sample(str(h5_file), idx)
                samples_to_viz.append((rgb, depth, metadata, h5_file.stem, idx))
                samples_collected += 1
        
        if samples_collected >= args.num_samples:
            break
    
    # Visualize each sample
    print(f"Visualizing {len(samples_to_viz)} samples...\n")
    
    all_depths = []
    
    for i, (rgb, depth, metadata, file_stem, idx) in enumerate(samples_to_viz):
        # Save RGB-Depth pair
        title = f"{args.split.capitalize()} Sample {i+1} (file: {file_stem}, idx: {idx})"
        output_path = save_dir / f"sample_{i+1:03d}.png"
        
        save_rgb_depth_pair(
            rgb, depth, str(output_path),
            title=title,
            depth_colormap=args.colormap
        )
        
        print(f"  [{i+1}/{len(samples_to_viz)}] Saved: {output_path.name}")
        print(f"      RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
        print(f"      Depth range: [{np.min(depth):.3f}, {np.max(depth):.3f}]")
        
        # Collect for statistics
        all_depths.append(depth.flatten())
    
    # Show overall statistics if requested
    if args.show_stats and len(all_depths) > 0:
        print("\nComputing overall depth statistics...")
        combined_depth = np.concatenate(all_depths)
        combined_depth_2d = combined_depth.reshape(-1, 1)  # Force shape for visualization
        
        stats_path = save_dir / "depth_statistics.png"
        stats = visualize_depth_statistics(combined_depth_2d, str(stats_path))
        
        print(f"\nDepth Statistics ({len(samples_to_viz)} samples):")
        print(f"  Min:    {stats['min']:.3f}")
        print(f"  Max:    {stats['max']:.3f}")
        print(f"  Mean:   {stats['mean']:.3f}")
        print(f"  Median: {stats['median']:.3f}")
        print(f"  Std:    {stats['std']:.3f}")
        print(f"  Valid:  {stats['valid_pixels']:,}")
        print(f"  Invalid: {stats['invalid_pixels']:,}")
        print(f"  Coverage: {stats['coverage']*100:.1f}%")
        print(f"\nStatistics plot saved: {stats_path}")
    
    print(f"\n✓ Visualization complete!")
    print(f"  Saved {len(samples_to_viz)} visualizations to: {save_dir}")


if __name__ == "__main__":
    main()
