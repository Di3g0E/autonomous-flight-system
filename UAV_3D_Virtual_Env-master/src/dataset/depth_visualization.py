"""
Visualization tools for depth maps and RGB-Depth pairs.

This module provides utilities for visualizing depth data in various formats,
including colormaps, side-by-side comparisons, and 3D point clouds.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, Tuple


def apply_colormap_to_depth(
    depth_map: np.ndarray,
    colormap: str = 'turbo',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    invalid_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Apply colormap to depth map for visualization.
    
    Args:
        depth_map: Depth map, shape (H, W) or (H, W, 1), dtype float32
        colormap: Matplotlib colormap name ('turbo', 'viridis', 'jet', 'plasma', etc.)
        vmin: Minimum depth value for colormap normalization (None = auto from data)
        vmax: Maximum depth value for colormap normalization (None = auto from data)
        invalid_color: RGB color for invalid pixels (inf, nan), default black
    
    Returns:
        rgb_depth: Colorized depth map, shape (H, W, 3), dtype uint8
    """
    # Squeeze to (H, W) if needed
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(-1)
    
    assert depth_map.ndim == 2, f"Depth must be 2D, got shape {depth_map.shape}"
    
    # Create mask for valid pixels
    valid_mask = np.isfinite(depth_map)
    
    # Compute vmin/vmax from valid pixels if not provided
    if vmin is None:
        vmin = np.min(depth_map[valid_mask]) if valid_mask.any() else 0
    if vmax is None:
        vmax = np.max(depth_map[valid_mask]) if valid_mask.any() else 1
    
    # Normalize to [0, 1]
    depth_norm = np.clip((depth_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    rgb_depth = cmap(depth_norm)[:, :, :3]  # Drop alpha channel
    
    # Convert to uint8
    rgb_depth = (rgb_depth * 255).astype(np.uint8)
    
    # Apply invalid color to invalid pixels
    rgb_depth[~valid_mask] = invalid_color
    
    return rgb_depth


def save_rgb_depth_pair(
    rgb: np.ndarray,
    depth: np.ndarray,
    save_path: str,
    title: Optional[str] = None,
    depth_colormap: str = 'turbo',
    depth_vmin: Optional[float] = None,
    depth_vmax: Optional[float] = None
):
    """
    Save side-by-side RGB and depth visualization.
    
    Args:
        rgb: RGB image, shape (H, W, 3), dtype uint8
        depth: Depth map, shape (H, W, 1) or (H, W), dtype float32
        save_path: Output file path
        title: Optional title for the figure
        depth_colormap: Colormap for depth visualization
        depth_vmin: Min depth for colormap
        depth_vmax: Max depth for colormap
    """
    # Colorize depth
    depth_rgb = apply_colormap_to_depth(depth, depth_colormap, depth_vmin, depth_vmax)
    
    # Create side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # RGB image
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Depth map
    im = axes[1].imshow(depth_rgb)
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    
    # Add colorbar for depth
    fig.colorbar(cm.ScalarMappable(cmap=depth_colormap), ax=axes[1], fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_depth_statistics(
    depth_map: np.ndarray,
    save_path: Optional[str] = None
) -> Dict:
    """
    Visualize depth statistics with histogram.
    
    Args:
        depth_map: Depth map, shape (H, W, 1) or (H, W)
        save_path: Optional path to save figure
    
    Returns:
        stats: Dictionary with depth statistics
    """
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(-1)
    
    # Compute statistics
    valid_mask = np.isfinite(depth_map)
    valid_depth = depth_map[valid_mask]
    
    stats = {
        "min": float(np.min(valid_depth)) if valid_mask.any() else None,
        "max": float(np.max(valid_depth)) if valid_mask.any() else None,
        "mean": float(np.mean(valid_depth)) if valid_mask.any() else None,
        "median": float(np.median(valid_depth)) if valid_mask.any() else None,
        "std": float(np.std(valid_depth)) if valid_mask.any() else None,
        "valid_pixels": int(np.sum(valid_mask)),
        "invalid_pixels": int(np.sum(~valid_mask)),
        "coverage": float(np.sum(valid_mask) / depth_map.size)
    }
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid_depth.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Depth Distribution\nMean: {stats["mean"]:.2f}m, Std: {stats["std"]:.2f}m')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Min: {stats['min']:.2f}m\nMax: {stats['max']:.2f}m\nMedian: {stats['median']:.2f}m\nCoverage: {stats['coverage']*100:.1f}%"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return stats


def create_depth_comparison_grid(
    rgb_list: list,
    depth_list: list,
    titles: list,
    save_path: str,
    depth_colormap: str = 'turbo'
):
    """
    Create a grid of RGB-Depth pairs for comparison.
    
    Args:
        rgb_list: List of RGB images
        depth_list: List of depth maps
        titles: List of titles for each pair
        save_path: Output file path
        depth_colormap: Colormap for depth
    """
    n = len(rgb_list)
    assert n == len(depth_list) == len(titles), "Lists must have same length"
    
    fig, axes = plt.subplots(n, 2, figsize=(10, 4*n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, (rgb, depth, title) in enumerate(zip(rgb_list, depth_list, titles)):
        # RGB
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f'{title} - RGB')
        axes[i, 0].axis('off')
        
        # Depth
        depth_rgb = apply_colormap_to_depth(depth, depth_colormap)
        axes[i, 1].imshow(depth_rgb)
        axes[i, 1].set_title(f'{title} - Depth')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Depth visualization utilities")
    print("Import this module to use visualization functions")
