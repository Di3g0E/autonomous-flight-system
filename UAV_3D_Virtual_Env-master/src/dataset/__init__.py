"""
Dataset utilities for depth estimation.

This package provides tools for collecting, loading, and visualizing RGB-Depth datasets.
"""

from .depth_dataset_collector import DepthDatasetCollector, load_dataset_sample

__all__ = ['DepthDatasetCollector', 'load_dataset_sample']
