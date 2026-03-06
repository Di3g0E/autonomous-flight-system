"""
Models package for depth prediction.

This package contains neural network architectures for monocular depth estimation.
"""

from .depth_unet import LightweightUNet, get_model

__all__ = ['LightweightUNet', 'get_model']
