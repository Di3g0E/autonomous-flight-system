"""
Custom Feature Extractors for Stable-Baselines3.

Provides CNN-based feature extractors for Dict observations that combine
state vectors with depth maps. Used for PPO/SAC training with depth input.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StateDepthExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observations with state + depth.
    
    Processes the depth map through a lightweight CNN and concatenates
    the resulting features with the raw state vector.
    
    Expected observation space (Dict):
        - 'state': Box(13,) - drone state vector
        - 'depth_high_freq': Box(H, W, 1) - depth map
    
    Architecture:
        depth -> CNN(3 layers) -> flatten -> FC(64) -> concat(state) -> FC(128) -> features
    
    Args:
        observation_space: Gymnasium Dict observation space
        features_dim: Output features dimension (default: 128)
        depth_key: Key for depth in observation dict (default: 'depth_high_freq')
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128,
                 depth_key: str = 'depth_high_freq'):
        # Must call super with the final features_dim
        super().__init__(observation_space, features_dim)
        
        self.depth_key = depth_key
        
        # Get dimensions from observation space
        state_dim = observation_space['state'].shape[0]
        depth_shape = observation_space[depth_key].shape  # (H, W, C)
        depth_h, depth_w, depth_c = depth_shape
        
        # CNN for depth processing
        # Input: (B, C, H, W) - note: SB3 auto-transposes channels
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(depth_c, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed output: 32 * 4 * 4 = 512
            nn.Flatten()
        )
        
        # Compute CNN output size
        cnn_output_dim = 32 * 4 * 4  # = 512
        
        # Depth feature reducer
        self.depth_fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 64),
            nn.ReLU()
        )
        
        # Combine state + depth features
        combined_dim = state_dim + 64
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: dict) -> torch.Tensor:
        # Extract state and depth
        state = observations['state']
        depth = observations[self.depth_key]
        
        # Depth: (B, H, W, C) -> (B, C, H, W) for PyTorch conv
        if depth.dim() == 4 and depth.shape[-1] < depth.shape[-2]:
            depth = depth.permute(0, 3, 1, 2)
        
        # Process depth through CNN
        depth_features = self.depth_cnn(depth)
        depth_features = self.depth_fc(depth_features)
        
        # Concatenate state + depth features
        combined = torch.cat([state, depth_features], dim=1)
        
        # Final projection
        return self.combined_fc(combined)


class StateOnlyExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that only uses the state vector from a Dict observation.
    
    Used as baseline for comparison: ignores depth/camera data and processes
    only the raw state vector through an MLP.
    
    Args:
        observation_space: Gymnasium Dict observation space
        features_dim: Output features dimension (default: 64)
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        state_dim = observation_space['state'].shape[0]
        
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: dict) -> torch.Tensor:
        return self.state_net(observations['state'])


class StateCameraExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observations with state + RGB camera.
    
    Processes the camera image through a CNN and concatenates
    with the state vector. Used for vision-based goal following.
    
    Expected observation space (Dict):
        - 'state': Box(13,) - drone state vector
        - 'camera_high_freq': Box(H, W, 3) - RGB camera image
    
    Architecture:
        camera -> CNN(3 layers) -> flatten -> FC(64) -> concat(state) -> FC(128) -> features
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128,
                 camera_key: str = 'camera_high_freq'):
        super().__init__(observation_space, features_dim)
        
        self.camera_key = camera_key
        
        # Get dimensions
        state_dim = observation_space['state'].shape[0]
        # NOTE: SB3 auto-preprocesses image observations (uint8 Box spaces)
        # from (H, W, C) -> (C, H, W), so the shape here is already (C, H, W)
        cam_shape = observation_space[camera_key].shape
        cam_c, cam_h, cam_w = cam_shape  # Channel-first after SB3 preprocessing
        
        # CNN for camera image processing
        # Input arrives as (B, C, H, W) from SB3 preprocessing
        self.camera_cnn = nn.Sequential(
            nn.Conv2d(cam_c, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed output: 64 * 4 * 4 = 1024
            nn.Flatten()
        )
        
        cnn_output_dim = 64 * 4 * 4  # = 1024
        
        # Camera feature reducer
        self.camera_fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Combine state + camera features
        combined_dim = state_dim + 64
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: dict) -> torch.Tensor:
        state = observations['state']
        camera = observations[self.camera_key].float() / 255.0  # Normalize uint8 to [0,1]
        
        # SB3 already provides camera as (B, C, H, W) — no permute needed
        
        # Process camera through CNN
        cam_features = self.camera_cnn(camera)
        cam_features = self.camera_fc(cam_features)
        
        # Concatenate state + camera features
        combined = torch.cat([state, cam_features], dim=1)
        
        return self.combined_fc(combined)
