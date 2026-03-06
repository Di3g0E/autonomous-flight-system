"""
Lightweight U-Net for Monocular Depth Prediction.

Optimized for training on personal computers with limited GPU memory.
Uses depthwise separable convolutions and efficient skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DownBlock(nn.Module):
    """Encoder block with downsampling."""
    
    def __init__(self, in_channels, out_channels, use_depthwise=True):
        super().__init__()
        
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
            self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Decoder block with upsampling."""
    
    def __init__(self, in_channels, out_channels, use_depthwise=True):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
            self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch (padding if needed)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net for monocular depth estimation.
    
    Optimized for 64x64 input images on personal computers.
    Uses ~1.2M parameters (vs ~31M for standard U-Net).
    
    Args:
        in_channels: Input channels (3 for RGB)
        out_channels: Output channels (1 for depth)
        base_channels: Base number of channels (default: 32)
        use_depthwise: Use depthwise separable convolutions (default: True)
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=32,
        use_depthwise=True
    ):
        super().__init__()
        
        c = base_channels
        
        # Encoder
        self.down1 = DownBlock(in_channels, c, use_depthwise)     # 64 -> 32
        self.down2 = DownBlock(c, c*2, use_depthwise)             # 32 -> 16
        self.down3 = DownBlock(c*2, c*4, use_depthwise)           # 16 -> 8
        
        # Bottleneck
        if use_depthwise:
            self.bottleneck = nn.Sequential(
                DepthwiseSeparableConv(c*4, c*8),
                DepthwiseSeparableConv(c*8, c*8)
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(c*4, c*8, 3, padding=1, bias=False),
                nn.BatchNorm2d(c*8),
                nn.ReLU(inplace=True),
                nn.Conv2d(c*8, c*8, 3, padding=1, bias=False),
                nn.BatchNorm2d(c*8),
                nn.ReLU(inplace=True)
            )
        
        # Decoder
        self.up1 = UpBlock(c*8, c*4, use_depthwise)               # 8 -> 16
        self.up2 = UpBlock(c*4, c*2, use_depthwise)               # 16 -> 32
        self.up3 = UpBlock(c*2, c, use_depthwise)                 # 32 -> 64
        
        # Output
        self.out_conv = nn.Conv2d(c, out_channels, 1)
        self.sigmoid = nn.Sigmoid()  # For normalized depth [0, 1]
    
    def forward(self, x):
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        
        # Output
        x = self.out_conv(x)
        x = self.sigmoid(x)  # Normalize to [0, 1]
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_type='lightweight', in_channels=3, out_channels=1):
    """
    Factory function to create depth prediction models.
    
    Args:
        model_type: 'lightweight' or 'standard'
        in_channels: Input channels (3 for RGB)
        out_channels: Output channels (1 for depth)
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'lightweight':
        model = LightweightUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,
            use_depthwise=True
        )
    elif model_type == 'standard':
        model = LightweightUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=64,
            use_depthwise=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test model
    model = get_model('lightweight')
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
