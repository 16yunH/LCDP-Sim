"""
Vision Encoder Module
Provides ResNet-18 and Vision Transformer (ViT) encoders for image feature extraction.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal, Optional


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax layer that converts feature maps to keypoint coordinates.
    This helps preserve spatial structure information.
    """
    def __init__(self, height: int, width: int, channel: int, temperature: Optional[float] = None):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel
        
        if temperature is not None:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0
        
        # Create coordinate grids
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1., 1., self.height),
            torch.linspace(-1., 1., self.width),
            indexing='ij'
        )
        pos_x = pos_x.reshape(self.height * self.width)
        pos_y = pos_y.reshape(self.height * self.width)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map
        Returns:
            [B, C*2] spatial coordinates
        """
        batch_size = x.shape[0]
        
        # Flatten spatial dimensions: [B, C, H*W]
        x = x.view(batch_size, self.channel, self.height * self.width)
        
        # Apply softmax: [B, C, H*W]
        softmax_attention = nn.functional.softmax(x / self.temperature, dim=-1)
        
        # Compute expected coordinates
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=-1, keepdim=True)  # [B, C, 1]
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=-1, keepdim=True)  # [B, C, 1]
        
        # Concatenate: [B, C*2]
        expected_xy = torch.cat([expected_x, expected_y], dim=-1)
        feature_keypoints = expected_xy.view(batch_size, -1)
        
        return feature_keypoints


class ResNetEncoder(nn.Module):
    """
    ResNet-18 based vision encoder with spatial feature preservation.
    """
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_spatial_softmax: bool = True,
        output_dim: int = 512,
    ):
        super(ResNetEncoder, self).__init__()
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final FC layer and avgpool to keep spatial structure
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.conv_layers.parameters():
                param.requires_grad = False
        
        # Get feature map size (assuming input is 224x224)
        # After ResNet-18: 512 x 7 x 7
        self.feature_channels = 512
        self.feature_height = 7
        self.feature_width = 7
        
        self.use_spatial_softmax = use_spatial_softmax
        
        if use_spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(
                height=self.feature_height,
                width=self.feature_width,
                channel=self.feature_channels,
                temperature=1.0
            )
            spatial_output_dim = self.feature_channels * 2
            self.fc = nn.Linear(spatial_output_dim, output_dim)
        else:
            # Global average pooling
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.feature_channels, output_dim)
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB images (typically 224x224)
        Returns:
            [B, output_dim] visual features
        """
        # Extract convolutional features: [B, 512, 7, 7]
        features = self.conv_layers(x)
        
        if self.use_spatial_softmax:
            # Use spatial softmax to preserve spatial structure
            features = self.spatial_softmax(features)  # [B, 1024]
        else:
            # Global average pooling
            features = self.avgpool(features)
            features = features.flatten(1)  # [B, 512]
        
        # Project to output dimension
        features = self.fc(features)
        
        return features


class VisionEncoder(nn.Module):
    """
    Unified Vision Encoder supporting multiple backbone architectures.
    """
    def __init__(
        self,
        encoder_type: Literal["resnet18", "vit"] = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 512,
        **kwargs
    ):
        super(VisionEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == "resnet18":
            self.encoder = ResNetEncoder(
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                output_dim=output_dim,
                **kwargs
            )
        elif encoder_type == "vit":
            # Placeholder for Vision Transformer implementation
            # Can use timm library: timm.create_model('vit_base_patch16_224', pretrained=True)
            raise NotImplementedError("ViT encoder will be implemented in future versions")
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB images
        Returns:
            [B, output_dim] visual features
        """
        return self.encoder(x)


if __name__ == "__main__":
    # Test the encoder
    encoder = VisionEncoder(
        encoder_type="resnet18",
        pretrained=True,
        freeze_backbone=False,
        output_dim=512,
        use_spatial_softmax=True
    )
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dim: {encoder.output_dim}")
