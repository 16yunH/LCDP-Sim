"""
1D U-Net Architecture for Action Sequence Modeling
Adapted for temporal action sequences in diffusion policy.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position encoding for diffusion timesteps.
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B,] diffusion timesteps
        Returns:
            [B, dim] position embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        
        return embeddings


class Conv1dBlock(nn.Module):
    """
    1D Convolutional block with GroupNorm and activation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 8,
        activation: str = "mish"
    ):
        super(Conv1dBlock, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(groups, out_channels)
        
        if activation == "mish":
            self.activation = nn.Mish()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock1D(nn.Module):
    """
    Residual block for 1D convolutions with time embedding conditioning.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        kernel_size: int = 3,
        groups: int = 8
    ):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, groups=groups)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, groups=groups)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, T] input features
            time_emb: [B, time_emb_dim] time embeddings
        Returns:
            [B, out_channels, T] output features
        """
        residual = self.residual_conv(x)
        
        # First conv
        h = self.conv1(x)
        
        # Add time embedding: [B, out_channels, 1]
        time_emb = self.time_mlp(time_emb)[:, :, None]
        h = h + time_emb
        
        # Second conv
        h = self.conv2(h)
        
        return h + residual


class DownBlock1D(nn.Module):
    """
    Downsampling block for U-Net encoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        downsample: bool = True
    ):
        super(DownBlock1D, self).__init__()
        
        self.layers = nn.ModuleList([
            ResidualBlock1D(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])
        
        if downsample:
            self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        for layer in self.layers:
            x = layer(x, time_emb)
        
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x_down, x  # Return both downsampled and skip connection
        else:
            return x, x


class UpBlock1D(nn.Module):
    """
    Upsampling block for U-Net decoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        upsample: bool = True
    ):
        super(UpBlock1D, self).__init__()
        
        if upsample:
            self.upsample = nn.ConvTranspose1d(in_channels, in_channels, 4, stride=2, padding=1)
        else:
            self.upsample = None
        
        self.layers = nn.ModuleList([
            ResidualBlock1D(
                in_channels + out_channels if i == 0 else out_channels,  # +out_channels for skip connection
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor):
        if self.upsample is not None:
            x = self.upsample(x)
        
        # Ensure skip connection matches spatial dimension after upsampling
        if x.shape[2] != skip.shape[2]:
            # Interpolate skip to match x's temporal dimension
            skip = torch.nn.functional.interpolate(skip, size=x.shape[2], mode='linear', align_corners=False)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        for layer in self.layers:
            x = layer(x, time_emb)
        
        return x


class UNet1D(nn.Module):
    """
    1D U-Net for action sequence denoising in diffusion policy.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        time_emb_dim: int = 128,
        base_channels: int = 256,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input action dimension (e.g., 7 for robot actions)
            output_dim: Output action dimension (same as input_dim for denoising)
            time_emb_dim: Dimension of time embeddings
            base_channels: Base number of channels
            channel_mult: Channel multipliers for each resolution
            num_res_blocks: Number of residual blocks per resolution
            dropout: Dropout rate
        """
        super(UNet1D, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        time_emb_dim = time_emb_dim * 4
        
        # Initial projection
        self.input_proj = nn.Conv1d(input_dim, base_channels, 1)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        in_ch = base_channels
        
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            is_last = (i == len(channel_mult) - 1)
            
            self.down_blocks.append(
                DownBlock1D(
                    in_ch,
                    out_ch,
                    time_emb_dim,
                    num_layers=num_res_blocks,
                    downsample=not is_last
                )
            )
            channels.append(out_ch)
            in_ch = out_ch
        
        # Middle
        self.mid_block = ResidualBlock1D(in_ch, in_ch, time_emb_dim)
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            is_last = (i == 0)
            
            self.up_blocks.append(
                UpBlock1D(
                    in_ch,
                    out_ch,
                    time_emb_dim,
                    num_layers=num_res_blocks + 1,
                    upsample=not is_last
                )
            )
            in_ch = out_ch
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.Mish(),
            nn.Conv1d(base_channels, output_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim, T] noisy action sequences
            timesteps: [B,] diffusion timesteps
        Returns:
            [B, output_dim, T] denoised action sequences (or predicted noise)
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)  # [B, time_emb_dim]
        
        # Initial projection
        h = self.input_proj(x)  # [B, base_channels, T]
        
        # Encoder
        skip_connections = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb)
            skip_connections.append(skip)
        
        # Middle
        h = self.mid_block(h, time_emb)
        
        # Decoder
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            h = up_block(h, skip, time_emb)
        
        # Output projection
        h = self.output_proj(h)
        
        return h


if __name__ == "__main__":
    # Test U-Net
    batch_size = 4
    action_dim = 7  # [x, y, z, roll, pitch, yaw, gripper]
    horizon = 16
    
    model = UNet1D(
        input_dim=action_dim,
        output_dim=action_dim,
        time_emb_dim=128,
        base_channels=256,
        channel_mult=(1, 2, 4),
        num_res_blocks=2
    )
    
    # Create dummy inputs
    noisy_actions = torch.randn(batch_size, action_dim, horizon)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    output = model(noisy_actions, timesteps)
    
    print(f"Input shape: {noisy_actions.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
