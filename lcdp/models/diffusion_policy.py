"""
Diffusion Policy - Main Model
Integrates vision, language, and diffusion-based action generation.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Dict, Any
from diffusers import DDPMScheduler, DDIMScheduler

from lcdp.models.vision_encoder import VisionEncoder
from lcdp.models.language_encoder import LanguageEncoder
from lcdp.models.unet1d import UNet1D
from lcdp.models.conditioning import MultimodalFiLM, MultimodalCrossAttention


class DiffusionPolicy(nn.Module):
    """
    Language-Conditioned Diffusion Policy for Robot Manipulation.
    
    This model combines:
    - Vision encoder (ResNet/ViT) for image observations
    - Language encoder (CLIP) for text instructions
    - Conditional U-Net for diffusion-based action generation
    """
    def __init__(
        self,
        # Action space
        action_dim: int = 7,
        action_horizon: int = 16,
        
        # Vision encoder
        vision_encoder: str = "resnet18",
        vision_feature_dim: int = 512,
        freeze_vision_backbone: bool = False,
        
        # Language encoder
        language_model: str = "ViT-B/32",
        language_feature_dim: int = 512,
        freeze_language: bool = True,
        
        # Conditioning mechanism
        conditioning_type: Literal["film", "cross_attention"] = "cross_attention",
        
        # U-Net architecture
        unet_base_channels: int = 256,
        unet_channel_mult: tuple = (1, 2, 4),
        unet_num_res_blocks: int = 2,
        
        # Diffusion settings
        num_diffusion_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",  # "epsilon" or "sample"
        
        # Other
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(DiffusionPolicy, self).__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.conditioning_type = conditioning_type
        self.device = device
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(
            encoder_type=vision_encoder,
            pretrained=True,
            freeze_backbone=freeze_vision_backbone,
            output_dim=vision_feature_dim,
            use_spatial_softmax=True
        )
        
        # Language encoder
        self.language_encoder = LanguageEncoder(
            model_name=language_model,
            freeze=freeze_language,
            output_dim=language_feature_dim,
            device=device
        )
        
        # U-Net for action denoising
        self.unet = UNet1D(
            input_dim=action_dim,
            output_dim=action_dim,
            time_emb_dim=128,
            base_channels=unet_base_channels,
            channel_mult=unet_channel_mult,
            num_res_blocks=unet_num_res_blocks,
            dropout=dropout
        )
        
        # Conditioning mechanism
        if conditioning_type == "film":
            self.conditioning = MultimodalFiLM(
                feature_dim=action_dim,  # Condition on action features
                vision_dim=vision_feature_dim,
                language_dim=language_feature_dim
            )
            self.use_cross_attention = False
        elif conditioning_type == "cross_attention":
            self.conditioning = MultimodalCrossAttention(
                query_dim=action_dim,  # Query is action features
                vision_dim=vision_feature_dim,
                language_dim=language_feature_dim,
                num_heads=4,  # Reduced for smaller action_dim
                dropout=dropout
            )
            self.use_cross_attention = True
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning_type}")
        
        # Diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=True,
            clip_sample_range=1.0
        )
        
        self.num_diffusion_steps = num_diffusion_steps
        self.to(device)
    
    def encode_observations(
        self,
        images: torch.Tensor,
        instructions: Any  # Can be str, List[str], or tokenized tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Encode visual observations and language instructions.
        
        Args:
            images: [B, 3, H, W] RGB images
            instructions: Language instructions (various formats)
        Returns:
            Dictionary with 'vision' and 'language' features
        """
        # Encode vision
        vision_features = self.vision_encoder(images)  # [B, vision_feature_dim]
        
        # Encode language
        language_features = self.language_encoder(instructions)  # [B, language_feature_dim]
        
        return {
            'vision': vision_features,
            'language': language_features
        }
    
    def apply_conditioning(
        self,
        x: torch.Tensor,
        vision_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply multimodal conditioning to action features.
        
        Args:
            x: [B, action_dim, T] action features
            vision_features: [B, vision_dim] vision features
            language_features: [B, language_dim] language features
        Returns:
            [B, action_dim, T] conditioned features
        """
        # Both FiLM and CrossAttention work on [B, T, feature_dim]
        # x comes in as [B, action_dim, T], transpose to [B, T, action_dim]
        x = x.transpose(1, 2)
        x = self.conditioning(x, vision_features, language_features)
        x = x.transpose(1, 2)  # Back to [B, action_dim, T]
        
        return x
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        instructions: Any
    ) -> torch.Tensor:
        """
        Forward pass during training.
        
        Args:
            noisy_actions: [B, action_dim, horizon] noisy action sequences
            timesteps: [B,] diffusion timesteps
            images: [B, 3, H, W] RGB images
            instructions: Language instructions
        Returns:
            [B, action_dim, horizon] predicted noise or actions
        """
        # Encode observations
        obs_features = self.encode_observations(images, instructions)
        
        # Pass through U-Net
        x = self.unet(noisy_actions, timesteps)  # [B, action_dim, horizon]
        
        # Apply conditioning
        x = self.apply_conditioning(
            x,
            obs_features['vision'],
            obs_features['language']
        )
        
        return x
    
    def get_action(
        self,
        images: torch.Tensor,
        instructions: Any,
        num_inference_steps: int = 10,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate actions using DDIM sampling (inference).
        
        Args:
            images: [B, 3, H, W] RGB images
            instructions: Language instructions
            num_inference_steps: Number of denoising steps (use DDIM for speed)
            generator: Random generator for reproducibility
        Returns:
            [B, action_dim, horizon] predicted action sequences
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Encode observations once
        obs_features = self.encode_observations(images, instructions)
        
        # Initialize with random noise
        actions = torch.randn(
            batch_size,
            self.action_dim,
            self.action_horizon,
            device=device,
            generator=generator
        )
        
        # Create DDIM scheduler for faster inference
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule=self.noise_scheduler.config.beta_schedule,
            prediction_type=self.noise_scheduler.config.prediction_type,
            clip_sample=True
        )
        ddim_scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Denoising loop
        for t in ddim_scheduler.timesteps:
            # Prepare timestep
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(actions, timesteps)
                noise_pred = self.apply_conditioning(
                    noise_pred,
                    obs_features['vision'],
                    obs_features['language']
                )
            
            # Denoise step
            actions = ddim_scheduler.step(
                noise_pred,
                t,
                actions,
                generator=generator
            ).prev_sample
        
        return actions
    
    def compute_loss(
        self,
        actions: torch.Tensor,
        images: torch.Tensor,
        instructions: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.
        
        Args:
            actions: [B, action_dim, horizon] clean action sequences
            images: [B, 3, H, W] RGB images
            instructions: Language instructions
        Returns:
            Dictionary with loss and metrics
        """
        batch_size = actions.shape[0]
        device = actions.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Add noise to actions (forward diffusion)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # Predict noise
        noise_pred = self.forward(noisy_actions, timesteps, images, instructions)
        
        # Compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            loss = nn.functional.mse_loss(noise_pred, noise)
        elif self.noise_scheduler.config.prediction_type == "sample":
            loss = nn.functional.mse_loss(noise_pred, actions)
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.config.prediction_type}")
        
        return {
            'loss': loss,
            'mse': loss.item()
        }


if __name__ == "__main__":
    # Test the diffusion policy
    print("Testing Diffusion Policy...")
    
    model = DiffusionPolicy(
        action_dim=7,
        action_horizon=16,
        vision_encoder="resnet18",
        conditioning_type="cross_attention",
        num_diffusion_steps=100
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create dummy inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(model.device)
    instructions = ["Pick up the red cube", "Push the blue block"]
    actions = torch.randn(batch_size, 7, 16).to(model.device)
    
    # Test training
    print("\nTesting training...")
    loss_dict = model.compute_loss(actions, images, instructions)
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    
    # Test inference
    print("\nTesting inference...")
    predicted_actions = model.get_action(images, instructions, num_inference_steps=10)
    print(f"Predicted actions shape: {predicted_actions.shape}")
