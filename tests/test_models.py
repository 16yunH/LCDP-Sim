"""
Unit tests for model components
"""

import torch
import pytest
from lcdp.models.vision_encoder import VisionEncoder
from lcdp.models.language_encoder import LanguageEncoder
from lcdp.models.unet1d import UNet1D
from lcdp.models.conditioning import FiLM, CrossAttention
from lcdp.models.diffusion_policy import DiffusionPolicy


class TestVisionEncoder:
    """Test vision encoder"""
    
    def test_resnet_encoder(self):
        encoder = VisionEncoder(
            encoder_type="resnet18",
            output_dim=512
        )
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        
        output = encoder(images)
        
        assert output.shape == (batch_size, 512)
        assert not torch.isnan(output).any()
    
    def test_spatial_softmax(self):
        encoder = VisionEncoder(
            encoder_type="resnet18",
            output_dim=512,
            use_spatial_softmax=True
        )
        
        images = torch.randn(2, 3, 224, 224)
        output = encoder(images)
        
        assert output.shape == (2, 512)


class TestLanguageEncoder:
    """Test language encoder"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CLIP requires CUDA")
    def test_clip_encoder(self):
        encoder = LanguageEncoder(
            model_name="ViT-B/32",
            output_dim=512,
            device="cuda"
        )
        
        instructions = ["Pick up the cube", "Push the block"]
        output = encoder(instructions)
        
        assert output.shape == (2, 512)
        assert not torch.isnan(output).any()


class TestUNet1D:
    """Test 1D U-Net"""
    
    def test_forward_pass(self):
        model = UNet1D(
            input_dim=7,
            output_dim=7,
            time_emb_dim=128,
            base_channels=64,
            channel_mult=(1, 2),
            num_res_blocks=1
        )
        
        batch_size = 4
        horizon = 16
        
        actions = torch.randn(batch_size, 7, horizon)
        timesteps = torch.randint(0, 100, (batch_size,))
        
        output = model(actions, timesteps)
        
        assert output.shape == (batch_size, 7, horizon)
        assert not torch.isnan(output).any()
    
    def test_different_horizons(self):
        model = UNet1D(input_dim=7, output_dim=7)
        
        for horizon in [8, 16, 32]:
            actions = torch.randn(2, 7, horizon)
            timesteps = torch.randint(0, 100, (2,))
            output = model(actions, timesteps)
            assert output.shape == (2, 7, horizon)


class TestConditioning:
    """Test conditioning mechanisms"""
    
    def test_film(self):
        film = FiLM(feature_dim=256, condition_dim=512)
        
        features = torch.randn(4, 16, 256)
        condition = torch.randn(4, 512)
        
        output = film(features, condition)
        
        assert output.shape == features.shape
        assert not torch.isnan(output).any()
    
    def test_cross_attention(self):
        cross_attn = CrossAttention(
            query_dim=256,
            context_dim=512,
            num_heads=8
        )
        
        query = torch.randn(4, 16, 256)
        context = torch.randn(4, 2, 512)
        
        output = cross_attn(query, context)
        
        assert output.shape == query.shape
        assert not torch.isnan(output).any()


class TestDiffusionPolicy:
    """Test complete diffusion policy"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_forward_pass(self):
        model = DiffusionPolicy(
            action_dim=7,
            action_horizon=16,
            conditioning_type="film",  # Use FiLM to avoid CLIP dependency
            device="cpu"  # Use CPU for testing
        )
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        actions = torch.randn(batch_size, 7, 16)
        timesteps = torch.randint(0, 100, (batch_size,))
        instructions = ["Test instruction 1", "Test instruction 2"]
        
        output = model(actions, timesteps, images, instructions)
        
        assert output.shape == actions.shape
        assert not torch.isnan(output).any()
    
    def test_compute_loss(self):
        model = DiffusionPolicy(
            action_dim=7,
            action_horizon=16,
            conditioning_type="film",
            device="cpu"
        )
        
        images = torch.randn(2, 3, 224, 224)
        actions = torch.randn(2, 7, 16)
        instructions = ["Instruction 1", "Instruction 2"]
        
        loss_dict = model.compute_loss(actions, images, instructions)
        
        assert 'loss' in loss_dict
        assert loss_dict['loss'].item() >= 0
        assert not torch.isnan(loss_dict['loss'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
