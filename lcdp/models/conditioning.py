"""
Conditioning Mechanisms for Diffusion Policy
Implements FiLM (Feature-wise Linear Modulation) and Cross-Attention for multimodal fusion.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies affine transformation to features conditioned on context.
    
    FiLM(x, c) = γ(c) ⊙ x + β(c)
    where γ and β are learned functions of the conditioning vector c.
    """
    def __init__(self, feature_dim: int, condition_dim: int):
        """
        Args:
            feature_dim: Dimension of features to be modulated
            condition_dim: Dimension of conditioning vector
        """
        super(FiLM, self).__init__()
        
        # Networks to predict scale (gamma) and shift (beta)
        self.scale_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.shift_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, feature_dim] or [B, feature_dim] features to modulate
            condition: [B, condition_dim] conditioning vector
        Returns:
            Modulated features with same shape as input features
        """
        gamma = self.scale_net(condition)  # [B, feature_dim]
        beta = self.shift_net(condition)   # [B, feature_dim]
        
        # Handle both 2D and 3D tensors
        if features.dim() == 3:
            # features is [B, T, feature_dim], expand gamma/beta to [B, 1, feature_dim]
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        # Apply FiLM
        modulated = gamma * features + beta
        
        return modulated


class MultimodalFiLM(nn.Module):
    """
    FiLM layer that handles multiple conditioning modalities (vision + language).
    """
    def __init__(self, feature_dim: int, vision_dim: int, language_dim: int):
        super(MultimodalFiLM, self).__init__()
        
        # Combine vision and language features
        combined_dim = vision_dim + language_dim
        self.fusion = nn.Linear(combined_dim, feature_dim)
        
        # FiLM modulation
        self.film = FiLM(feature_dim, feature_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        vision_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, feature_dim] features to modulate
            vision_features: [B, vision_dim] vision embedding
            language_features: [B, language_dim] language embedding
        Returns:
            [B, T, feature_dim] modulated features
        """
        # Concatenate multimodal features
        multimodal = torch.cat([vision_features, language_features], dim=-1)  # [B, combined_dim]
        
        # Fuse and condition
        condition = self.fusion(multimodal)  # [B, feature_dim]
        modulated = self.film(features, condition)
        
        return modulated


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for conditioning on multimodal context.
    Allows the model to attend to relevant parts of the conditioning information.
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of query features (action features)
            context_dim: Dimension of context features (vision + language)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossAttention, self).__init__()
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, T_q, query_dim] query features (e.g., noisy actions)
            context: [B, T_c, context_dim] context features (e.g., vision + language)
            mask: Optional [B, T_q, T_c] attention mask
        Returns:
            [B, T_q, query_dim] attended features
        """
        B, T_q, _ = query.shape
        T_c = context.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query)    # [B, T_q, query_dim]
        K = self.k_proj(context)  # [B, T_c, query_dim]
        V = self.v_proj(context)  # [B, T_c, query_dim]
        
        # Reshape for multi-head attention: [B, num_heads, T, head_dim]
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_c, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_c, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: [B, num_heads, T_q, T_c]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: [B, num_heads, T_q, head_dim]
        attended = torch.matmul(attn_weights, V)
        
        # Reshape back: [B, T_q, query_dim]
        attended = attended.transpose(1, 2).contiguous().view(B, T_q, -1)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output


class MultimodalCrossAttention(nn.Module):
    """
    Cross-Attention module that attends to both vision and language features.
    """
    def __init__(
        self,
        query_dim: int,
        vision_dim: int,
        language_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(MultimodalCrossAttention, self).__init__()
        
        # Project vision and language to common dimension
        self.vision_proj = nn.Linear(vision_dim, query_dim)
        self.language_proj = nn.Linear(language_dim, query_dim)
        
        # Cross-attention
        self.cross_attn = CrossAttention(
            query_dim=query_dim,
            context_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(query_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        vision_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: [B, T, query_dim] query features
            vision_features: [B, vision_dim] vision embedding
            language_features: [B, language_dim] language embedding
        Returns:
            [B, T, query_dim] attended features
        """
        B = query.shape[0]
        
        # Project modalities: [B, 1, query_dim]
        vision_proj = self.vision_proj(vision_features).unsqueeze(1)
        language_proj = self.language_proj(language_features).unsqueeze(1)
        
        # Concatenate as context: [B, 2, query_dim]
        context = torch.cat([vision_proj, language_proj], dim=1)
        
        # Apply cross-attention
        attended = self.cross_attn(query, context)
        
        # Residual connection + layer norm
        output = self.norm(query + attended)
        
        return output


if __name__ == "__main__":
    # Test FiLM
    print("Testing FiLM...")
    feature_dim = 256
    condition_dim = 512
    batch_size = 4
    time_steps = 16
    
    film = FiLM(feature_dim, condition_dim)
    features = torch.randn(batch_size, time_steps, feature_dim)
    condition = torch.randn(batch_size, condition_dim)
    
    modulated = film(features, condition)
    print(f"FiLM input: {features.shape}, condition: {condition.shape}")
    print(f"FiLM output: {modulated.shape}\n")
    
    # Test Cross-Attention
    print("Testing Cross-Attention...")
    query_dim = 256
    context_dim = 512
    
    cross_attn = CrossAttention(query_dim, context_dim, num_heads=8)
    query = torch.randn(batch_size, time_steps, query_dim)
    context = torch.randn(batch_size, 2, context_dim)  # 2 context tokens (vision + language)
    
    attended = cross_attn(query, context)
    print(f"Cross-Attn query: {query.shape}, context: {context.shape}")
    print(f"Cross-Attn output: {attended.shape}")
