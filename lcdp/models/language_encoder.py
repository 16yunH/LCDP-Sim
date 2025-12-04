"""
Language Encoder Module
Provides CLIP-based text encoding for natural language instructions.
"""

import torch
import torch.nn as nn
from typing import List, Union
import clip


class LanguageEncoder(nn.Module):
    """
    CLIP-based language encoder for processing natural language instructions.
    The encoder is frozen by default to leverage pre-trained knowledge.
    """
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        freeze: bool = True,
        output_dim: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_name: CLIP model variant ("RN50", "RN101", "ViT-B/32", "ViT-B/16", "ViT-L/14")
            freeze: Whether to freeze CLIP parameters
            output_dim: Output feature dimension
            device: Device to load model on
        """
        super(LanguageEncoder, self).__init__()
        
        # Load pretrained CLIP model
        self.clip_model, self.preprocess = clip.load(model_name, device=device)
        
        # Get CLIP text feature dimension
        self.clip_dim = self.clip_model.text_projection.shape[1]
        
        # Freeze CLIP parameters if needed
        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Project CLIP features to desired output dimension
        if output_dim != self.clip_dim:
            self.projection = nn.Linear(self.clip_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        self.output_dim = output_dim
        self.device = device
    
    def tokenize(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenize text inputs using CLIP's tokenizer.
        
        Args:
            texts: Single string or list of strings
        Returns:
            Tokenized text tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        return clip.tokenize(texts).to(self.device)
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode tokenized text using CLIP.
        
        Args:
            text_tokens: [B, context_length] tokenized text
        Returns:
            [B, clip_dim] text features
        """
        with torch.set_grad_enabled(self.training and not self._is_frozen()):
            text_features = self.clip_model.encode_text(text_tokens)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()
    
    def _is_frozen(self) -> bool:
        """Check if CLIP parameters are frozen."""
        return not next(self.clip_model.parameters()).requires_grad
    
    def forward(self, texts: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through language encoder.
        
        Args:
            texts: Either raw text strings/list or pre-tokenized tensor
        Returns:
            [B, output_dim] language features
        """
        # Tokenize if input is text
        if isinstance(texts, (str, list)):
            text_tokens = self.tokenize(texts)
        else:
            text_tokens = texts
        
        # Encode with CLIP
        clip_features = self.encode_text(text_tokens)
        
        # Project to output dimension
        features = self.projection(clip_features)
        
        return features


class SimpleLanguageEncoder(nn.Module):
    """
    A simpler language encoder using learnable embeddings for a fixed vocabulary.
    Useful for debugging or when CLIP is not available.
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 512,
        max_length: int = 50
    ):
        super(SimpleLanguageEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.output_dim = output_dim
        self.max_length = max_length
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] token indices
        Returns:
            [B, output_dim] language features
        """
        # Embed tokens: [B, L, embedding_dim]
        embedded = self.embedding(token_ids)
        
        # LSTM encoding: [B, L, hidden_dim*2]
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Use final hidden state: [B, hidden_dim*2]
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Project to output: [B, output_dim]
        output = self.fc(final_hidden)
        
        return output


if __name__ == "__main__":
    # Test CLIP-based encoder
    print("Testing CLIP Language Encoder...")
    encoder = LanguageEncoder(
        model_name="ViT-B/32",
        freeze=True,
        output_dim=512
    )
    
    # Test with sample instructions
    instructions = [
        "Pick up the red cube",
        "Push the blue block to the left",
        "Stack the green cube on top of the red one"
    ]
    
    # Forward pass
    features = encoder(instructions)
    print(f"Input: {len(instructions)} instructions")
    print(f"Output shape: {features.shape}")
    print(f"Output dim: {encoder.output_dim}")
    
    # Test with different instruction
    single_instruction = "Grasp the object"
    single_feature = encoder(single_instruction)
    print(f"\nSingle instruction output shape: {single_feature.shape}")
