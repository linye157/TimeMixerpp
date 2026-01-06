"""
Metric Learning Encoder for Sequence-level Embeddings.

This module implements a TemporalConvEmbedder that:
1. Takes multi-scale features (B, L, 64) and produces sequence embeddings (B, emb_dim)
2. Uses 3-layer Conv1d with attention pooling (NOT simple mean pooling)
3. Outputs L2-normalized embeddings for metric learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns which timesteps are important.
    
    Unlike simple mean pooling, this preserves temporal importance information.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, hidden_dim)
            
        Returns:
            pooled: (B, hidden_dim) - attention-weighted pooled representation
            alpha: (B, L) - attention weights for interpretability
        """
        # Compute attention scores
        scores = self.attention(x)  # (B, L, 1)
        alpha = F.softmax(scores, dim=1)  # (B, L, 1)
        
        # Weighted sum
        pooled = (alpha * x).sum(dim=1)  # (B, hidden_dim)
        
        return pooled, alpha.squeeze(-1)


class TemporalConvEmbedder(nn.Module):
    """
    Temporal Convolutional Embedder for sequence-level representations.
    
    Architecture:
    1. Input: (B, L, 64) -> transpose to (B, 64, L) for Conv1d
    2. 3-layer Conv1d (kernel=3, pad=1) with GELU and Dropout
    3. Transpose back to (B, L, hidden) for attention pooling
    4. Attention pooling: (B, L, hidden) -> (B, hidden)
    5. Projection MLP: hidden -> emb_dim
    6. L2 normalize output embedding
    
    Optional classification head for BCE joint training.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        emb_dim: int = 128,
        dropout: float = 0.1,
        use_classification_head: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.use_classification_head = use_classification_head
        
        # 3-layer Conv1d backbone
        self.conv_layers = nn.Sequential(
            # Layer 1: input_dim -> hidden_dim
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Layer 2: hidden_dim -> hidden_dim
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Layer 3: hidden_dim -> hidden_dim
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Attention pooling (必须，不能用简单 mean pooling)
        self.attention_pool = AttentionPooling(hidden_dim, dropout)
        
        # Projection MLP: hidden -> emb_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        
        # Optional classification head
        if use_classification_head:
            self.classifier = nn.Linear(emb_dim, 1)
        else:
            self.classifier = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (B, L, 64)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - 'embedding': L2-normalized embedding (B, emb_dim)
                - 'logits': Classification logits (B, 1) if use_classification_head
                - 'attention': Attention weights (B, L) if return_attention
                - 'top_timesteps': Top-3 attended timestep indices (B, 3) if return_attention
        """
        B, L, D = x.shape
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, got {D}"
        
        # Transpose for Conv1d: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)  # (B, 64, L)
        
        # 3-layer Conv1d
        x = self.conv_layers(x)  # (B, hidden_dim, L)
        
        # Transpose back: (B, hidden_dim, L) -> (B, L, hidden_dim)
        x = x.transpose(1, 2)  # (B, L, hidden_dim)
        
        # Attention pooling
        pooled, alpha = self.attention_pool(x)  # pooled: (B, hidden), alpha: (B, L)
        
        # Projection
        embedding = self.projection(pooled)  # (B, emb_dim)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        result = {'embedding': embedding}
        
        # Classification head
        if self.classifier is not None:
            logits = self.classifier(embedding)  # (B, 1)
            result['logits'] = logits
            result['probs'] = torch.sigmoid(logits)
        
        # Attention info
        if return_attention:
            result['attention'] = alpha
            # Top-3 attended timesteps
            _, top_indices = alpha.topk(min(3, L), dim=1)
            result['top_timesteps'] = top_indices
        
        return result
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get only the embedding (convenience method).
        
        Args:
            x: Input tensor, shape (B, L, 64)
            
        Returns:
            L2-normalized embedding (B, emb_dim)
        """
        return self.forward(x)['embedding']


class MultiScaleEmbedder(nn.Module):
    """
    Wrapper that applies TemporalConvEmbedder to multiple scales.
    
    Shares the same encoder across all scales (parameter efficient).
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        emb_dim: int = 128,
        dropout: float = 0.1,
        use_classification_head: bool = False
    ):
        super().__init__()
        
        # Single shared encoder
        self.encoder = TemporalConvEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            dropout=dropout,
            use_classification_head=use_classification_head
        )
        
        # Learnable fusion weights (for learned fusion mode)
        self.fusion_logits = nn.Parameter(torch.zeros(3))  # For 3 scales
    
    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass for 3 scales.
        
        Args:
            x0: Scale 0 features (B, 48, 64)
            x1: Scale 1 features (B, 24, 64)
            x2: Scale 2 features (B, 12, 64)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with embeddings and optionally logits/attention for each scale
        """
        out0 = self.encoder(x0, return_attention=return_attention)
        out1 = self.encoder(x1, return_attention=return_attention)
        out2 = self.encoder(x2, return_attention=return_attention)
        
        result = {
            'e0': out0['embedding'],
            'e1': out1['embedding'],
            'e2': out2['embedding'],
        }
        
        if 'logits' in out0:
            result['logits0'] = out0['logits']
            result['logits1'] = out1['logits']
            result['logits2'] = out2['logits']
        
        if return_attention:
            result['attn0'] = out0['attention']
            result['attn1'] = out1['attention']
            result['attn2'] = out2['attention']
            result['top_timesteps0'] = out0['top_timesteps']
            result['top_timesteps1'] = out1['top_timesteps']
            result['top_timesteps2'] = out2['top_timesteps']
        
        # Fusion weights
        result['fusion_weights'] = F.softmax(self.fusion_logits, dim=0)
        
        return result
    
    def get_fusion_weights(self) -> torch.Tensor:
        """Get learned fusion weights as probabilities."""
        return F.softmax(self.fusion_logits, dim=0)

