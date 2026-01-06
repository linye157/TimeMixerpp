"""
Loss functions for metric learning and classification.

Implements:
1. SupConLoss - Supervised Contrastive Learning loss
2. Combined loss for SupCon + BCE joint training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    From: "Supervised Contrastive Learning" (Khosla et al., 2020)
    https://arxiv.org/abs/2004.11362
    
    For each anchor, positives are samples with the same label.
    The loss encourages embeddings of same-label samples to be close,
    while pushing apart embeddings of different-label samples.
    
    Key features:
    - Handles batch-level positive pairs
    - Safe handling when no positives exist for an anchor
    - Temperature scaling for stability
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Args:
            temperature: Temperature for scaling similarities
            base_temperature: Base temperature (for scaling the loss)
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: L2-normalized embeddings, shape (B, emb_dim)
            labels: Integer or float labels, shape (B,)
            mask: Optional mask for valid samples, shape (B,)
            
        Returns:
            Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Handle edge cases
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Convert labels to integer for comparison
        # (handles both float and int labels)
        labels = labels.contiguous().view(-1)
        
        # For float labels (probabilities), threshold at 0.5
        if labels.dtype == torch.float32 or labels.dtype == torch.float64:
            labels = (labels >= 0.5).long()
        
        # Compute similarity matrix (already L2-normalized, so dot product = cosine)
        # Shape: (B, B)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Create mask for positive pairs (same label, excluding self)
        # Shape: (B, B)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        
        # Exclude self-comparisons
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = labels_equal & ~self_mask
        
        # Count positives for each anchor
        num_positives = positive_mask.sum(dim=1)  # (B,)
        
        # Handle samples with no positives (skip them)
        has_positives = num_positives > 0
        
        if not has_positives.any():
            # No valid positive pairs in this batch
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Scale similarities by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # For numerical stability, subtract max
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Mask out self-comparisons for denominator
        exp_logits = torch.exp(logits) * (~self_mask).float()
        
        # Log-sum-exp for denominator
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # Compute mean log-likelihood over positive pairs
        # Only for samples that have positives
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / (num_positives + 1e-12)
        
        # Loss (only for samples with positives)
        loss = -mean_log_prob_pos * has_positives.float()
        
        # Average over samples with positives
        loss = loss.sum() / (has_positives.sum() + 1e-12)
        
        # Scale by temperature ratio
        loss = loss * (self.temperature / self.base_temperature)
        
        return loss


class CombinedSupConBCELoss(nn.Module):
    """
    Combined loss: SupCon + BCE for joint metric and classification learning.
    
    loss = supcon_loss + lambda_bce * bce_loss
    
    The BCE component provides direct classification supervision,
    while SupCon encourages well-clustered embeddings.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        lambda_bce: float = 0.5,
        pos_weight: Optional[float] = None
    ):
        """
        Args:
            temperature: SupCon temperature
            lambda_bce: Weight for BCE loss
            pos_weight: Positive class weight for BCE (for imbalanced data)
        """
        super().__init__()
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.lambda_bce = lambda_bce
        
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            embeddings: L2-normalized embeddings (B, emb_dim)
            labels: Labels (B,)
            logits: Classification logits (B, 1), required if lambda_bce > 0
            
        Returns:
            Dictionary with total loss and components
        """
        # SupCon loss
        supcon = self.supcon_loss(embeddings, labels)
        
        result = {
            'supcon': supcon,
            'total': supcon
        }
        
        # BCE loss
        if self.lambda_bce > 0 and logits is not None:
            # Ensure labels shape matches logits
            labels_bce = labels.view(-1, 1).float()
            
            # Handle device mismatch for pos_weight
            if hasattr(self.bce_loss, 'pos_weight') and self.bce_loss.pos_weight is not None:
                self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(logits.device)
            
            bce = self.bce_loss(logits, labels_bce)
            result['bce'] = bce
            result['total'] = supcon + self.lambda_bce * bce
        
        return result


class MultiScaleSupConLoss(nn.Module):
    """
    Multi-scale SupCon loss with configurable scale weights.
    
    loss = w0 * SupCon(e0) + w1 * SupCon(e1) + w2 * SupCon(e2)
    
    Optionally includes BCE loss for each scale.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        scale_weights: tuple = (0.5, 0.3, 0.2),
        use_bce: bool = False,
        lambda_bce: float = 0.5,
        pos_weight: Optional[float] = None
    ):
        """
        Args:
            temperature: SupCon temperature
            scale_weights: Weights for each scale (w0, w1, w2), will be normalized
            use_bce: Whether to include BCE loss
            lambda_bce: Weight for BCE loss
            pos_weight: Positive class weight for BCE
        """
        super().__init__()
        
        self.supcon_loss = SupConLoss(temperature=temperature)
        
        # Normalize scale weights
        total = sum(scale_weights)
        self.scale_weights = tuple(w / total for w in scale_weights)
        
        self.use_bce = use_bce
        self.lambda_bce = lambda_bce
        
        if use_bce:
            if pos_weight is not None:
                self.bce_loss = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pos_weight])
                )
            else:
                self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        e0: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
        labels: torch.Tensor,
        logits0: Optional[torch.Tensor] = None,
        logits1: Optional[torch.Tensor] = None,
        logits2: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute multi-scale loss.
        
        Args:
            e0, e1, e2: Embeddings for each scale (B, emb_dim)
            labels: Labels (B,)
            logits0, logits1, logits2: Classification logits for each scale (B, 1)
            
        Returns:
            Dictionary with total loss and components
        """
        w0, w1, w2 = self.scale_weights
        
        # SupCon for each scale
        supcon0 = self.supcon_loss(e0, labels)
        supcon1 = self.supcon_loss(e1, labels)
        supcon2 = self.supcon_loss(e2, labels)
        
        supcon_total = w0 * supcon0 + w1 * supcon1 + w2 * supcon2
        
        result = {
            'supcon0': supcon0,
            'supcon1': supcon1,
            'supcon2': supcon2,
            'supcon': supcon_total,
            'total': supcon_total
        }
        
        # BCE for each scale (if enabled)
        if self.use_bce and logits0 is not None:
            labels_bce = labels.view(-1, 1).float()
            
            # Handle device
            if hasattr(self.bce_loss, 'pos_weight') and self.bce_loss.pos_weight is not None:
                self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(logits0.device)
            
            bce0 = self.bce_loss(logits0, labels_bce)
            bce1 = self.bce_loss(logits1, labels_bce)
            bce2 = self.bce_loss(logits2, labels_bce)
            
            bce_total = w0 * bce0 + w1 * bce1 + w2 * bce2
            
            result['bce0'] = bce0
            result['bce1'] = bce1
            result['bce2'] = bce2
            result['bce'] = bce_total
            result['total'] = supcon_total + self.lambda_bce * bce_total
        
        return result

