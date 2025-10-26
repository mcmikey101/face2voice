import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple

class CompoundLoss(nn.Module):
    """
    Combined loss function for training face-to-voice mapping.
    
    Combines:
    1. Cosine similarity loss (alignment)
    2. MSE loss (magnitude)
    3. Contrastive loss (discrimination)
    4. Distribution alignment loss
    """
    
    def __init__(
        self,
        cosine_weight: float = 1.0,
        mse_weight: float = 0.5,
        contrastive_weight: float = 0.5,
        distribution_weight: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.contrastive_weight = contrastive_weight
        self.distribution_weight = distribution_weight
        self.temperature = temperature
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            predicted: Predicted voice embeddings, shape (B, D)
            target: Target voice embeddings, shape (B, D)
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        batch_size = predicted.shape[0]
        
        # 1. Cosine similarity loss
        pred_norm = F.normalize(predicted, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
        cosine_loss = (1 - cosine_sim).mean()
        
        # 2. MSE loss
        mse_loss = F.mse_loss(predicted, target)
        
        # 3. Contrastive loss (InfoNCE)
        logits = torch.matmul(pred_norm, target_norm.T) / self.temperature
        labels = torch.arange(batch_size, device=predicted.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        
        # 4. Distribution alignment loss
        pred_mean = predicted.mean(dim=0)
        target_mean = target.mean(dim=0)
        pred_std = predicted.std(dim=0)
        target_std = target.std(dim=0)
        
        distribution_loss = (
            F.mse_loss(pred_mean, target_mean) +
            F.mse_loss(pred_std, target_std)
        )
        
        # Combined loss
        total_loss = (
            self.cosine_weight * cosine_loss +
            self.mse_weight * mse_loss +
            self.contrastive_weight * contrastive_loss +
            self.distribution_weight * distribution_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'cosine': cosine_loss.item(),
            'mse': mse_loss.item(),
            'contrastive': contrastive_loss.item(),
            'distribution': distribution_loss.item()
        }
        
        return total_loss, loss_dict