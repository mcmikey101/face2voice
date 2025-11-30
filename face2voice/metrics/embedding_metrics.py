"""
Embedding metrics for face-to-voice model evaluation.

This module provides functions to compute metrics between predicted
and target voice embeddings.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Optional


def cosine_similarity(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> Union[torch.Tensor, float]:
    """
    Compute cosine similarity between predicted and target embeddings.
    
    Args:
        predicted: Predicted embeddings, shape (B, D) or (D,)
        target: Target embeddings, shape (B, D) or (D,)
        reduction: 'mean', 'none', or 'sum'. If 'mean' or 'sum', returns scalar.
    
    Returns:
        Cosine similarity score(s)
    """
    # Handle single embedding case
    if predicted.dim() == 1:
        predicted = predicted.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Normalize embeddings
    pred_norm = F.normalize(predicted, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)
    
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
    
    if reduction == 'mean':
        return cosine_sim.mean().item()
    elif reduction == 'sum':
        return cosine_sim.sum().item()
    else:
        return cosine_sim


def euclidean_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> Union[torch.Tensor, float]:
    """
    Compute Euclidean distance between predicted and target embeddings.
    
    Args:
        predicted: Predicted embeddings, shape (B, D) or (D,)
        target: Target embeddings, shape (B, D) or (D,)
        reduction: 'mean', 'none', or 'sum'. If 'mean' or 'sum', returns scalar.
    
    Returns:
        Euclidean distance(s)
    """
    # Handle single embedding case
    if predicted.dim() == 1:
        predicted = predicted.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Compute Euclidean distance
    euclidean_dist = torch.norm(predicted - target, p=2, dim=-1)
    
    if reduction == 'mean':
        return euclidean_dist.mean().item()
    elif reduction == 'sum':
        return euclidean_dist.sum().item()
    else:
        return euclidean_dist


def mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> Union[torch.Tensor, float]:
    """
    Compute Mean Squared Error between predicted and target embeddings.
    
    Args:
        predicted: Predicted embeddings, shape (B, D) or (D,)
        target: Target embeddings, shape (B, D) or (D,)
        reduction: 'mean', 'none', or 'sum'. If 'mean' or 'sum', returns scalar.
    
    Returns:
        MSE score(s)
    """
    # Handle single embedding case
    if predicted.dim() == 1:
        predicted = predicted.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Compute MSE
    mse_loss = F.mse_loss(predicted, target, reduction='none')
    
    if reduction == 'mean':
        return mse_loss.mean().item()
    elif reduction == 'sum':
        return mse_loss.sum().item()
    else:
        return mse_loss.mean(dim=-1)  # Mean over embedding dimension


def mae(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> Union[torch.Tensor, float]:
    """
    Compute Mean Absolute Error between predicted and target embeddings.
    
    Args:
        predicted: Predicted embeddings, shape (B, D) or (D,)
        target: Target embeddings, shape (B, D) or (D,)
        reduction: 'mean', 'none', or 'sum'. If 'mean' or 'sum', returns scalar.
    
    Returns:
        MAE score(s)
    """
    # Handle single embedding case
    if predicted.dim() == 1:
        predicted = predicted.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Compute MAE
    mae_loss = F.l1_loss(predicted, target, reduction='none')
    
    if reduction == 'mean':
        return mae_loss.mean().item()
    elif reduction == 'sum':
        return mae_loss.sum().item()
    else:
        return mae_loss.mean(dim=-1)  # Mean over embedding dimension


def compute_all_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    metric_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute all specified metrics between predicted and target embeddings.
    
    Args:
        predicted: Predicted embeddings, shape (B, D)
        target: Target embeddings, shape (B, D)
        metric_names: List of metric names to compute. If None, computes all.
    
    Returns:
        Dictionary with metric names as keys and values as scores
    """
    if metric_names is None:
        metric_names = ['cosine_similarity', 'euclidean_distance', 'mse', 'mae']
    
    metrics = {}
    
    for metric_name in metric_names:
        if metric_name == 'cosine_similarity':
            metrics['cosine_similarity'] = cosine_similarity(predicted, target)
        elif metric_name == 'euclidean_distance':
            metrics['euclidean_distance'] = euclidean_distance(predicted, target)
        elif metric_name == 'mse':
            metrics['mse'] = mse(predicted, target)
        elif metric_name == 'mae':
            metrics['mae'] = mae(predicted, target)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    return metrics


def compute_embedding_statistics(
    embeddings: torch.Tensor
) -> Dict[str, float]:
    """
    Compute statistics for embeddings (mean, std, min, max per dimension).
    
    Args:
        embeddings: Embeddings tensor, shape (B, D)
    
    Returns:
        Dictionary with statistics
    """
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    
    stats = {
        'mean': embeddings.mean(dim=0).cpu().numpy().tolist(),
        'std': embeddings.std(dim=0).cpu().numpy().tolist(),
        'min': embeddings.min(dim=0)[0].cpu().numpy().tolist(),
        'max': embeddings.max(dim=0)[0].cpu().numpy().tolist(),
        'mean_norm': embeddings.norm(p=2, dim=1).mean().item(),
        'std_norm': embeddings.norm(p=2, dim=1).std().item(),
    }
    
    return stats

