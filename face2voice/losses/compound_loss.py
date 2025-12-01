import torch
import torch.nn as nn
import torch.nn.functional as F


class CompoundLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, cosine_weight = 1, nce_weight = 0.25):
        super().__init__()
        self.temperature = temperature
        self.cosine_weight = cosine_weight
        self.nce_weight = nce_weight

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        pred = F.normalize(predicted, p=2, dim=1)
        tgt = F.normalize(target, p=2, dim=1)

        batch = pred.size(0)

        pos = (pred * tgt).sum(dim=1)

        logits = pred @ tgt.t() / self.temperature
        labels = torch.arange(batch, device=pred.device)

        nce = F.cross_entropy(logits, labels)

        cos = 1 - pos.mean()

        total = cos * self.cosine_weight + nce * self.nce_weight

        return total, {
            "total": total.item(),
            "cosine": cos.item(),
            "info_nce": nce.item(),
        }