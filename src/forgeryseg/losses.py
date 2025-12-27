from __future__ import annotations

import torch
from torch import nn


def _flatten_binary_scores(logits: torch.Tensor, targets: torch.Tensor):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    return probs, targets


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, targets = _flatten_binary_scores(logits, targets)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, targets = _flatten_binary_scores(logits, targets)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class BCETverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, tversky_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.tversky_weight * self.tversky(logits, targets)
