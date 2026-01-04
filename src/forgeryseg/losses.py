from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.float()
    targets = targets.float()

    dims = tuple(range(2, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, *, bce_weight: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    return bce_weight * bce + dice_loss(logits, targets)

