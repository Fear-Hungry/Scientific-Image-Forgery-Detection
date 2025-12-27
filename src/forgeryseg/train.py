from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrainStats:
    loss: float


@torch.no_grad()
def _batch_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return dice.mean()


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool = False) -> TrainStats:
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)

    return TrainStats(loss=total_loss / max(len(loader.dataset), 1))


def validate(model, loader, criterion, device) -> tuple[TrainStats, float]:
    model.eval()
    total_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += float(loss.item()) * images.size(0)
            dice_scores.append(float(_batch_dice(logits, masks)))

    mean_loss = total_loss / max(len(loader.dataset), 1)
    mean_dice = float(sum(dice_scores) / max(len(dice_scores), 1))
    return TrainStats(loss=mean_loss), mean_dice
