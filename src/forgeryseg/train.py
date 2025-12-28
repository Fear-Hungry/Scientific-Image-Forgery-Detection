from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrainStats:
    loss: float


def _ensure_no_empty_parameters(model) -> None:
    """
    Fail fast on invalid models that contain parameters with zero elements.

    This can happen when a segmentation model is built with `classes=0` or
    `decoder_channels` containing 0, which otherwise triggers cryptic runtime
    errors inside conv layers.
    """
    if getattr(model, "_forgeryseg_checked_params", False):
        return
    empty = [name for name, p in model.named_parameters() if p is not None and p.numel() == 0]
    if empty:
        shown = ", ".join(empty[:5])
        suffix = "" if len(empty) <= 5 else f" (+{len(empty) - 5} more)"
        raise ValueError(
            "Model has parameter tensor(s) with zero elements "
            f"({shown}{suffix}). Check your model config (e.g., `classes` or `decoder_channels`)."
        )
    setattr(model, "_forgeryseg_checked_params", True)


def _maybe_tqdm(iterable, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:  # pragma: no cover - optional dependency
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, leave=False)
    except Exception:
        return iterable


def _autocast_ctx(device: str, enabled: bool):
    """
    Compatibility helper for AMP autocast across torch versions.

    `torch.cuda.amp.autocast` is deprecated in recent torch; prefer `torch.amp.autocast`.
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        return torch.amp.autocast(device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(device: str, enabled: bool):
    """
    Compatibility helper for AMP GradScaler across torch versions.

    `torch.cuda.amp.GradScaler` is deprecated in recent torch; prefer `torch.amp.GradScaler`.
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        return torch.amp.GradScaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_amp: bool = False,
    progress: bool = True,
    desc: str = "train",
) -> TrainStats:
    model.train()
    _ensure_no_empty_parameters(model)
    total_loss = 0.0
    scaler = _grad_scaler(device, enabled=use_amp)

    for images, masks in _maybe_tqdm(loader, progress, desc):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)

    return TrainStats(loss=total_loss / max(len(loader.dataset), 1))


def validate(
    model,
    loader,
    criterion,
    device,
    progress: bool = True,
    desc: str = "val",
) -> tuple[TrainStats, float]:
    model.eval()
    _ensure_no_empty_parameters(model)
    total_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks in _maybe_tqdm(loader, progress, desc):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += float(loss.item()) * images.size(0)
            dice_scores.append(float(_batch_dice(logits, masks)))

    mean_loss = total_loss / max(len(loader.dataset), 1)
    mean_dice = float(sum(dice_scores) / max(len(dice_scores), 1))
    return TrainStats(loss=mean_loss), mean_dice
