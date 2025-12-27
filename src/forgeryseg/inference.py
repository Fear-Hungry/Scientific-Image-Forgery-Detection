from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def to_numpy(array: Any) -> np.ndarray:
    """Convert torch tensors or numpy arrays into numpy arrays."""
    if isinstance(array, np.ndarray):
        return array
    try:  # torch tensor support
        import torch

        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
    except ImportError:
        pass
    raise TypeError("Unsupported array type for conversion to numpy")


def load_prediction(path: str | Path) -> np.ndarray:
    """Load a saved prediction array from disk (numpy .npy)."""
    path = Path(path)
    return np.load(path)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_image(image: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (image - mean) / std


def _tile_coords(length: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_size must be larger than overlap")
    if length <= tile_size:
        return [(0, tile_size)]
    coords = list(range(0, length - tile_size + 1, stride))
    if coords[-1] != length - tile_size:
        coords.append(length - tile_size)
    return [(start, start + tile_size) for start in coords]


def _pad_image(image: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    return padded, (pad_h, pad_w)


def _predict_tensor(model, tensor, device):
    import torch

    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
    return probs


def predict_image(
    model,
    image: np.ndarray,
    device,
    tile_size: int = 0,
    overlap: int = 0,
    max_size: int = 0,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    orig_h, orig_w = image.shape[:2]

    if tile_size and tile_size > 0:
        padded, _ = _pad_image(image, tile_size, tile_size)
        pad_h, pad_w = padded.shape[0], padded.shape[1]
        pred_sum = np.zeros((pad_h, pad_w), dtype=np.float32)
        pred_count = np.zeros((pad_h, pad_w), dtype=np.float32)

        ys = _tile_coords(padded.shape[0], tile_size, overlap)
        xs = _tile_coords(padded.shape[1], tile_size, overlap)
        for y0, y1 in ys:
            for x0, x1 in xs:
                tile = padded[y0:y1, x0:x1]
                tile_norm = normalize_image(tile, mean=mean, std=std)
                tile_tensor = torch.from_numpy(tile_norm).permute(2, 0, 1).unsqueeze(0)
                probs = _predict_tensor(model, tile_tensor, device)
                prob_tile = probs.squeeze(0).squeeze(0).cpu().numpy()
                pred_sum[y0:y1, x0:x1] += prob_tile
                pred_count[y0:y1, x0:x1] += 1.0

        pred = pred_sum / np.maximum(pred_count, 1.0)
        return pred[:orig_h, :orig_w]

    image_norm = normalize_image(image, mean=mean, std=std)
    tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
    if max_size and max(orig_h, orig_w) > max_size:
        scale = max_size / float(max(orig_h, orig_w))
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    probs = _predict_tensor(model, tensor, device)
    if probs.shape[-2:] != (orig_h, orig_w):
        probs = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return probs.squeeze(0).squeeze(0).cpu().numpy()
