from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .image import LetterboxMeta, letterbox_reflect, unletterbox
from .postprocess import PostprocessParams, postprocess_prob
from .rle import masks_to_annotation
from .tta import HFlipTTA, IdentityTTA, TTATransform, ZoomOutTTA, predict_with_tta


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


@dataclass(frozen=True)
class TilingParams:
    tile_size: int
    overlap: int = 0
    batch_size: int = 4


def _tile_positions(length: int, *, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, step))
    last = length - tile
    if pos[-1] != last:
        pos.append(last)
    return pos


def _feather_weights_1d(length: int, *, start: bool, end: bool, overlap: int) -> np.ndarray:
    w = np.ones(int(length), dtype=np.float32)
    o = int(min(max(0, overlap), length))
    if o <= 0:
        return w

    if start:
        ramp_in = np.linspace(0.0, 1.0, o, endpoint=False, dtype=np.float32)
        w[:o] *= ramp_in
    if end:
        ramp_out = np.linspace(1.0, 0.0, o, endpoint=False, dtype=np.float32)
        w[-o:] *= ramp_out
    return w


def _tile_weights(
    *,
    tile_h: int,
    tile_w: int,
    y0: int,
    x0: int,
    full_h: int,
    full_w: int,
    overlap: int,
) -> np.ndarray:
    wy = _feather_weights_1d(
        tile_h,
        start=(y0 > 0),
        end=(y0 + tile_h < full_h),
        overlap=overlap,
    )
    wx = _feather_weights_1d(
        tile_w,
        start=(x0 > 0),
        end=(x0 + tile_w < full_w),
        overlap=overlap,
    )
    return wy[:, None] * wx[None, :]


@torch.no_grad()
def predict_prob_map(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    input_size: int,
    device: torch.device,
    tta_transforms: list[TTATransform] | None = None,
    tta_weights: list[float] | None = None,
) -> np.ndarray:
    padded, meta = letterbox_reflect(image, input_size)
    x = torch.from_numpy(padded).permute(2, 0, 1).contiguous().float() / 255.0
    x = x.unsqueeze(0).to(device)

    if tta_transforms is None:
        tta_transforms = [IdentityTTA()]
        tta_weights = [1.0]

    model = model.to(device)
    model.eval()
    prob = predict_with_tta(model, x, transforms=tta_transforms, weights=tta_weights)[0, 0]
    prob_np = prob.detach().cpu().numpy().astype(np.float32)
    prob_orig = unletterbox(prob_np, meta)
    return prob_orig


@torch.no_grad()
def predict_prob_map_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    input_size: int,
    device: torch.device,
    tiling: TilingParams,
    tta_transforms: list[TTATransform] | None = None,
    tta_weights: list[float] | None = None,
) -> np.ndarray:
    if tiling.tile_size <= 0:
        raise ValueError("tiling.tile_size must be > 0")
    if tiling.overlap < 0:
        raise ValueError("tiling.overlap must be >= 0")
    if tiling.overlap >= tiling.tile_size:
        raise ValueError("tiling.overlap must be < tiling.tile_size")
    step = int(tiling.tile_size - tiling.overlap)
    if step <= 0:
        raise ValueError("tiling.tile_size - tiling.overlap must be > 0")

    if tta_transforms is None:
        tta_transforms = [IdentityTTA()]
        tta_weights = [1.0]

    model = model.to(device)
    model.eval()

    h, w = image.shape[:2]
    ys = _tile_positions(h, tile=int(tiling.tile_size), step=step)
    xs = _tile_positions(w, tile=int(tiling.tile_size), step=step)

    prob_sum = np.zeros((h, w), dtype=np.float32)
    w_sum = np.zeros((h, w), dtype=np.float32)

    tiles: list[tuple[int, int, np.ndarray, LetterboxMeta]] = []
    for y0 in ys:
        for x0 in xs:
            y1 = min(h, int(y0 + tiling.tile_size))
            x1 = min(w, int(x0 + tiling.tile_size))
            tile = image[int(y0) : int(y1), int(x0) : int(x1)]
            padded, meta = letterbox_reflect(tile, int(input_size))
            tiles.append((int(y0), int(x0), padded, meta))

    batch_size = int(max(1, tiling.batch_size))
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i : i + batch_size]
        x = torch.stack(
            [torch.from_numpy(pad).permute(2, 0, 1).contiguous().float() / 255.0 for _, _, pad, _ in batch],
            dim=0,
        ).to(device)
        prob_batch = predict_with_tta(model, x, transforms=tta_transforms, weights=tta_weights)[:, 0]
        prob_np = prob_batch.detach().cpu().numpy().astype(np.float32)

        for (y0, x0, _, meta), prob_tile in zip(batch, prob_np, strict=True):
            prob_tile = unletterbox(prob_tile, meta)
            tile_h, tile_w = prob_tile.shape[:2]
            ww = _tile_weights(
                tile_h=tile_h,
                tile_w=tile_w,
                y0=y0,
                x0=x0,
                full_h=h,
                full_w=w,
                overlap=int(tiling.overlap),
            )
            prob_sum[y0 : y0 + tile_h, x0 : x0 + tile_w] += prob_tile * ww
            w_sum[y0 : y0 + tile_h, x0 : x0 + tile_w] += ww

    return prob_sum / np.maximum(w_sum, 1e-6)


def default_tta(
    *,
    zoom_scale: float = 0.9,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> tuple[list[TTATransform], list[float]]:
    transforms: list[TTATransform] = [IdentityTTA(), HFlipTTA(), ZoomOutTTA(scale=float(zoom_scale))]
    return transforms, list(weights)


def predict_annotation(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    input_size: int,
    device: torch.device,
    post: PostprocessParams,
    tta_transforms: list[TTATransform] | None = None,
    tta_weights: list[float] | None = None,
) -> str:
    prob = predict_prob_map(
        model,
        image,
        input_size=input_size,
        device=device,
        tta_transforms=tta_transforms,
        tta_weights=tta_weights,
    )
    instances = postprocess_prob(prob, post)
    return masks_to_annotation(instances)
