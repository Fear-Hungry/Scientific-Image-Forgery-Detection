from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

MorphOrder = Literal["open_close", "close_open"]


@dataclass(frozen=True)
class PostprocessParams:
    prob_threshold: float = 0.5
    prob_threshold_low: float | None = None
    gaussian_sigma: float = 0.0
    sobel_weight: float = 0.0

    open_kernel: int = 0
    close_kernel: int = 0
    morph_order: MorphOrder = "open_close"

    final_open_kernel: int = 0
    final_close_kernel: int = 0
    fill_holes: bool = False

    min_area: int = 0
    min_mean_conf: float = 0.0
    min_prob_std: float = 0.0

    small_area: int | None = None
    small_min_mean_conf: float | None = None

    authentic_area_max: int | None = None
    authentic_conf_max: float | None = None


def _kernel(k: int) -> np.ndarray:
    k = int(k)
    if k <= 1:
        return np.ones((1, 1), dtype=np.uint8)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _apply_morph(mask: np.ndarray, *, open_kernel: int, close_kernel: int, order: MorphOrder) -> np.ndarray:
    """
    Apply opening/closing in a configurable order.
    """
    mask = mask.astype(np.uint8)
    open_kernel = int(open_kernel)
    close_kernel = int(close_kernel)

    def _open(x: np.ndarray) -> np.ndarray:
        if open_kernel > 1:
            return cv2.morphologyEx(x, cv2.MORPH_OPEN, _kernel(open_kernel))
        return x

    def _close(x: np.ndarray) -> np.ndarray:
        if close_kernel > 1:
            return cv2.morphologyEx(x, cv2.MORPH_CLOSE, _kernel(close_kernel))
        return x

    if order == "open_close":
        return _close(_open(mask))
    if order == "close_open":
        return _open(_close(mask))
    raise ValueError(f"Unknown morph_order: {order}")


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes of a binary mask (uint8 0/1).

    Uses flood-fill on the inverted padded mask to identify holes.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_u8.shape}")

    padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
    inv = (1 - padded).astype(np.uint8)
    flood = inv.copy()
    h, w = flood.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=0)
    holes = flood
    filled = np.clip(padded + holes, 0, 1).astype(np.uint8)
    return filled[1:-1, 1:-1]


def _connected_components(mask: np.ndarray) -> list[tuple[np.ndarray, int]]:
    mask_u8 = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    comps: list[tuple[np.ndarray, int]] = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comps.append(((labels == label).astype(np.uint8), area))
    return comps


def postprocess_prob(
    prob: np.ndarray,
    params: PostprocessParams,
) -> list[np.ndarray]:
    if prob.ndim != 2:
        raise ValueError(f"Expected 2D prob map, got shape={prob.shape}")
    prob = prob.astype(np.float32)

    if params.gaussian_sigma and params.gaussian_sigma > 0:
        prob = cv2.GaussianBlur(prob, ksize=(0, 0), sigmaX=float(params.gaussian_sigma))

    if params.sobel_weight and params.sobel_weight > 0:
        sx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        denom = float(mag.max()) + 1e-6
        prob = np.clip(prob + (mag / denom) * float(params.sobel_weight), 0.0, 1.0)

    if params.min_prob_std and float(prob.std()) < float(params.min_prob_std):
        return []

    thr_high = float(params.prob_threshold)
    thr_low = params.prob_threshold_low

    if thr_low is not None and float(thr_low) > 0 and float(thr_low) < float(thr_high):
        mask_high = (prob >= float(thr_high)).astype(np.uint8)
        if int(mask_high.sum()) == 0:
            return []
        mask_low = (prob >= float(thr_low)).astype(np.uint8)
        kept_low: list[np.ndarray] = []
        for comp_mask, _area in _connected_components(mask_low):
            if np.any(mask_high[comp_mask.astype(bool)]):
                kept_low.append(comp_mask)
        if not kept_low:
            return []
        mask = np.clip(np.sum(np.stack(kept_low, axis=0), axis=0), 0, 1).astype(np.uint8)
    else:
        mask = (prob >= float(thr_high)).astype(np.uint8)

    mask = _apply_morph(mask, open_kernel=int(params.open_kernel), close_kernel=int(params.close_kernel), order=params.morph_order)

    kept: list[np.ndarray] = []
    for comp_mask, area in _connected_components(mask):
        if params.min_area and area < int(params.min_area):
            continue

        mean_conf = float(prob[comp_mask.astype(bool)].mean()) if area > 0 else 0.0
        if params.small_area is not None and params.small_min_mean_conf is not None:
            if area < int(params.small_area) and mean_conf < float(params.small_min_mean_conf):
                continue
        kept.append(comp_mask)

    if not kept:
        return []

    union = np.clip(np.sum(np.stack(kept, axis=0), axis=0), 0, 1).astype(np.uint8)

    if params.final_open_kernel > 1 or params.final_close_kernel > 1:
        union = _apply_morph(
            union,
            open_kernel=int(params.final_open_kernel),
            close_kernel=int(params.final_close_kernel),
            order=params.morph_order,
        )
    if params.fill_holes:
        union = _fill_holes(union)

    union_area = int(union.sum())
    union_mean = float(prob[union.astype(bool)].mean()) if union_area > 0 else 0.0

    if params.min_mean_conf and union_mean < float(params.min_mean_conf):
        return []

    if params.authentic_area_max is not None and params.authentic_conf_max is not None:
        if union_area < int(params.authentic_area_max) and union_mean < float(params.authentic_conf_max):
            return []

    instances: list[np.ndarray] = []
    for comp_mask, area in _connected_components(union):
        if params.min_area and area < int(params.min_area):
            continue
        instances.append(comp_mask)
    return instances
