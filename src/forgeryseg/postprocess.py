from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PostprocessParams:
    prob_threshold: float = 0.5
    gaussian_sigma: float = 0.0
    sobel_weight: float = 0.0

    open_kernel: int = 0
    close_kernel: int = 0

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

    mask = (prob >= float(params.prob_threshold)).astype(np.uint8)

    if params.open_kernel and params.open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(params.open_kernel))
    if params.close_kernel and params.close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(params.close_kernel))

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
