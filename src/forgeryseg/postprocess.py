from __future__ import annotations

from typing import List

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from scipy.ndimage import label as cc_label
    from scipy.ndimage import binary_fill_holes
except ImportError:  # pragma: no cover - optional dependency
    cc_label = None
    binary_fill_holes = None


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a probability map or mask into uint8 binary (0/1)."""
    return (np.asarray(mask) >= threshold).astype(np.uint8)


def morph_close(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Morphological closing (dilation then erosion) on a binary mask.
    Returns a uint8 (0/1) mask.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    kernel_size = int(kernel_size)
    iterations = int(iterations)
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if iterations <= 0:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if cv2 is None:
        raise ImportError("opencv-python (cv2) is required for morphological operations")

    m = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return (out > 0).astype(np.uint8)


def morph_open(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Morphological opening (erosion then dilation) on a binary mask.
    Returns a uint8 (0/1) mask.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    kernel_size = int(kernel_size)
    iterations = int(iterations)
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if iterations <= 0:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if cv2 is None:
        raise ImportError("opencv-python (cv2) is required for morphological operations")

    m = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return (out > 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in a binary mask. Returns a uint8 (0/1) mask.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    m = (mask > 0)
    if binary_fill_holes is not None:
        return binary_fill_holes(m).astype(np.uint8)

    if cv2 is None:
        raise ImportError("scipy or opencv-python (cv2) is required for hole filling")

    im = (m.astype(np.uint8) * 255)
    padded = np.pad(im, 1, mode="constant", constant_values=0)
    h, w = padded.shape
    flood = padded.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, flood_inv)
    filled = filled[1:-1, 1:-1]
    return (filled > 0).astype(np.uint8)


def median_smooth(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Median smoothing on a binary mask. Returns a uint8 (0/1) mask.

    Note: kernel_size must be an odd integer >= 3.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    kernel_size = int(kernel_size)
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and >= 3, got {kernel_size}")
    if cv2 is None:
        raise ImportError("opencv-python (cv2) is required for median smoothing")

    im = (mask > 0).astype(np.uint8) * 255
    sm = cv2.medianBlur(im, kernel_size)
    return (sm >= 128).astype(np.uint8)


def postprocess_binary(
    mask: np.ndarray,
    *,
    closing_ksize: int = 0,
    closing_iters: int = 1,
    opening_ksize: int = 0,
    opening_iters: int = 1,
    fill_holes_enabled: bool = False,
    median_ksize: int = 0,
) -> np.ndarray:
    """
    Apply optional morphological post-processing to a binary mask.

    All operations are disabled by default, keeping backward compatibility.
    """
    out = (np.asarray(mask) > 0).astype(np.uint8)

    if closing_ksize:
        out = morph_close(out, kernel_size=int(closing_ksize), iterations=int(closing_iters))
    if opening_ksize:
        out = morph_open(out, kernel_size=int(opening_ksize), iterations=int(opening_iters))
    if fill_holes_enabled:
        out = fill_holes(out)
    if median_ksize:
        out = median_smooth(out, kernel_size=int(median_ksize))

    return out.astype(np.uint8)


def prob_to_instances(
    prob: np.ndarray,
    *,
    threshold: float = 0.5,
    adaptive_threshold: bool = False,
    threshold_factor: float = 0.3,
    min_area: int = 0,
    min_area_percent: float = 0.0,
    min_confidence: float = 0.0,
    closing_ksize: int = 0,
    closing_iters: int = 1,
    opening_ksize: int = 0,
    opening_iters: int = 1,
    fill_holes_enabled: bool = False,
    median_ksize: int = 0,
) -> List[np.ndarray]:
    """
    Convert a 2D probability map into post-processed connected components.
    """
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"Expected 2D prob map, got shape {prob.shape}")

    if adaptive_threshold:
        thr = adaptive_threshold_value(prob, factor=float(threshold_factor))
    else:
        thr = float(threshold)
    bin_mask = binarize(prob, threshold=float(thr))
    bin_mask = postprocess_binary(
        bin_mask,
        closing_ksize=int(closing_ksize),
        closing_iters=int(closing_iters),
        opening_ksize=int(opening_ksize),
        opening_iters=int(opening_iters),
        fill_holes_enabled=bool(fill_holes_enabled),
        median_ksize=int(median_ksize),
    )
    total_pixels = int(prob.shape[0] * prob.shape[1])
    min_area_percent = _normalize_percent(min_area_percent)
    min_area_px = int(round(min_area_percent * total_pixels)) if min_area_percent > 0 else 0
    effective_min_area = max(int(min_area), int(min_area_px))
    instances = extract_components(bin_mask, min_area=effective_min_area)

    if min_area_percent > 0 or min_confidence > 0:
        if not instances:
            return []
        union = mask_from_instances(instances, prob.shape)
        area = int(union.sum())
        area_percent = float(area) / float(total_pixels) if total_pixels > 0 else 0.0
        if min_area_percent > 0 and area_percent < float(min_area_percent):
            return []
        if float(min_confidence) > 0:
            mean_conf = float(prob[union > 0].mean()) if area > 0 else 0.0
            if mean_conf < float(min_confidence):
                return []

    return instances


def adaptive_threshold_value(prob: np.ndarray, factor: float = 0.3) -> float:
    """
    Compute an adaptive threshold: mean + factor * std, clamped to [0, 1].
    """
    arr = np.asarray(prob, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D prob map, got shape {arr.shape}")
    mean = float(arr.mean())
    std = float(arr.std())
    thr = mean + float(factor) * std
    eps = float(np.finfo(np.float32).eps)
    return float(np.clip(thr, eps, 1.0))


def _normalize_percent(value: float) -> float:
    v = float(value)
    if v < 0:
        raise ValueError("min_area_percent must be >= 0")
    if v == 0:
        return 0.0
    if v > 1.0:
        v = v / 100.0
    return float(v)


def extract_components(mask: np.ndarray, min_area: int = 0) -> List[np.ndarray]:
    """
    Extract 4-connected components from a binary mask.
    Returns a list of binary masks, one per component.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if mask.max() == 0:
        return []
    if cc_label is None:
        raise ImportError("scipy is required for connected components")

    labeled, num = cc_label(mask > 0)
    instances: List[np.ndarray] = []
    for idx in range(1, num + 1):
        comp = (labeled == idx)
        if min_area and int(comp.sum()) < min_area:
            continue
        instances.append(comp.astype(np.uint8))
    return instances


def mask_from_instances(instances: List[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Union of instance masks into a single binary mask."""
    union = np.zeros(shape, dtype=np.uint8)
    for inst in instances:
        union = np.logical_or(union, inst)
    return union.astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove components smaller than min_area from a binary mask."""
    instances = extract_components(mask, min_area=min_area)
    return mask_from_instances(instances, mask.shape)
