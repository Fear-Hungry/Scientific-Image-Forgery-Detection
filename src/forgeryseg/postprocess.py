from __future__ import annotations

from typing import List

import numpy as np

try:
    from scipy.ndimage import label as cc_label
except ImportError:  # pragma: no cover - optional dependency
    cc_label = None


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a probability map or mask into uint8 binary (0/1)."""
    return (np.asarray(mask) >= threshold).astype(np.uint8)


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
