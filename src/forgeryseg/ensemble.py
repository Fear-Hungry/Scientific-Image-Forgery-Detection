from __future__ import annotations

from typing import Literal, Sequence

import cv2
import numpy as np

from .rle import annotation_to_union_mask, masks_to_annotation

EnsembleMethod = Literal["weighted", "majority", "union", "intersection"]


def rank_weights_by_score(scores: Sequence[float]) -> list[float]:
    n = len(scores)
    if n == 0:
        return []
    order = np.argsort(np.asarray(scores))  # lowest score -> highest weight
    raw = np.zeros(n, dtype=np.float32)
    for rank, idx in enumerate(order):
        raw[idx] = float(n - rank)
    w = raw / raw.sum()
    return w.tolist()


def binary_mask_to_instances(mask: np.ndarray) -> list[np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out: list[np.ndarray] = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        out.append((labels == label).astype(np.uint8))
    return out


def ensemble_annotations(
    annotations: Sequence[str | float | None],
    *,
    shape: tuple[int, int],
    method: EnsembleMethod = "weighted",
    weights: Sequence[float] | None = None,
    threshold: float = 0.5,
) -> str:
    if len(annotations) == 0:
        return "authentic"

    masks = [annotation_to_union_mask(a, shape) for a in annotations]

    if method == "union":
        combined = np.clip(np.sum(np.stack(masks, axis=0), axis=0), 0, 1).astype(np.uint8)
    elif method == "intersection":
        combined = np.all(np.stack(masks, axis=0).astype(bool), axis=0).astype(np.uint8)
    elif method == "majority":
        votes = np.sum(np.stack(masks, axis=0), axis=0)
        combined = (votes >= (len(masks) // 2 + 1)).astype(np.uint8)
    else:  # weighted
        if weights is None:
            weights = [1.0 / len(masks)] * len(masks)
        total_w = float(sum(weights))
        if total_w <= 0:
            raise ValueError("sum(weights) must be > 0")
        weights = [float(w) / total_w for w in weights]
        prob = np.zeros(shape, dtype=np.float32)
        for m, wgt in zip(masks, weights, strict=True):
            prob += m.astype(np.float32) * float(wgt)
        combined = (prob > float(threshold)).astype(np.uint8)

    return masks_to_annotation(binary_mask_to_instances(combined))

