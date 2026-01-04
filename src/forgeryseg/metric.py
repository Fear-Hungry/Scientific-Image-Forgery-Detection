from __future__ import annotations

from typing import Sequence

import numpy as np


def _pixel_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def of1_score(pred_masks: Sequence[np.ndarray], gt_masks: Sequence[np.ndarray]) -> float:
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:  # pragma: no cover
        raise ImportError("SciPy is required for oF1 scoring (Hungarian matching).") from e

    f1 = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            f1[i, j] = _pixel_f1(p, g)

    row_ind, col_ind = linear_sum_assignment(-f1)
    matched_sum = float(f1[row_ind, col_ind].sum())
    return matched_sum / float(max(n_pred, n_gt))

