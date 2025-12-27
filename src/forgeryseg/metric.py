from __future__ import annotations

from typing import Iterable, List

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - optional dependency
    linear_sum_assignment = None

from .postprocess import extract_components


def _as_instance_list(masks: Iterable[np.ndarray] | np.ndarray | None) -> List[np.ndarray]:
    if masks is None:
        return []
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            return extract_components(masks)
        if masks.ndim == 3:
            return [(masks[i] > 0).astype(np.uint8) for i in range(masks.shape[0])]
    if isinstance(masks, (list, tuple)):
        return [(np.asarray(m) > 0).astype(np.uint8) for m in masks]
    raise ValueError("Unsupported mask container")


def _build_f1_matrix(gt_instances: List[np.ndarray], pred_instances: List[np.ndarray]) -> np.ndarray:
    gt_count = len(gt_instances)
    pred_count = len(pred_instances)
    if gt_count == 0 or pred_count == 0:
        return np.zeros((gt_count, pred_count), dtype=np.float32)

    gt_sums = [int(m.sum()) for m in gt_instances]
    pred_sums = [int(m.sum()) for m in pred_instances]

    f1_matrix = np.zeros((gt_count, pred_count), dtype=np.float32)
    for i, gt_mask in enumerate(gt_instances):
        if gt_sums[i] == 0:
            continue
        for j, pred_mask in enumerate(pred_instances):
            if pred_sums[j] == 0:
                continue
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            if intersection == 0:
                continue
            f1_matrix[i, j] = (2.0 * intersection) / (gt_sums[i] + pred_sums[j])
    return f1_matrix


def score_image(gt_masks: Iterable[np.ndarray] | np.ndarray | None,
                pred_masks: Iterable[np.ndarray] | np.ndarray | None) -> float:
    """
    Compute RecodAI F1 score for a single image.
    Inputs can be lists of instance masks, or 2D/3D arrays.
    """
    gt_instances = _as_instance_list(gt_masks)
    pred_instances = _as_instance_list(pred_masks)

    gt_count = len(gt_instances)
    pred_count = len(pred_instances)

    if gt_count == 0 and pred_count == 0:
        return 1.0
    if gt_count == 0 and pred_count > 0:
        return 0.0
    if gt_count > 0 and pred_count == 0:
        return 0.0

    if linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian matching")

    f1_matrix = _build_f1_matrix(gt_instances, pred_instances)
    row_ind, col_ind = linear_sum_assignment(-f1_matrix)
    if row_ind.size == 0:
        return 0.0

    matched = f1_matrix[row_ind, col_ind]
    if pred_count < gt_count:
        base = float(matched.sum() / gt_count)
    else:
        base = float(matched.mean())
    penalty = gt_count / max(pred_count, gt_count)
    return base * penalty


def score_dataset(gt_list: Iterable[Iterable[np.ndarray] | np.ndarray | None],
                  pred_list: Iterable[Iterable[np.ndarray] | np.ndarray | None]) -> float:
    """Compute mean RecodAI F1 over a dataset."""
    scores = [score_image(gt, pred) for gt, pred in zip(gt_list, pred_list)]
    if not scores:
        return 0.0
    return float(np.mean(scores))
