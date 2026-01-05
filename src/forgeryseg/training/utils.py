from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def fold_out_path(base: Path, *, fold: int, folds: int) -> Path:
    if folds <= 1:
        return base
    return base.with_name(f"{base.stem}_fold{fold}{base.suffix}")


def stratified_splits(
    labels: np.ndarray,
    *,
    folds: int,
    val_fraction: float,
    seed: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if folds < 1:
        raise ValueError("folds must be >= 1")

    if folds == 1:
        if not (0.0 < float(val_fraction) < 1.0):
            raise ValueError("val_fraction must be between 0 and 1 when folds=1")
        try:
            from sklearn.model_selection import StratifiedShuffleSplit

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(val_fraction), random_state=int(seed))
            train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
            return [(0, train_idx, val_idx)]
        except Exception as e:
            print(f"[warn] stratified split failed ({type(e).__name__}: {e}); falling back to random split")
            indices = np.random.permutation(len(labels))
            n_val = int(round(len(indices) * float(val_fraction)))
            return [(0, indices[n_val:], indices[:n_val])]

    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
        return [(i, tr, va) for i, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels))]
    except Exception as e:
        raise RuntimeError(f"StratifiedKFold unavailable ({type(e).__name__}: {e})") from e


def apply_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    prob: float,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix for segmentation batches.

    - `x`: (B, C, H, W) float tensor
    - `y`: (B, 1, H, W) float tensor
    """
    p = float(prob)
    if p <= 0.0:
        return x, y
    if x.ndim != 4 or y.ndim != 4:
        return x, y
    b = int(x.shape[0])
    if b < 2:
        return x, y
    if torch.rand((), device=x.device).item() > p:
        return x, y

    a = float(alpha)
    lam = float(np.random.beta(a, a)) if a > 0 else 1.0
    lam = float(np.clip(lam, 0.0, 1.0))

    _, _, h, w = x.shape
    cut_w = int(round(w * float(np.sqrt(max(0.0, 1.0 - lam)))))
    cut_h = int(round(h * float(np.sqrt(max(0.0, 1.0 - lam)))))
    if cut_w <= 1 or cut_h <= 1:
        return x, y

    cx = int(torch.randint(0, w, (), device=x.device).item())
    cy = int(torch.randint(0, h, (), device=x.device).item())
    x0 = max(0, cx - cut_w // 2)
    x1 = min(w, cx + cut_w // 2)
    y0 = max(0, cy - cut_h // 2)
    y1 = min(h, cy + cut_h // 2)
    if x1 <= x0 or y1 <= y0:
        return x, y

    perm = torch.randperm(b, device=x.device)
    if torch.equal(perm, torch.arange(b, device=x.device)):
        perm = torch.roll(perm, shifts=1, dims=0)
    x2 = x[perm]
    y2 = y[perm]

    x = x.clone()
    y = y.clone()
    x[..., y0:y1, x0:x1] = x2[..., y0:y1, x0:x1]
    y[..., y0:y1, x0:x1] = y2[..., y0:y1, x0:x1]
    return x, y
