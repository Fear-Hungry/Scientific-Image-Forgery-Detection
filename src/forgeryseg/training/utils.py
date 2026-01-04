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
