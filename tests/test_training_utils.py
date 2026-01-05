from __future__ import annotations

import numpy as np
import torch

from forgeryseg.training.utils import apply_cutmix, stratified_splits


def test_stratified_splits_kfold_contains_both_classes_per_fold() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    splits = stratified_splits(labels, folds=3, val_fraction=0.2, seed=123)
    assert len(splits) == 3
    for _, _, val_idx in splits:
        assert set(labels[val_idx].tolist()) == {0, 1}


def test_apply_cutmix_changes_batch_when_enabled() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    x = torch.zeros((2, 3, 10, 10), dtype=torch.float32)
    y = torch.zeros((2, 1, 10, 10), dtype=torch.float32)
    x[1] = 1.0
    y[1] = 1.0

    x2, y2 = apply_cutmix(x, y, prob=1.0, alpha=1.0)
    assert x2.shape == x.shape
    assert y2.shape == y.shape

    # Both samples should contain mixed content.
    assert float(x2[0].min()) == 0.0 and float(x2[0].max()) == 1.0
    assert float(x2[1].min()) == 0.0 and float(x2[1].max()) == 1.0
    assert float(y2[0].min()) == 0.0 and float(y2[0].max()) == 1.0
    assert float(y2[1].min()) == 0.0 and float(y2[1].max()) == 1.0
