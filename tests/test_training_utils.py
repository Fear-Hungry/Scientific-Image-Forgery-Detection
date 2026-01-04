from __future__ import annotations

import numpy as np

from forgeryseg.training.utils import stratified_splits


def test_stratified_splits_kfold_contains_both_classes_per_fold() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    splits = stratified_splits(labels, folds=3, val_fraction=0.2, seed=123)
    assert len(splits) == 3
    for _, _, val_idx in splits:
        assert set(labels[val_idx].tolist()) == {0, 1}
