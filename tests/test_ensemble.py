from __future__ import annotations

import numpy as np

from forgeryseg.ensemble import ensemble_annotations
from forgeryseg.rle import annotation_to_union_mask, masks_to_annotation


def test_ensemble_union_preserves_mask() -> None:
    shape = (6, 6)
    m = np.zeros(shape, dtype=np.uint8)
    m[2:4, 1:3] = 1
    ann = masks_to_annotation([m])

    out = ensemble_annotations([ann, "authentic"], shape=shape, method="union")
    out_mask = annotation_to_union_mask(out, shape)
    assert np.array_equal(out_mask, m)


def test_ensemble_weighted_threshold() -> None:
    shape = (4, 4)
    m = np.zeros(shape, dtype=np.uint8)
    m[0, 0] = 1
    ann = masks_to_annotation([m])

    out = ensemble_annotations([ann, "authentic", "authentic"], shape=shape, method="weighted", weights=[0.6, 0.2, 0.2])
    out_mask = annotation_to_union_mask(out, shape)
    assert out_mask[0, 0] == 1
