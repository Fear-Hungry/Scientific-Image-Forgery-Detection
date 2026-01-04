from __future__ import annotations

import numpy as np

from forgeryseg.rle import annotation_to_masks, masks_to_annotation


def test_rle_roundtrip_single_instance() -> None:
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2, 1] = 1

    ann = masks_to_annotation([mask])
    assert ann == "[1, 1, 6, 1]"

    decoded = annotation_to_masks(ann, mask.shape)
    assert len(decoded) == 1
    assert np.array_equal(decoded[0], mask)


def test_rle_multiple_instances() -> None:
    m1 = np.zeros((4, 4), dtype=np.uint8)
    m1[0, 0] = 1
    m2 = np.zeros((4, 4), dtype=np.uint8)
    m2[3, 3] = 1

    ann = masks_to_annotation([m1, m2])
    decoded = annotation_to_masks(ann, m1.shape)
    assert len(decoded) == 2
    assert np.array_equal(decoded[0], m1)
    assert np.array_equal(decoded[1], m2)


def test_rle_authentic_is_empty() -> None:
    ann = masks_to_annotation([])
    assert ann == "authentic"
    assert annotation_to_masks(ann, (5, 6)) == []

