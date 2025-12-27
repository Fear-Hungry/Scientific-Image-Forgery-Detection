import pytest
import numpy as np


def test_fill_holes_fills_enclosed_hole():
    pytest.importorskip("scipy")
    from forgeryseg.postprocess import fill_holes

    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[1:6, 1:6] = 1
    mask[2:5, 2:5] = 0  # enclosed hole

    filled = fill_holes(mask)
    assert filled.dtype == np.uint8
    assert filled.shape == mask.shape
    assert int(filled.sum()) == 25  # 5x5 square


def test_postprocess_binary_closing_merges_components():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from forgeryseg.postprocess import extract_components, postprocess_binary

    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 1:3] = 1
    mask[2:5, 4:6] = 1  # gap of 1 column between blocks

    assert len(extract_components(mask)) == 2
    closed = postprocess_binary(mask, closing_ksize=3, closing_iters=1)
    assert len(extract_components(closed)) == 1

