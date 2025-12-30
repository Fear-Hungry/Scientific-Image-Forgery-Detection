import numpy as np
import pytest

from forgeryseg.inference import TTA_MODES, apply_tta, undo_tta


@pytest.mark.parametrize("mode", TTA_MODES)
def test_tta_round_trip_2d(mode: str):
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    out = undo_tta(apply_tta(arr, mode), mode)
    assert np.array_equal(out, arr)


@pytest.mark.parametrize("mode", TTA_MODES)
def test_tta_round_trip_3d(mode: str):
    arr = np.arange(3 * 4 * 2, dtype=np.int64).reshape(3, 4, 2)
    out = undo_tta(apply_tta(arr, mode), mode)
    assert np.array_equal(out, arr)


def test_tta_invalid_mode_raises():
    arr = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="Invalid TTA mode"):
        apply_tta(arr, "not-a-mode")


def test_tta_axes_for_chw():
    chw = np.arange(3 * 2 * 4, dtype=np.int64).reshape(3, 2, 4)
    out = undo_tta(apply_tta(chw, "hflip", axes=(1, 2)), "hflip", axes=(1, 2))
    assert np.array_equal(out, chw)
