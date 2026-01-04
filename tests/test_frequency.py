from __future__ import annotations

import numpy as np

from forgeryseg.frequency import FFTParams, fft_representation


def test_fft_representation_logmag_shape_and_range() -> None:
    img = np.zeros((64, 80, 3), dtype=np.uint8)
    img[10:20, 5:15, :] = 255

    params = FFTParams(mode="logmag", input_size=128)
    out = fft_representation(img, params)
    assert out.shape == (128, 128)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) <= 1.0
    assert 0.0 <= float(out.max()) <= 1.0


def test_fft_representation_hp_residual_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    img = (rng.random((96, 96, 3)) * 255).astype(np.uint8)

    params = FFTParams(mode="hp_residual", input_size=96, hp_radius_fraction=0.2)
    out = fft_representation(img, params)
    assert out.shape == (96, 96)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) <= 1.0
    assert 0.0 <= float(out.max()) <= 1.0


def test_fft_representation_phase_only_shape_and_range() -> None:
    rng = np.random.default_rng(1)
    img = (rng.random((72, 96, 3)) * 255).astype(np.uint8)

    params = FFTParams(mode="phase_only", input_size=128)
    out = fft_representation(img, params)
    assert out.shape == (128, 128)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) <= 1.0
    assert 0.0 <= float(out.max()) <= 1.0
