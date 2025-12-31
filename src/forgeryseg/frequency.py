from __future__ import annotations

import numpy as np


def compute_fft_mag(image: np.ndarray) -> np.ndarray:
    """
    Compute normalized log-magnitude of the 2D Fourier spectrum.

    Input:
      - image: (H, W, C) or (H, W) array (uint8 or float).
        If C>=3, only the first 3 channels are used (RGB).

    Output:
      - (H, W, 1) float32 in [0, 1]
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        if arr.shape[2] >= 3:
            rgb = arr[..., :3].astype(np.float32)
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        elif arr.shape[2] == 1:
            gray = arr[..., 0].astype(np.float32)
        else:
            gray = arr.mean(axis=2).astype(np.float32)
    else:
        raise ValueError(f"Expected 2D/3D image array, got shape {arr.shape}")

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift)).astype(np.float32)

    mag -= float(mag.min())
    denom = float(mag.max())
    if denom > 0:
        mag /= denom
    else:
        mag.fill(0.0)

    return mag[..., None]

