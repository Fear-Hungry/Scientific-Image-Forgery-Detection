from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch

from .image import letterbox_reflect

FFTMode = Literal["logmag", "hp_residual", "phase_only"]


@dataclass(frozen=True)
class FFTParams:
    mode: FFTMode = "logmag"
    input_size: int = 256
    hp_radius_fraction: float = 0.1
    normalize_percentiles: tuple[float, float] = (5.0, 95.0)


def _to_gray01(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape={image.shape}")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def _normalize_robust(x: np.ndarray, p: tuple[float, float] = (5.0, 95.0)) -> np.ndarray:
    lo, hi = np.percentile(x, [float(p[0]), float(p[1])])
    denom = float(hi - lo) + 1e-6
    y = (x - float(lo)) / denom
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def fft_log_magnitude(gray01: np.ndarray, *, shift: bool = True) -> np.ndarray:
    if gray01.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got shape={gray01.shape}")
    f = np.fft.fft2(gray01)
    if shift:
        f = np.fft.fftshift(f)
    mag = np.log1p(np.abs(f)).astype(np.float32)
    return mag


def fft_highpass_residual(gray01: np.ndarray, *, radius_fraction: float = 0.1) -> np.ndarray:
    if gray01.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got shape={gray01.shape}")
    f = np.fft.fft2(gray01)
    f = np.fft.fftshift(f)

    h, w = gray01.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r0 = float(radius_fraction) * float(min(h, w)) / 2.0
    mask = (rr >= r0).astype(np.float32)

    f_hp = f * mask
    inv = np.fft.ifft2(np.fft.ifftshift(f_hp))
    res = np.abs(inv.real).astype(np.float32)
    return res


def fft_phase_only_reconstruction(gray01: np.ndarray, *, shift: bool = True) -> np.ndarray:
    if gray01.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got shape={gray01.shape}")
    f = np.fft.fft2(gray01)
    if shift:
        f = np.fft.fftshift(f)
    phase = np.angle(f)
    f_phase = np.exp(1j * phase)
    inv = np.fft.ifft2(np.fft.ifftshift(f_phase) if shift else f_phase)
    return np.abs(inv).astype(np.float32)


def fft_representation(image_rgb: np.ndarray, params: FFTParams) -> np.ndarray:
    padded, _ = letterbox_reflect(image_rgb, int(params.input_size))
    gray01 = _to_gray01(padded)

    if params.mode == "logmag":
        raw = fft_log_magnitude(gray01, shift=True)
    elif params.mode == "hp_residual":
        raw = fft_highpass_residual(gray01, radius_fraction=float(params.hp_radius_fraction))
    elif params.mode == "phase_only":
        raw = fft_phase_only_reconstruction(gray01, shift=True)
    else:
        raise ValueError(f"Unknown FFT mode: {params.mode}")

    return _normalize_robust(raw, params.normalize_percentiles)


def fft_tensor(image_rgb: np.ndarray, params: FFTParams) -> torch.Tensor:
    x = fft_representation(image_rgb, params)
    return torch.from_numpy(x[None, ...]).contiguous().float()
