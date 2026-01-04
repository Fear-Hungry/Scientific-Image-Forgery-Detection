from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxMeta:
    orig_h: int
    orig_w: int
    new_h: int
    new_w: int
    pad_top: int
    pad_left: int


def letterbox_reflect(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxMeta]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left
    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    meta = LetterboxMeta(
        orig_h=h,
        orig_w=w,
        new_h=new_h,
        new_w=new_w,
        pad_top=pad_top,
        pad_left=pad_left,
    )
    return padded, meta


def unletterbox(image: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    if image.ndim == 2:
        cropped = image[
            meta.pad_top : meta.pad_top + meta.new_h,
            meta.pad_left : meta.pad_left + meta.new_w,
        ]
        return cv2.resize(cropped, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR)

    if image.ndim == 3:
        cropped = image[
            meta.pad_top : meta.pad_top + meta.new_h,
            meta.pad_left : meta.pad_left + meta.new_w,
            :,
        ]
        return cv2.resize(cropped, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR)

    raise ValueError(f"Expected 2D or 3D array, got shape={image.shape}")

