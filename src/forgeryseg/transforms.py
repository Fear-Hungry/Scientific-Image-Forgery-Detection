from __future__ import annotations

import inspect
from typing import Callable, Literal

import numpy as np

AugMode = Literal["none", "basic", "robust"]
TransformFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _safe_gauss_noise(*, A, p: float):
    """
    Albumentations 1.x used `var_limit`, 2.x uses `std_range` (fraction of max value).
    We keep a comparable noise magnitude across versions.
    """
    params = inspect.signature(A.GaussNoise).parameters
    if "std_range" in params:
        # Previous var_limit=(10,50) => sigma ~ (sqrt(10), sqrt(50)) on uint8 scale.
        # Albumentations 2.x expects std as a fraction of max value (255 for uint8).
        return A.GaussNoise(std_range=(0.012, 0.028), mean_range=(0.0, 0.0), p=float(p))
    if "var_limit" in params:
        return A.GaussNoise(var_limit=(10.0, 50.0), p=float(p))
    return A.GaussNoise(p=float(p))


def _safe_image_compression(*, A, p: float):
    """
    JPEG/WebP compression artifacts (JPEG by default).

    Albumentations 2.x: ImageCompression(compression_type='jpeg', quality_range=(q0, q1))
    Albumentations 1.x: ImageCompression(quality_lower=..., quality_upper=...)
    """
    if hasattr(A, "ImageCompression"):
        params = inspect.signature(A.ImageCompression).parameters
        if "quality_range" in params:
            return A.ImageCompression(compression_type="jpeg", quality_range=(30, 95), p=float(p))
        if "quality_lower" in params and "quality_upper" in params:
            return A.ImageCompression(quality_lower=30, quality_upper=95, p=float(p))
        return A.ImageCompression(p=float(p))
    if hasattr(A, "JpegCompression"):
        return A.JpegCompression(quality_lower=30, quality_upper=95, p=float(p))
    return None


def make_transforms(input_size: int, *, train: bool, aug: AugMode) -> TransformFn:
    """
    Albumentations pipeline for (image, mask).

    Notes:
    - Always resizes + pads to `input_size` (LongestMaxSize + PadIfNeeded).
    - When `train=True`, adds optional augmentations depending on `aug`.
    """
    import albumentations as A
    import cv2

    base = [
        A.LongestMaxSize(max_size=int(input_size), interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(
            min_height=int(input_size),
            min_width=int(input_size),
            border_mode=cv2.BORDER_REFLECT_101,
        ),
    ]

    if train:
        if aug == "none":
            pass
        elif aug in {"basic", "robust"}:
            base.append(A.HorizontalFlip(p=0.5))
            base.append(A.VerticalFlip(p=0.25))
            base.append(A.RandomBrightnessContrast(p=0.25))
        else:
            raise ValueError(f"Unknown aug mode: {aug}")

        if aug == "robust":
            base.extend(
                [
                    A.RandomRotate90(p=0.2),
                    # Albumentations 2.x warns that ShiftScaleRotate is a special case of Affine.
                    # Keep compatibility with older versions by falling back when needed.
                    (
                        A.Affine(
                            scale=(0.85, 1.15),
                            translate_percent=(-0.05, 0.05),
                            rotate=(-15, 15),
                            border_mode=cv2.BORDER_REFLECT_101,
                            p=0.5,
                        )
                        if hasattr(A, "Affine")
                        else A.ShiftScaleRotate(
                            shift_limit=0.05,
                            scale_limit=0.15,
                            rotate_limit=15,
                            border_mode=cv2.BORDER_REFLECT_101,
                            p=0.5,
                        )
                    ),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.MotionBlur(blur_limit=7, p=1.0),
                        ],
                        p=0.2,
                    ),
                    _safe_gauss_noise(A=A, p=0.2),
                ]
            )
            comp = _safe_image_compression(A=A, p=0.2)
            if comp is not None:
                base.append(comp)

    aug_tf = A.Compose(base)

    def _apply(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = aug_tf(image=img, mask=mask)
        return out["image"], out["mask"]

    return _apply
