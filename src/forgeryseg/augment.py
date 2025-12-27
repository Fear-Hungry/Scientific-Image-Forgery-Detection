from __future__ import annotations

import inspect
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    import albumentations as A
except ImportError:  # pragma: no cover - optional dependency
    A = None

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _as_hw(size: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    return int(size[0]), int(size[1])


def _has_param(transform_cls, name: str) -> bool:
    return name in inspect.signature(transform_cls).parameters


def _fill_kwargs(transform_cls, fill_value: float | int = 0) -> dict:
    kwargs = {}
    if _has_param(transform_cls, "fill"):
        kwargs["fill"] = fill_value
    if _has_param(transform_cls, "fill_mask"):
        kwargs["fill_mask"] = fill_value
    if _has_param(transform_cls, "value"):
        kwargs["value"] = fill_value
    if _has_param(transform_cls, "mask_value"):
        kwargs["mask_value"] = fill_value
    if _has_param(transform_cls, "cval"):
        kwargs["cval"] = fill_value
    if _has_param(transform_cls, "cval_mask"):
        kwargs["cval_mask"] = fill_value
    return kwargs


def _random_resized_crop(size_hw: tuple[int, int], scale: tuple[float, float], ratio: tuple[float, float], p: float):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    if _has_param(A.RandomResizedCrop, "size"):
        return A.RandomResizedCrop(size=size_hw, scale=scale, ratio=ratio, p=p)
    return A.RandomResizedCrop(height=size_hw[0], width=size_hw[1], scale=scale, ratio=ratio, p=p)


def _image_compression(quality_range: tuple[int, int], p: float):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    if _has_param(A.ImageCompression, "quality_range"):
        return A.ImageCompression(quality_range=quality_range, p=p)
    return A.ImageCompression(quality_lower=quality_range[0], quality_upper=quality_range[1], p=p)


def _coarse_dropout(p: float):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    if _has_param(A.CoarseDropout, "num_holes_range"):
        return A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.03, 0.20),
            hole_width_range=(0.03, 0.20),
            fill=0,
            p=p,
        )
    return A.CoarseDropout(
        max_holes=8,
        max_height=64,
        max_width=64,
        min_holes=1,
        min_height=8,
        min_width=8,
        fill_value=0,
        p=p,
    )


def _gauss_noise(p: float):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    if _has_param(A.GaussNoise, "std_range"):
        # Roughly matches var_limit=(5..50) for uint8 images (std ~= 1..5 px).
        return A.GaussNoise(std_range=(0.005, 0.02), p=p)
    return A.GaussNoise(var_limit=(5.0, 50.0), p=p)


def get_train_augment(patch_size: int | tuple[int, int] | None = None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    hw = _as_hw(patch_size)
    border_mode = cv2.BORDER_CONSTANT if cv2 is not None else 0

    affine_kwargs: dict = {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.05, 0.05),
        "rotate": (-180, 180),
        "p": 0.75,
        **_fill_kwargs(A.Affine, fill_value=0),
    }
    if _has_param(A.Affine, "interpolation"):
        affine_kwargs["interpolation"] = cv2.INTER_LINEAR if cv2 is not None else 1
    if _has_param(A.Affine, "mask_interpolation"):
        affine_kwargs["mask_interpolation"] = cv2.INTER_NEAREST if cv2 is not None else 0
    if _has_param(A.Affine, "border_mode"):
        affine_kwargs["border_mode"] = border_mode
    elif _has_param(A.Affine, "mode"):
        affine_kwargs["mode"] = border_mode

    transforms = [
        # Geometric transforms (copy-move can include flips/rotation/scale).
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(**affine_kwargs),
    ]

    if hw is not None:
        crop_h, crop_w = hw
        transforms.append(_random_resized_crop((crop_h, crop_w), scale=(0.75, 1.0), ratio=(0.85, 1.15), p=0.35))

    transforms.extend(
        [
            # Photometric / intensity transforms (robustness to acquisition variation).
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.CLAHE(clip_limit=(1.0, 3.0), p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    _gauss_noise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.25,
            ),
            # Local deformations / occlusions (avoid relying on brittle artifacts).
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1.0,
                        sigma=40.0,
                        border_mode=border_mode,
                        p=1.0,
                        **_fill_kwargs(A.ElasticTransform, fill_value=0),
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.05,
                        border_mode=border_mode,
                        p=1.0,
                        **_fill_kwargs(A.GridDistortion, fill_value=0),
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.05,
                        border_mode=border_mode,
                        p=1.0,
                        **_fill_kwargs(A.OpticalDistortion, fill_value=0),
                        **({"shift_limit": 0.05} if _has_param(A.OpticalDistortion, "shift_limit") else {}),
                    ),
                ],
                p=0.10,
            ),
            _coarse_dropout(p=0.20),
            _image_compression(quality_range=(60, 100), p=0.10),
        ]
    )

    return A.Compose(transforms)


def get_val_augment(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    return A.Compose([])
