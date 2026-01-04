from __future__ import annotations

from typing import Callable, Literal

import numpy as np

AugMode = Literal["none", "basic", "robust"]
TransformFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


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
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.15,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.MotionBlur(blur_limit=7, p=1.0),
                        ],
                        p=0.2,
                    ),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                ]
            )

    aug_tf = A.Compose(base)

    def _apply(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = aug_tf(image=img, mask=mask)
        return out["image"], out["mask"]

    return _apply

