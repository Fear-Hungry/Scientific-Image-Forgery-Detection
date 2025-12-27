from __future__ import annotations

import inspect
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np

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


if A is not None:
    from albumentations.core.transforms_interface import DualTransform

    class CopyMoveTransform(DualTransform):
        """Synthetic copy-move augmentation for training.

        If the incoming mask is empty (authentic sample), copies a random region from the image and pastes it into
        another location, and marks both source and destination regions as forged in the mask.
        """

        def __init__(
            self,
            min_area_frac: float = 0.05,
            max_area_frac: float = 0.20,
            rotation_limit: float = 15.0,
            scale_range: tuple[float, float] = (0.9, 1.1),
            irregular_prob: float = 0.5,
            max_tries: int = 10,
            only_if_empty_mask: bool = True,
            p: float = 0.25,
        ) -> None:
            super().__init__(p=float(p))
            self.min_area_frac = float(min_area_frac)
            self.max_area_frac = float(max_area_frac)
            self.rotation_limit = float(rotation_limit)
            self.scale_range = (float(scale_range[0]), float(scale_range[1]))
            self.irregular_prob = float(irregular_prob)
            self.max_tries = int(max_tries)
            self.only_if_empty_mask = bool(only_if_empty_mask)

            if not (0.0 < self.min_area_frac <= self.max_area_frac <= 1.0):
                raise ValueError("Expected 0 < min_area_frac <= max_area_frac <= 1")
            if self.max_tries <= 0:
                raise ValueError("max_tries must be > 0")
            if not (0.0 <= self.irregular_prob <= 1.0):
                raise ValueError("irregular_prob must be in [0, 1]")
            if self.scale_range[0] <= 0.0 or self.scale_range[1] <= 0.0:
                raise ValueError("scale_range values must be > 0")

        @property
        def targets_as_params(self):  # type: ignore[override]
            return ["image", "mask"]

        def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
            image = data["image"]
            mask = data.get("mask")
            if mask is None:
                return {"do": False}

            mask = np.asarray(mask)
            if mask.ndim != 2:
                return {"do": False}

            if self.only_if_empty_mask and mask.max() > 0:
                return {"do": False}

            h, w = mask.shape
            if h < 16 or w < 16:
                return {"do": False}

            rg = self.random_generator

            area_frac = float(rg.uniform(self.min_area_frac, self.max_area_frac))
            target_area = area_frac * float(h * w)

            # Prefer near-square regions; allow mild aspect variation.
            aspect = float(np.exp(rg.uniform(np.log(0.75), np.log(1.3333333333333333))))
            patch_h = int(round(np.sqrt(target_area / aspect)))
            patch_w = int(round(patch_h * aspect))
            patch_h = int(np.clip(patch_h, 8, h - 1))
            patch_w = int(np.clip(patch_w, 8, w - 1))

            y_choices = h - patch_h + 1
            x_choices = w - patch_w + 1
            if y_choices <= 0 or x_choices <= 0:
                return {"do": False}
            if y_choices == 1 and x_choices == 1:
                return {"do": False}

            src_y = int(rg.integers(0, y_choices))
            src_x = int(rg.integers(0, x_choices))

            # Pick destination (prefer non-overlapping paste; fallback to any different position).
            chosen: tuple[int, int] | None = None
            for _ in range(self.max_tries):
                cand_y = int(rg.integers(0, y_choices))
                cand_x = int(rg.integers(0, x_choices))
                if cand_y == src_y and cand_x == src_x:
                    continue

                if chosen is None:
                    chosen = (cand_y, cand_x)

                y_overlap = max(0, min(src_y + patch_h, cand_y + patch_h) - max(src_y, cand_y))
                x_overlap = max(0, min(src_x + patch_w, cand_x + patch_w) - max(src_x, cand_x))
                if (y_overlap * x_overlap) == 0:
                    chosen = (cand_y, cand_x)
                    break

            if chosen is None:
                # Deterministic shift (guarantees a different coordinate when possible).
                dst_y = (src_y + 1) % y_choices if y_choices > 1 else src_y
                dst_x = (src_x + 1) % x_choices if x_choices > 1 else src_x
                if dst_y == src_y and dst_x == src_x:
                    return {"do": False}
            else:
                dst_y, dst_x = chosen

            # Build a (possibly irregular) region mask inside the patch window.
            if float(rg.random()) < self.irregular_prob:
                yy, xx = np.mgrid[0:patch_h, 0:patch_w]
                cy = float(patch_h - 1) / 2.0 + float(rg.uniform(-0.15, 0.15)) * patch_h
                cx = float(patch_w - 1) / 2.0 + float(rg.uniform(-0.15, 0.15)) * patch_w
                ry = max(2.0, float(rg.uniform(0.35, 0.55)) * patch_h)
                rx = max(2.0, float(rg.uniform(0.35, 0.55)) * patch_w)
                mask_src = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0).astype(np.uint8)
            else:
                mask_src = np.ones((patch_h, patch_w), dtype=np.uint8)

            if int(mask_src.sum()) == 0:
                mask_src = np.ones((patch_h, patch_w), dtype=np.uint8)

            angle = float(rg.uniform(-self.rotation_limit, self.rotation_limit)) if self.rotation_limit > 0 else 0.0
            scale = float(rg.uniform(self.scale_range[0], self.scale_range[1]))

            # Transform destination mask (what we actually paste) if OpenCV is available.
            if cv2 is not None and (abs(angle) > 1e-6 or abs(scale - 1.0) > 1e-6):
                center = (float(patch_w) / 2.0, float(patch_h) / 2.0)
                mat = cv2.getRotationMatrix2D(center, angle, scale)
                mask_dst = cv2.warpAffine(
                    mask_src,
                    mat,
                    dsize=(patch_w, patch_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(np.uint8)
                if int(mask_dst.sum()) == 0:
                    mask_dst = mask_src
            else:
                mat = None
                mask_dst = mask_src

            return {
                "do": True,
                "src_y": src_y,
                "src_x": src_x,
                "dst_y": dst_y,
                "dst_x": dst_x,
                "patch_h": patch_h,
                "patch_w": patch_w,
                "mask_src": mask_src,
                "mask_dst": mask_dst,
                "mat": mat,
            }

        def apply(self, img: np.ndarray, *args, **params) -> np.ndarray:
            if not params.get("do", False):
                return img

            src_y = int(params["src_y"])
            src_x = int(params["src_x"])
            dst_y = int(params["dst_y"])
            dst_x = int(params["dst_x"])
            patch_h = int(params["patch_h"])
            patch_w = int(params["patch_w"])
            mask_dst = np.asarray(params["mask_dst"]).astype(bool)
            mat = params.get("mat", None)

            out = img.copy()
            src_patch = out[src_y : src_y + patch_h, src_x : src_x + patch_w].copy()

            if mat is not None and cv2 is not None:
                src_patch = cv2.warpAffine(
                    src_patch,
                    mat,
                    dsize=(patch_w, patch_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )

            dst_patch = out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w]
            dst_patch[mask_dst] = src_patch[mask_dst]
            out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = dst_patch
            return out

        def apply_to_mask(self, mask: np.ndarray, *args, **params) -> np.ndarray:
            if not params.get("do", False):
                return mask

            src_y = int(params["src_y"])
            src_x = int(params["src_x"])
            dst_y = int(params["dst_y"])
            dst_x = int(params["dst_x"])
            patch_h = int(params["patch_h"])
            patch_w = int(params["patch_w"])
            mask_src = np.asarray(params["mask_src"]).astype(bool)
            mask_dst = np.asarray(params["mask_dst"]).astype(bool)

            out = (np.asarray(mask) > 0).astype(np.uint8)
            out[src_y : src_y + patch_h, src_x : src_x + patch_w][mask_src] = 1
            out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w][mask_dst] = 1
            return out

else:  # pragma: no cover - albumentations optional
    CopyMoveTransform = None


def get_train_augment(
    patch_size: int | tuple[int, int] | None = None,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    copy_move_prob: float = 0.0,
    copy_move_min_area_frac: float = 0.05,
    copy_move_max_area_frac: float = 0.20,
    copy_move_rotation_limit: float = 15.0,
    copy_move_scale_range: tuple[float, float] = (0.9, 1.1),
):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    hw = _as_hw(patch_size)
    border_mode = cv2.BORDER_CONSTANT if cv2 is not None else 0

    affine_kwargs: dict = {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.05, 0.05),
        "rotate": (-20, 20),
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
        # Synthetic copy-move (applied only when mask is empty).
        *(
            [
                CopyMoveTransform(
                    min_area_frac=copy_move_min_area_frac,
                    max_area_frac=copy_move_max_area_frac,
                    rotation_limit=copy_move_rotation_limit,
                    scale_range=copy_move_scale_range,
                    p=float(copy_move_prob),
                )
            ]
            if CopyMoveTransform is not None and float(copy_move_prob) > 0.0
            else []
        ),
        # Geometric transforms (copy-move can include flips/rotation/scale).
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.25),
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
