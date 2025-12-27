from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .augment import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class Sample:
    case_id: str
    image_path: Path
    mask_path: Optional[Path]
    is_authentic: Optional[bool]
    split: str
    label: Optional[str]
    rel_path: Path


def build_train_index(data_root: str | Path, strict: bool = False) -> List[Sample]:
    data_root = Path(data_root)
    train_root = data_root / "train_images"
    mask_root = data_root / "train_masks"

    samples: List[Sample] = []
    for label in ("authentic", "forged"):
        for image_path in sorted((train_root / label).glob("*.png")):
            case_id = image_path.stem
            mask_path = None
            if label == "forged":
                candidate = mask_root / f"{case_id}.npy"
                if candidate.exists():
                    mask_path = candidate
                elif strict:
                    raise FileNotFoundError(f"Missing mask for {case_id}")
            samples.append(
                Sample(
                    case_id=case_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    is_authentic=(label == "authentic"),
                    split="train",
                    label=label,
                    rel_path=image_path.relative_to(data_root),
                )
            )
    return samples


def build_supplemental_index(data_root: str | Path, strict: bool = False) -> List[Sample]:
    data_root = Path(data_root)
    image_root = data_root / "supplemental_images"
    mask_root = data_root / "supplemental_masks"

    samples: List[Sample] = []
    for image_path in sorted(image_root.glob("*.png")):
        case_id = image_path.stem
        mask_path = None
        candidate = mask_root / f"{case_id}.npy"
        if candidate.exists():
            mask_path = candidate
        elif strict:
            raise FileNotFoundError(f"Missing supplemental mask for {case_id}")
        samples.append(
            Sample(
                case_id=case_id,
                image_path=image_path,
                mask_path=mask_path,
                is_authentic=False if mask_path is not None else None,
                split="supplemental",
                label=None,
                rel_path=image_path.relative_to(data_root),
            )
        )
    return samples


def build_test_index(data_root: str | Path) -> List[Sample]:
    data_root = Path(data_root)
    test_root = data_root / "test_images"

    samples: List[Sample] = []
    for image_path in sorted(test_root.glob("*.png")):
        case_id = image_path.stem
        samples.append(
            Sample(
                case_id=case_id,
                image_path=image_path,
                mask_path=None,
                is_authentic=None,
                split="test",
                label=None,
                rel_path=image_path.relative_to(data_root),
            )
        )
    return samples


def load_image(image_path: str | Path, as_rgb: bool = True) -> np.ndarray:
    from PIL import Image

    image_path = Path(image_path)
    with Image.open(image_path) as img:
        if as_rgb:
            img = img.convert("RGB")
        return np.array(img)


def load_mask_instances(mask_path: str | Path) -> List[np.ndarray]:
    mask_path = Path(mask_path)
    masks = np.load(mask_path)
    if masks.ndim == 2:
        masks = masks[None, ...]
    instances = [(m > 0).astype(np.uint8) for m in masks]
    return instances


def load_union_mask(mask_path: Optional[Path], shape: tuple[int, int]) -> np.ndarray:
    if mask_path is None:
        return np.zeros(shape, dtype=np.uint8)
    masks = np.load(mask_path)
    if masks.ndim == 2:
        union = masks
    else:
        union = masks.max(axis=0)
    union = (union > 0).astype(np.uint8)
    if union.shape != shape:
        raise ValueError(f"Mask shape {union.shape} does not match image shape {shape}")
    return union


def _pad_to_size(image: np.ndarray, mask: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    mask_pad = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return image_pad, mask_pad


def _random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_h: int,
    crop_w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    if h == crop_h and w == crop_w:
        return image, mask
    top = int(rng.integers(0, h - crop_h + 1))
    left = int(rng.integers(0, w - crop_w + 1))
    return image[top : top + crop_h, left : left + crop_w], mask[top : top + crop_h, left : left + crop_w]


def _center_crop(image: np.ndarray, mask: np.ndarray, crop_h: int, crop_w: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return image[top : top + crop_h, left : left + crop_w], mask[top : top + crop_h, left : left + crop_w]


def _positive_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_h: int,
    crop_w: int,
    rng: np.random.Generator,
    max_tries: int,
    min_pos_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return _random_crop(image, mask, crop_h, crop_w, rng)

    h, w = mask.shape
    for _ in range(max_tries):
        idx = int(rng.integers(0, len(ys)))
        center_y = int(ys[idx])
        center_x = int(xs[idx])
        top = max(min(center_y - crop_h // 2, h - crop_h), 0)
        left = max(min(center_x - crop_w // 2, w - crop_w), 0)
        crop_mask = mask[top : top + crop_h, left : left + crop_w]
        if int(crop_mask.sum()) >= min_pos_pixels:
            return image[top : top + crop_h, left : left + crop_w], crop_mask
    return _random_crop(image, mask, crop_h, crop_w, rng)


class PatchDataset:
    def __init__(
        self,
        samples: List[Sample],
        patch_size: int | tuple[int, int] = 512,
        train: bool = True,
        augment=None,
        positive_prob: float = 0.7,
        min_pos_pixels: int = 1,
        max_tries: int = 10,
        seed: int = 42,
        return_meta: bool = False,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        normalize: bool = True,
    ) -> None:
        self.samples = samples
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.train = train
        self.augment = augment
        self.positive_prob = positive_prob
        self.min_pos_pixels = min_pos_pixels
        self.max_tries = max_tries
        self.seed = seed
        self.return_meta = return_meta
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        sample = self.samples[idx]
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + idx + worker_id * 100000)
        image = load_image(sample.image_path)
        mask = load_union_mask(sample.mask_path, image.shape[:2])

        crop_h, crop_w = self.patch_size
        image, mask = _pad_to_size(image, mask, crop_h, crop_w)
        if self.train:
            wants_positive = (sample.is_authentic is False) and (rng.random() < self.positive_prob)
            if wants_positive:
                image, mask = _positive_crop(
                    image,
                    mask,
                    crop_h,
                    crop_w,
                    rng,
                    self.max_tries,
                    self.min_pos_pixels,
                )
            else:
                image, mask = _random_crop(image, mask, crop_h, crop_w, rng)
        else:
            image, mask = _center_crop(image, mask, crop_h, crop_w)

        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        if self.normalize:
            image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        mask = mask.astype(np.float32)[None, ...]

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        if self.return_meta:
            return image_tensor, mask_tensor, sample
        return image_tensor, mask_tensor
