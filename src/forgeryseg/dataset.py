from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .typing import Case, Pathish, Split


class RecodaiSample(NamedTuple):
    case_id: str
    image: torch.Tensor  # (3, H, W), float32 in [0, 1]
    mask: torch.Tensor  # (1, H, W), float32 in {0, 1}
    is_forged: bool


def list_cases(
    data_root: Pathish,
    split: Split,
    *,
    include_authentic: bool = True,
    include_forged: bool = True,
) -> list[Case]:
    root = Path(data_root)
    cases: list[Case] = []

    if split == "train":
        if include_authentic:
            for img_path in sorted((root / "train_images" / "authentic").glob("*.png")):
                cases.append(Case(case_id=img_path.stem, image_path=img_path, mask_path=None))
        if include_forged:
            for img_path in sorted((root / "train_images" / "forged").glob("*.png")):
                mask_path = root / "train_masks" / f"{img_path.stem}.npy"
                cases.append(Case(case_id=img_path.stem, image_path=img_path, mask_path=mask_path))

    elif split == "supplemental":
        if include_authentic:
            for img_path in sorted((root / "supplemental_images" / "authentic").glob("*.png")):
                cases.append(Case(case_id=img_path.stem, image_path=img_path, mask_path=None))
        if include_forged:
            for img_path in sorted((root / "supplemental_images" / "forged").glob("*.png")):
                mask_path = root / "supplemental_masks" / f"{img_path.stem}.npy"
                cases.append(Case(case_id=img_path.stem, image_path=img_path, mask_path=mask_path))

    elif split == "test":
        for img_path in sorted((root / "test_images").glob("*.png")):
            cases.append(Case(case_id=img_path.stem, image_path=img_path, mask_path=None))

    else:
        raise ValueError(f"Unknown split: {split}")

    return cases


def load_mask_instances(mask_path: Path) -> list[np.ndarray]:
    masks = np.load(mask_path)
    if masks.ndim != 3:
        raise ValueError(f"Expected (N, H, W) mask array, got shape={masks.shape} at {mask_path}")
    return [m.astype(np.uint8) for m in masks]


def union_mask(instances: list[np.ndarray], *, shape: tuple[int, int] | None = None) -> np.ndarray:
    if len(instances) == 0:
        if shape is None:
            raise ValueError("shape is required when instances is empty")
        return np.zeros(shape, dtype=np.uint8)
    stacked = np.stack(instances, axis=0).astype(bool)
    return np.any(stacked, axis=0).astype(np.uint8)


class RecodaiDataset(Dataset[RecodaiSample]):
    def __init__(
        self,
        data_root: Pathish,
        split: Split,
        *,
        training: bool | None = None,
        include_authentic: bool = True,
        include_forged: bool = True,
        transforms: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        image_mode: Literal["rgb", "l"] = "rgb",
        cache_images: bool = False,
        cache_masks: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.training = bool(training) if training is not None else split in {"train", "supplemental"}
        self.cases = list_cases(
            self.data_root,
            split,
            include_authentic=include_authentic,
            include_forged=include_forged,
        )
        self.transforms = transforms
        self.image_mode = image_mode
        self.cache_images = bool(cache_images)
        self.cache_masks = bool(cache_masks)

        self._image_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> RecodaiSample:
        case = self.cases[idx]
        if self.cache_images and case.case_id in self._image_cache:
            image = self._image_cache[case.case_id]
        else:
            with Image.open(case.image_path) as img:
                img = img.convert("RGB" if self.image_mode == "rgb" else "L")
                image = np.array(img)
            if self.cache_images:
                self._image_cache[case.case_id] = image
        if self.image_mode == "l":
            image = np.repeat(image[..., None], 3, axis=2)

        is_forged = case.mask_path is not None

        if not self.training:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        elif case.mask_path is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            if self.cache_masks and case.case_id in self._mask_cache:
                mask = self._mask_cache[case.case_id]
            else:
                instances = load_mask_instances(case.mask_path)
                mask = union_mask(instances, shape=image.shape[:2])
                if self.cache_masks:
                    self._mask_cache[case.case_id] = mask

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        mask_t = torch.from_numpy(mask[None, ...].astype(np.float32)).contiguous()
        return RecodaiSample(
            case_id=case.case_id,
            image=image_t,
            mask=mask_t,
            is_forged=is_forged,
        )
