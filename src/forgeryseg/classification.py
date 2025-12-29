from __future__ import annotations

from typing import Sequence

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover - optional dependency
    T = None

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    Dataset = object  # type: ignore[assignment,misc]

from PIL import Image

from .dataset import Sample

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_classification_transform(
    *,
    image_size: int,
    train: bool,
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
    brightness: float = 0.2,
    contrast: float = 0.2,
    grayscale_prob: float = 0.1,
    blur_prob: float = 0.1,
    cutout_prob: float = 0.2,
) -> "T.Compose":
    if T is None:
        raise ImportError("torchvision is required for classification transforms")

    aug: list[object] = []
    if train:
        aug.append(T.RandomHorizontalFlip(p=float(p_hflip)))
        aug.append(T.RandomVerticalFlip(p=float(p_vflip)))
        if float(grayscale_prob) > 0:
            if not hasattr(T, "RandomGrayscale"):
                raise ImportError("torchvision RandomGrayscale is required for grayscale augmentation")
            aug.append(T.RandomGrayscale(p=float(grayscale_prob)))
        if brightness or contrast:
            aug.append(T.ColorJitter(brightness=float(brightness), contrast=float(contrast)))
        if float(blur_prob) > 0:
            if not hasattr(T, "GaussianBlur"):
                raise ImportError("torchvision GaussianBlur is required for blur augmentation")
            aug.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))

    aug.extend(
        [
            T.Resize((int(image_size), int(image_size))),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if train and float(cutout_prob) > 0:
        if not hasattr(T, "RandomErasing"):
            raise ImportError("torchvision RandomErasing is required for cutout augmentation")
        aug.append(
            T.RandomErasing(
                p=float(cutout_prob),
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value=0,
            )
        )
    return T.Compose(aug)


class BinaryForgeryClsDataset(Dataset):
    """Binary classification dataset: authentic (0) vs forged (1)."""

    def __init__(self, samples: Sequence[Sample], transform: "T.Compose") -> None:
        if torch is None:
            raise ImportError("torch is required for BinaryForgeryClsDataset")
        if T is None:
            raise ImportError("torchvision is required for BinaryForgeryClsDataset")
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[int(idx)]
        if s.is_authentic is None:
            raise ValueError("BinaryForgeryClsDataset requires samples with known is_authentic")
        with Image.open(s.image_path) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        y = torch.tensor([0.0 if s.is_authentic else 1.0], dtype=torch.float32)
        return x, y
