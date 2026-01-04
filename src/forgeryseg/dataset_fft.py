from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import list_cases
from .frequency import FFTParams, fft_tensor
from .typing import Pathish, Split


class FFTClsSample(NamedTuple):
    case_id: str
    x: torch.Tensor  # (1, H, W)
    y: torch.Tensor  # (), float32 in {0,1}


class RecodaiFFTDataset(Dataset[FFTClsSample]):
    def __init__(
        self,
        data_root: Pathish,
        split: Split,
        *,
        fft_params: FFTParams,
        include_authentic: bool = True,
        include_forged: bool = True,
        image_transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.cases = list_cases(
            self.data_root,
            split,
            include_authentic=include_authentic,
            include_forged=include_forged,
        )
        self.fft_params = fft_params
        self.image_transforms = image_transforms

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> FFTClsSample:
        case = self.cases[idx]
        with Image.open(case.image_path) as img:
            image_rgb = np.array(img.convert("RGB"))

        x = fft_tensor(image_rgb, self.fft_params)
        if self.image_transforms is not None:
            x = self.image_transforms(x)

        y = torch.tensor(1.0 if case.mask_path is not None else 0.0, dtype=torch.float32)
        return FFTClsSample(case_id=case.case_id, x=x, y=y)
