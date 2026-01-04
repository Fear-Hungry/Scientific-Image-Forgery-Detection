from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import torch

Split = Literal["train", "test", "supplemental"]


@dataclass(frozen=True)
class Case:
    case_id: str
    image_path: Path
    mask_path: Path | None


NpBoolMask = np.ndarray  # (H, W), dtype=bool or uint8
TorchFloat = torch.Tensor
MaskList = Sequence[NpBoolMask]
Pathish = str | Path

