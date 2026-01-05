from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class TTATransform:
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityTTA(TTATransform):
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        return y


@dataclass(frozen=True)
class HFlipTTA(TTATransform):
    dim: int = -1  # width

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(self.dim,))

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        return torch.flip(y, dims=(self.dim,))


@dataclass(frozen=True)
class VFlipTTA(TTATransform):
    dim: int = -2  # height

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(self.dim,))

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        return torch.flip(y, dims=(self.dim,))


@dataclass(frozen=True)
class Rot90TTA(TTATransform):
    k: int = 1
    dims: tuple[int, int] = (-2, -1)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=int(self.k), dims=self.dims)

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        return torch.rot90(y, k=-int(self.k), dims=self.dims)


@dataclass(frozen=True)
class ZoomOutTTA(TTATransform):
    scale: float = 0.9
    pad_mode: str = "reflect"

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        new_h = max(1, int(round(h * self.scale)))
        new_w = max(1, int(round(w * self.scale)))
        x_small = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left

        if pad_top == pad_bottom == pad_left == pad_right == 0:
            return x_small
        return F.pad(x_small, (pad_left, pad_right, pad_top, pad_bottom), mode=self.pad_mode)

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        b, c, h, w = y.shape
        new_h = max(1, int(round(h * self.scale)))
        new_w = max(1, int(round(w * self.scale)))

        pad_top = (h - new_h) // 2
        pad_left = (w - new_w) // 2
        cropped = y[..., pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


@dataclass(frozen=True)
class ZoomInTTA(TTATransform):
    """
    Center crop + resize back (a "crop TTA").

    `invert()` places the resized prediction back into the center and fills the outside with zeros.
    """

    scale: float = 1.1

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        scale = float(self.scale)
        if scale <= 1.0:
            return x
        crop_h = max(1, int(round(h / scale)))
        crop_w = max(1, int(round(w / scale)))
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        x_crop = x[..., top : top + crop_h, left : left + crop_w]
        return F.interpolate(x_crop, size=(h, w), mode="bilinear", align_corners=False)

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        b, c, h, w = y.shape
        scale = float(self.scale)
        if scale <= 1.0:
            return y
        crop_h = max(1, int(round(h / scale)))
        crop_w = max(1, int(round(w / scale)))
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        y_small = F.interpolate(y, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
        out = torch.zeros_like(y)
        out[..., top : top + crop_h, left : left + crop_w] = y_small
        return out


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    transforms: list[TTATransform],
    weights: list[float] | None = None,
) -> torch.Tensor:
    if weights is None:
        weights = [1.0 / len(transforms)] * len(transforms)
    if len(weights) != len(transforms):
        raise ValueError("weights and transforms must have the same length")

    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("sum(weights) must be > 0")
    weights = [w / total_w for w in weights]

    prob_sum = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=torch.float32)
    for tta, w in zip(transforms, weights, strict=True):
        x_t = tta.apply(x)
        logits = model(x_t)
        prob = torch.sigmoid(logits)
        prob = tta.invert(prob)
        prob_sum += prob.float() * float(w)
    return prob_sum
