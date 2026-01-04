from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .image import letterbox_reflect, unletterbox
from .postprocess import PostprocessParams, postprocess_prob
from .rle import masks_to_annotation
from .tta import HFlipTTA, IdentityTTA, TTATransform, ZoomOutTTA, predict_with_tta


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


@torch.no_grad()
def predict_prob_map(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    input_size: int,
    device: torch.device,
    tta_transforms: list[TTATransform] | None = None,
    tta_weights: list[float] | None = None,
) -> np.ndarray:
    padded, meta = letterbox_reflect(image, input_size)
    x = torch.from_numpy(padded).permute(2, 0, 1).contiguous().float() / 255.0
    x = x.unsqueeze(0).to(device)

    if tta_transforms is None:
        tta_transforms = [IdentityTTA()]
        tta_weights = [1.0]

    model = model.to(device)
    model.eval()
    prob = predict_with_tta(model, x, transforms=tta_transforms, weights=tta_weights)[0, 0]
    prob_np = prob.detach().cpu().numpy().astype(np.float32)
    prob_orig = unletterbox(prob_np, meta)
    return prob_orig


def default_tta(
    *,
    zoom_scale: float = 0.9,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> tuple[list[TTATransform], list[float]]:
    transforms: list[TTATransform] = [IdentityTTA(), HFlipTTA(), ZoomOutTTA(scale=float(zoom_scale))]
    return transforms, list(weights)


def predict_annotation(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    input_size: int,
    device: torch.device,
    post: PostprocessParams,
    tta_transforms: list[TTATransform] | None = None,
    tta_weights: list[float] | None = None,
) -> str:
    prob = predict_prob_map(
        model,
        image,
        input_size=input_size,
        device=device,
        tta_transforms=tta_transforms,
        tta_weights=tta_weights,
    )
    instances = postprocess_prob(prob, post)
    return masks_to_annotation(instances)
