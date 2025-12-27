from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .models import builders
from .models.classifier import build_classifier


def load_checkpoint(path: str | Path) -> tuple[dict, dict]:
    """Load a checkpoint saved by the notebooks (state_dict + config)."""
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("config", {})
    if isinstance(ckpt, dict):
        return ckpt, {}
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def build_segmentation_from_config(cfg: dict[str, Any]) -> nn.Module:
    """
    Build a segmentation model matching the config dictionary saved in a checkpoint.

    Notes:
    - We force `encoder_weights=None` because checkpoints already include full weights and Kaggle may run offline.
    """
    backend = str(cfg.get("backend", "smp"))
    arch = str(cfg.get("arch", cfg.get("model_id", "unetplusplus"))).lower()
    classes = int(cfg.get("classes", 1))

    if backend == "smp":
        encoder_name = str(cfg.get("encoder_name", "efficientnet-b4"))
        encoder_depth = int(cfg.get("encoder_depth", 5))

        if arch in {"unetplusplus", "unetpp"}:
            return builders.build_unetplusplus(
                encoder_name=encoder_name,
                encoder_weights=None,
                encoder_depth=encoder_depth,
                classes=classes,
                strict_weights=True,
            )
        if arch == "unet":
            return builders.build_unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                encoder_depth=encoder_depth,
                classes=classes,
                strict_weights=True,
            )
        if arch in {"deeplabv3plus", "deeplabv3+", "deeplabv3p"}:
            return builders.build_deeplabv3plus(
                encoder_name=encoder_name,
                encoder_weights=None,
                encoder_depth=encoder_depth,
                classes=classes,
                strict_weights=True,
            )
        if arch in {"segformer", "mit"}:
            encoder_name = str(cfg.get("encoder_name", cfg.get("segformer_encoder", "mit_b2")))
            return builders.build_segformer(
                encoder_name=encoder_name,
                encoder_weights=None,
                classes=classes,
                strict_weights=True,
            )

        raise ValueError(f"Unknown SMP segmentation arch: {arch!r}")

    if backend == "torchvision":
        # Minimal fallback for older notebooks/configs.
        if "deeplab" not in arch:
            raise ValueError(f"Unsupported torchvision segmentation arch: {arch!r}")
        from torchvision.models.segmentation import deeplabv3_resnet50

        try:
            base = deeplabv3_resnet50(weights=None, weights_backbone=None)
        except TypeError:
            base = deeplabv3_resnet50(pretrained=False)

        head = base.classifier[-1]
        base.classifier[-1] = nn.Conv2d(head.in_channels, classes, kernel_size=1)

        class _Wrap(nn.Module):
            def __init__(self, m: nn.Module):
                super().__init__()
                self.m = m

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.m(x)
                if isinstance(out, dict):
                    out = out["out"]
                return out

        return _Wrap(base)

    raise ValueError(f"Unknown segmentation backend: {backend!r}")


def build_classifier_from_config(cfg: dict[str, Any]) -> tuple[nn.Module, int]:
    """Build a classifier model matching checkpoint config. Returns (model, image_size)."""
    backend = str(cfg.get("backend", "timm"))
    model_name = str(cfg.get("model_name", "tf_efficientnet_b4_ns"))
    image_size = int(cfg.get("image_size", 384))

    if backend == "timm":
        return build_classifier(model_name=model_name, pretrained=False, num_classes=1), image_size

    if backend == "torchvision":
        return build_classifier(model_name="resnet50", pretrained=False, num_classes=1), image_size

    raise ValueError(f"Unknown classifier backend: {backend!r}")

