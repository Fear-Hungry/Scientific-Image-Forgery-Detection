from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2_decoder import DinoV2EncoderSpec

FuseMode = Literal["concat", "sum"]


def _group_norm(num_channels: int, *, max_groups: int = 8) -> nn.GroupNorm:
    for g in range(int(min(max_groups, num_channels)), 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = _group_norm(out_channels)
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        return self.dropout(x)


class DinoDeepDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int = 256,
        out_channels: int = 1,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(depth) <= 0:
            raise ValueError("decoder depth must be > 0")
        blocks: list[nn.Module] = []
        ch = int(in_channels)
        for _ in range(int(depth)):
            blocks.append(_ConvBlock(ch, int(hidden_channels), dropout=float(dropout)))
            ch = int(hidden_channels)
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(ch, int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


@dataclass(frozen=True)
class MultiScaleSpec:
    """
    Multi-layer feature extraction spec for DINOv2 ViTs (timm).

    `layers` accepts either:
      - 0-based block indices (e.g. [2, 5, 8, 11])
      - 1-based "layer numbers" when the max equals num_blocks (e.g. [3, 6, 9, 12] for a 12-block ViT)
    """

    layers: list[int] = field(default_factory=lambda: [2, 5, 8, 11])
    proj_channels: int = 256
    fuse: FuseMode = "concat"
    decoder_depth: int = 4


def _canonicalize_layers(layers: Sequence[int], *, num_blocks: int) -> list[int]:
    if not layers:
        raise ValueError("multiscale.layers must be non-empty")
    raw = [int(x) for x in layers]
    mx = max(raw)
    idx = list(raw)

    if mx == int(num_blocks):
        idx = [int(x) - 1 for x in idx]

    if min(idx) < 0 or max(idx) >= int(num_blocks):
        raise ValueError(
            f"multiscale.layers out of range for encoder blocks={num_blocks}: {layers} "
            "(use 0-based indices or 1-based up to num_blocks)"
        )

    out: list[int] = []
    seen: set[int] = set()
    for x in idx:
        if x not in seen:
            out.append(int(x))
            seen.add(int(x))
    return out


class DinoV2MultiScaleSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: DinoV2EncoderSpec = DinoV2EncoderSpec(),
        *,
        decoder_hidden_channels: int = 256,
        decoder_dropout: float = 0.0,
        freeze_encoder: bool = True,
        multiscale: MultiScaleSpec = MultiScaleSpec(),
    ) -> None:
        super().__init__()

        self.encoder = timm.create_model(
            encoder.model_name,
            pretrained=bool(encoder.pretrained) and encoder.checkpoint_path is None,
            checkpoint_path=str(encoder.checkpoint_path) if encoder.checkpoint_path else "",
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        embed_dim = getattr(self.encoder, "embed_dim", None)
        if embed_dim is None:
            raise ValueError(f"Expected timm ViT-like encoder with embed_dim, got {type(self.encoder)}")

        blocks = getattr(self.encoder, "blocks", None)
        num_blocks = len(blocks) if blocks is not None else 0
        if int(num_blocks) <= 0:
            raise ValueError(f"Expected ViT-like encoder with blocks, got {type(self.encoder)}")
        layer_indices = _canonicalize_layers(multiscale.layers, num_blocks=int(num_blocks))
        self._layer_indices = layer_indices
        self.multiscale = multiscale

        proj_channels = int(multiscale.proj_channels)
        if proj_channels <= 0:
            raise ValueError("multiscale.proj_channels must be > 0")
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(embed_dim), proj_channels, kernel_size=1),
                    _group_norm(proj_channels),
                    nn.GELU(),
                )
                for _ in layer_indices
            ]
        )

        if multiscale.fuse == "concat":
            in_ch = proj_channels * len(layer_indices)
        elif multiscale.fuse == "sum":
            in_ch = proj_channels
        else:
            raise ValueError(f"Unknown multiscale.fuse: {multiscale.fuse}")

        self.decoder = DinoDeepDecoder(
            in_ch,
            hidden_channels=int(decoder_hidden_channels),
            out_channels=1,
            depth=int(multiscale.decoder_depth),
            dropout=float(decoder_dropout),
        )

        cfg = getattr(self.encoder, "default_cfg", {}) or {}
        mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
        if len(mean) != 3 or len(std) != 3:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        self.register_buffer("_in_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("_in_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._in_mean.to(device=x.device, dtype=x.dtype)
        std = self._in_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + 1e-6)

    def _forward_intermediates(self, x: torch.Tensor) -> list[torch.Tensor]:
        if hasattr(self.encoder, "forward_intermediates"):
            out = self.encoder.forward_intermediates(
                x,
                indices=list(self._layer_indices),
                return_prefix_tokens=False,
                norm=False,
                output_fmt="NCHW",
                intermediates_only=True,
            )
            if isinstance(out, tuple):
                out = out[1]
            if not isinstance(out, list):
                raise RuntimeError(f"Unexpected forward_intermediates output: {type(out)}")
            return out

        if hasattr(self.encoder, "get_intermediate_layers"):
            out = self.encoder.get_intermediate_layers(
                x,
                n=list(self._layer_indices),
                reshape=True,
                return_prefix_tokens=False,
                norm=False,
            )
            if not isinstance(out, list):
                raise RuntimeError(f"Unexpected get_intermediate_layers output: {type(out)}")
            return out

        raise RuntimeError(
            f"Encoder {type(self.encoder)} does not support intermediate extraction. "
            "Upgrade timm or use model.type=dinov2."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self._normalize_input(x)
        inter = self._forward_intermediates(x_norm)
        if len(inter) != len(self.proj):
            raise RuntimeError(f"Expected {len(self.proj)} intermediates, got {len(inter)}")

        feats: list[torch.Tensor] = []
        for proj, feat in zip(self.proj, inter, strict=True):
            if feat.ndim != 4:
                raise RuntimeError(f"Expected feature map (B,C,H,W), got shape={tuple(feat.shape)}")
            feats.append(proj(feat))

        if self.multiscale.fuse == "concat":
            fused = torch.cat(feats, dim=1)
        else:
            fused = feats[0]
            for f in feats[1:]:
                fused = fused + f

        logits_patch = self.decoder(fused)
        return F.interpolate(
            logits_patch,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

