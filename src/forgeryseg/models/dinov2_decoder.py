from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoTinyDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int = 256,
        out_channels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, hidden_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.gn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.gelu(self.gn2(self.conv2(x)))
        return self.head(x)


@dataclass(frozen=True)
class DinoV2EncoderSpec:
    model_name: str = "vit_base_patch14_dinov2"
    checkpoint_path: str | Path | None = None
    pretrained: bool = False


class DinoV2SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: DinoV2EncoderSpec = DinoV2EncoderSpec(),
        *,
        decoder_hidden_channels: int = 256,
        decoder_dropout: float = 0.0,
        freeze_encoder: bool = True,
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

        self.decoder = DinoTinyDecoder(
            int(embed_dim),
            hidden_channels=decoder_hidden_channels,
            out_channels=1,
            dropout=decoder_dropout,
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @property
    def patch_size(self) -> tuple[int, int]:
        patch = getattr(getattr(self.encoder, "patch_embed", None), "patch_size", None)
        if patch is None:
            return (14, 14)
        if isinstance(patch, int):
            return (patch, patch)
        return (int(patch[0]), int(patch[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(x)  # (B, 1 + HW, C)
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise RuntimeError(f"Unexpected encoder output shape: {tokens.shape}")

        patch_tokens = tokens[:, 1:, :]
        b, n, c = patch_tokens.shape
        patch_h, patch_w = self.patch_size
        grid_h = x.shape[2] // patch_h
        grid_w = x.shape[3] // patch_w
        if grid_h * grid_w != n:
            grid = int(math.sqrt(n))
            if grid * grid != n:
                raise RuntimeError(f"Cannot infer token grid from n={n} for input shape={tuple(x.shape)}")
            grid_h = grid_w = grid

        feat = patch_tokens.transpose(1, 2).reshape(b, c, grid_h, grid_w)
        logits_patch = self.decoder(feat)
        return F.interpolate(
            logits_patch,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
