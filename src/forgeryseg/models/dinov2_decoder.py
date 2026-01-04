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
        tokens = self.encoder.forward_features(x)  # (B, 1 + (reg) + HW, C)
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise RuntimeError(f"Unexpected encoder output shape: {tokens.shape}")

        # Robust token parsing across timm versions:
        # - some DINOv2 variants add register tokens after cls (ex.: reg4 => 1 cls + 4 reg)
        # - older timm versions may not expose `num_prefix_tokens` / `patch_embed.grid_size`
        patch_h, patch_w = self.patch_size
        token_count = int(tokens.shape[1])

        cands: list[tuple[int, int, int]] = []  # (n_patches, grid_h, grid_w)
        grid_size = getattr(getattr(self.encoder, "patch_embed", None), "grid_size", None)
        if grid_size is not None:
            try:
                gh, gw = int(grid_size[0]), int(grid_size[1])
                cands.append((gh * gw, gh, gw))
            except Exception:
                pass

        # dynamic_img_pad=True implies ceil; keep floor as fallback for older timm / non-padded inputs
        gh_ceil = int(math.ceil(x.shape[2] / patch_h))
        gw_ceil = int(math.ceil(x.shape[3] / patch_w))
        gh_floor = int(max(1, x.shape[2] // patch_h))
        gw_floor = int(max(1, x.shape[3] // patch_w))
        cands.extend([(gh_ceil * gw_ceil, gh_ceil, gw_ceil), (gh_floor * gw_floor, gh_floor, gw_floor)])

        # Prefer the largest patch grid that yields a valid positive prefix (minimizes extra tokens).
        chosen: tuple[int, int, int] | None = None
        for n_patches, gh, gw in sorted(set(cands), key=lambda t: t[0], reverse=True):
            prefix = token_count - int(n_patches)
            if prefix < 1:
                continue
            if int(n_patches) <= 0:
                continue
            # Validate: after stripping prefix, we must have exactly n_patches tokens.
            if tokens[:, prefix:, :].shape[1] == int(n_patches):
                chosen = (int(n_patches), int(gh), int(gw))
                break

        if chosen is None:
            # Last resort: use timm hint if present, then infer square grid.
            prefix = int(getattr(self.encoder, "num_prefix_tokens", 1))
            patch_tokens = tokens[:, prefix:, :]
            n = int(patch_tokens.shape[1])
            grid = int(math.sqrt(n))
            if grid * grid != n:
                raise RuntimeError(f"Cannot infer token grid from n={n} for input shape={tuple(x.shape)}")
            grid_h = grid_w = grid
        else:
            n_patches, grid_h, grid_w = chosen
            prefix = token_count - int(n_patches)
            patch_tokens = tokens[:, prefix:, :]
            n = int(patch_tokens.shape[1])
            if n != int(n_patches):
                raise RuntimeError(
                    f"Token parsing failed: expected n_patches={n_patches} got n={n} (prefix={prefix}) for input shape={tuple(x.shape)}"
                )

        b, n, c = patch_tokens.shape

        feat = patch_tokens.transpose(1, 2).reshape(b, c, grid_h, grid_w)
        logits_patch = self.decoder(feat)
        return F.interpolate(
            logits_patch,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
