from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2_decoder import DinoTinyDecoder, DinoV2EncoderSpec

FreqMode = Literal["logmag", "hp_residual", "phase_only", "lp_hp"]
FreqNormalize = Literal["zscore", "none"]


@dataclass(frozen=True)
class FreqFusionSpec:
    mode: FreqMode = "hp_residual"
    hp_radius_fraction: float = 0.1
    freq_channels: int = 64
    normalize: FreqNormalize = "zscore"


def _group_norm(num_channels: int, *, max_groups: int = 8) -> nn.GroupNorm:
    for g in range(int(max_groups), 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class DinoV2FreqFusionSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: DinoV2EncoderSpec = DinoV2EncoderSpec(),
        *,
        decoder_hidden_channels: int = 256,
        decoder_dropout: float = 0.0,
        freeze_encoder: bool = True,
        freq: FreqFusionSpec = FreqFusionSpec(),
    ) -> None:
        super().__init__()

        self.encoder = timm.create_model(
            encoder.model_name,
            pretrained=False,
            checkpoint_path=str(encoder.checkpoint_path) if encoder.checkpoint_path else "",
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        embed_dim = getattr(self.encoder, "embed_dim", None)
        if embed_dim is None:
            raise ValueError(f"Expected timm ViT-like encoder with embed_dim, got {type(self.encoder)}")

        self.freq = freq
        freq_in_chans = 2 if freq.mode == "lp_hp" else 1
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(freq_in_chans, int(freq.freq_channels), kernel_size=3, padding=1),
            _group_norm(int(freq.freq_channels)),
            nn.GELU(),
            nn.Conv2d(int(freq.freq_channels), int(freq.freq_channels), kernel_size=3, padding=1),
            _group_norm(int(freq.freq_channels)),
            nn.GELU(),
        )

        self.decoder = DinoTinyDecoder(
            int(embed_dim) + int(freq.freq_channels),
            hidden_channels=decoder_hidden_channels,
            out_channels=1,
            dropout=decoder_dropout,
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self._mask_cache: torch.Tensor | None = None
        self._mask_cache_key: tuple[int, int, torch.device, torch.dtype] | None = None

    @property
    def patch_size(self) -> tuple[int, int]:
        patch = getattr(getattr(self.encoder, "patch_embed", None), "patch_size", None)
        if patch is None:
            return (14, 14)
        if isinstance(patch, int):
            return (patch, patch)
        return (int(patch[0]), int(patch[1]))

    def _lowpass_mask(self, h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(h), int(w), device, dtype)
        if self._mask_cache is not None and self._mask_cache_key == key:
            return self._mask_cache

        cy, cx = h // 2, w // 2
        yy = torch.arange(h, device=device, dtype=dtype) - float(cy)
        xx = torch.arange(w, device=device, dtype=dtype) - float(cx)
        rr = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        r0 = float(self.freq.hp_radius_fraction) * float(min(h, w)) / 2.0
        mask = (rr < float(r0)).to(dtype)

        self._mask_cache = mask
        self._mask_cache_key = key
        return mask

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input (B, 3, H, W), got shape={tuple(x.shape)}")
        return (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).contiguous()

    def _extract_freq(self, x: torch.Tensor) -> torch.Tensor:
        gray = self._to_gray(x)
        f = torch.fft.fft2(gray, dim=(-2, -1))
        f = torch.fft.fftshift(f, dim=(-2, -1))

        if self.freq.mode == "logmag":
            feat = torch.log1p(torch.abs(f)).unsqueeze(1)
        elif self.freq.mode == "phase_only":
            phase = torch.angle(f)
            f_phase = torch.exp(1j * phase)
            inv = torch.fft.ifft2(torch.fft.ifftshift(f_phase, dim=(-2, -1)), dim=(-2, -1))
            feat = torch.abs(inv).unsqueeze(1)
        elif self.freq.mode == "hp_residual":
            mask_low = self._lowpass_mask(gray.shape[-2], gray.shape[-1], device=gray.device, dtype=gray.dtype)
            mask_high = 1.0 - mask_low
            f_hp = f * mask_high
            inv = torch.fft.ifft2(torch.fft.ifftshift(f_hp, dim=(-2, -1)), dim=(-2, -1))
            feat = torch.abs(inv.real).unsqueeze(1)
        elif self.freq.mode == "lp_hp":
            mask_low = self._lowpass_mask(gray.shape[-2], gray.shape[-1], device=gray.device, dtype=gray.dtype)
            mask_high = 1.0 - mask_low
            f_low = f * mask_low
            f_high = f * mask_high
            inv_low = torch.fft.ifft2(torch.fft.ifftshift(f_low, dim=(-2, -1)), dim=(-2, -1))
            inv_high = torch.fft.ifft2(torch.fft.ifftshift(f_high, dim=(-2, -1)), dim=(-2, -1))
            feat = torch.stack([torch.abs(inv_low.real), torch.abs(inv_high.real)], dim=1)
        else:
            raise ValueError(f"Unknown freq.mode: {self.freq.mode}")

        if self.freq.normalize == "zscore":
            mean = feat.mean(dim=(-2, -1), keepdim=True)
            std = feat.std(dim=(-2, -1), keepdim=True)
            feat = (feat - mean) / (std + 1e-6)
        elif self.freq.normalize != "none":
            raise ValueError(f"Unknown freq.normalize: {self.freq.normalize}")

        return feat

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

        freq_feat = self._extract_freq(x)
        freq_feat = F.interpolate(freq_feat, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        freq_feat = self.freq_encoder(freq_feat)

        fused = torch.cat([feat, freq_feat], dim=1)
        logits_patch = self.decoder(fused)
        return F.interpolate(
            logits_patch,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
