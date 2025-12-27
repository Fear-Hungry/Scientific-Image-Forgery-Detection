from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SelfCorrelationBlock(nn.Module):
    """Non-local self-correlation over spatial positions (with optional downsample).

    Designed for copy-move: highlights repeated patterns by comparing embeddings across the image.
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: Optional[int] = None,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if embed_channels is None:
            embed_channels = max(8, in_channels // 8)
        if embed_channels <= 0:
            raise ValueError("embed_channels must be > 0")
        self.in_channels = int(in_channels)
        self.embed_channels = int(embed_channels)
        self.max_tokens = int(max_tokens)

        self.qkv = nn.Conv2d(self.in_channels, self.embed_channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(self.embed_channels, self.in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _maybe_downsample(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        b, c, h, w = x.shape
        if self.max_tokens <= 0:
            return x, (h, w)
        tokens = h * w
        if tokens <= self.max_tokens:
            return x, (h, w)

        scale = math.sqrt(tokens / float(self.max_tokens))
        new_h = max(1, int(round(h / scale)))
        new_w = max(1, int(round(w / scale)))
        if new_h == h and new_w == w:
            return x, (h, w)
        x_small = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x_small, (h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected BCHW tensor")

        x_small, orig_hw = self._maybe_downsample(x)
        b, _, h, w = x_small.shape
        n = h * w

        qkv = self.qkv(x_small)
        q, k, v = qkv.chunk(3, dim=1)  # (B, D, H, W)

        d = q.shape[1]
        q = q.view(b, d, n).transpose(1, 2)  # (B, N, D)
        k = k.view(b, d, n)  # (B, D, N)
        v = v.view(b, d, n).transpose(1, 2)  # (B, N, D)

        attn = torch.bmm(q, k) * (1.0 / math.sqrt(float(d)))  # (B, N, N)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v)  # (B, N, D)
        out = out.transpose(1, 2).contiguous().view(b, d, h, w)
        out = self.proj(out)

        if (h, w) != orig_hw:
            out = F.interpolate(out, size=orig_hw, mode="bilinear", align_corners=False)

        return x + self.gamma * out


@dataclass(frozen=True)
class CorrelationConfig:
    feature_index: int = -1
    embed_channels: Optional[int] = None
    max_tokens: int = 256


class SmpCorrelationWrapper(nn.Module):
    """Wrap an SMP model and inject self-correlation into one encoder feature map."""

    def __init__(self, model: nn.Module, config: CorrelationConfig = CorrelationConfig()) -> None:
        super().__init__()
        self.model = model
        self.config = config

        if not hasattr(model, "encoder") or not hasattr(model, "decoder") or not hasattr(model, "segmentation_head"):
            raise TypeError("Expected a segmentation_models_pytorch model with encoder/decoder/segmentation_head")

        self.corr: Optional[SelfCorrelationBlock] = None

    def _get_corr(self, feature: torch.Tensor) -> SelfCorrelationBlock:
        if self.corr is None:
            self.corr = SelfCorrelationBlock(
                in_channels=int(feature.shape[1]),
                embed_channels=self.config.embed_channels,
                max_tokens=self.config.max_tokens,
            )
        return self.corr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "check_input_shape"):
            self.model.check_input_shape(x)

        features = list(self.model.encoder(x))
        idx = self.config.feature_index
        if idx < 0:
            idx = len(features) + idx
        if idx < 0 or idx >= len(features):
            raise IndexError(f"feature_index out of range: {self.config.feature_index}")

        feature = features[idx]
        if feature.numel() > 0:
            features[idx] = self._get_corr(feature)(feature)

        decoder_output = self.model.decoder(features)
        masks = self.model.segmentation_head(decoder_output)
        return masks

