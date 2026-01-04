from __future__ import annotations

import timm
import torch
import torch.nn as nn


class FFTClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        in_chans: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=False,
            in_chans=int(in_chans),
            num_classes=1,
            drop_rate=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if out.ndim == 2 and out.shape[1] == 1:
            out = out[:, 0]
        return out

