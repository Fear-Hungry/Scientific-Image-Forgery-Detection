from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class SegmentationEnsemble(nn.Module):
    """Simple ensembling wrapper for segmentation models.

    By default, returns averaged logits (good for training with BCE/Dice losses).
    For inference, you may prefer averaging probabilities (sigmoid outputs).
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        weights: Sequence[float] | None = None,
        output: str = "logits",
    ) -> None:
        super().__init__()
        if len(models) == 0:
            raise ValueError("models must be non-empty")
        if output not in {"logits", "probs"}:
            raise ValueError("output must be 'logits' or 'probs'")

        self.models = nn.ModuleList(models)
        self.output = output

        if weights is None:
            weights = [1.0] * len(models)
        if len(weights) != len(models):
            raise ValueError("weights must have the same length as models")
        w = torch.tensor(list(weights), dtype=torch.float32)
        if float(w.sum()) <= 0.0:
            raise ValueError("weights must sum to > 0")
        self.register_buffer("_weights", w / w.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = []
        for model in self.models:
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds.append(out)

        if self.output == "logits":
            out = preds[0] * self._weights[0]
            for w, p in zip(self._weights[1:], preds[1:]):
                out = out + p * w
            return out

        out = torch.sigmoid(preds[0]) * self._weights[0]
        for w, p in zip(self._weights[1:], preds[1:]):
            out = out + torch.sigmoid(p) * w
        return out

