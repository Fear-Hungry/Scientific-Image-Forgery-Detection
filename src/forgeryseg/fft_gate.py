from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_flexible_state_dict, warn_state_dict
from .config import FFTGateConfig
from .frequency import FFTParams, fft_tensor
from .models.fft_classifier import FFTClassifier
from .paths import resolve_existing_path


@dataclass(frozen=True)
class FFTGate:
    model: FFTClassifier
    params: FFTParams
    threshold: float
    device: torch.device

    @classmethod
    def from_config(
        cls,
        cfg: FFTGateConfig,
        *,
        device: torch.device,
        path_roots: list[Path] | None = None,
    ) -> FFTGate:
        if not cfg.checkpoint:
            raise ValueError("fft_gate.checkpoint is required")
        ckpt_path = resolve_existing_path(cfg.checkpoint, roots=path_roots, search_kaggle_input=True)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"FFT checkpoint não encontrado: {cfg.checkpoint} (tentado: {ckpt_path})")

        percentiles = tuple(float(x) for x in cfg.fft.normalize_percentiles)
        if len(percentiles) != 2:
            raise ValueError("fft_gate.fft.normalize_percentiles must have 2 values")

        params = FFTParams(
            mode=cfg.fft.mode,  # type: ignore[arg-type]
            input_size=int(cfg.fft.input_size),
            hp_radius_fraction=float(cfg.fft.hp_radius_fraction),
            normalize_percentiles=percentiles,  # type: ignore[arg-type]
        )
        model = FFTClassifier(
            backbone=str(cfg.backbone),
            in_chans=1,
            dropout=float(cfg.dropout),
        ).to(device)
        missing, unexpected = load_flexible_state_dict(model, ckpt_path)
        warn_state_dict(missing, unexpected)
        model.eval()
        return cls(model=model, params=params, threshold=float(cfg.threshold), device=device)

    @torch.no_grad()
    def predict_prob_forged(self, image_rgb: np.ndarray) -> float:
        x_fft = fft_tensor(image_rgb, self.params).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.model(x_fft))[0].detach().cpu().item())

    def should_override(self, image_rgb: np.ndarray) -> bool:
        return self.predict_prob_forged(image_rgb) >= float(self.threshold)
