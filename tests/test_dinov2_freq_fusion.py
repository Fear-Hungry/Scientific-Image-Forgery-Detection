from __future__ import annotations

import torch

from forgeryseg.models.dinov2_decoder import DinoV2EncoderSpec
from forgeryseg.models.dinov2_freq_fusion import DinoV2FreqFusionSegmentationModel, FreqFusionSpec


def test_dino_freq_fusion_forward_shape_logmag() -> None:
    encoder = DinoV2EncoderSpec(model_name="vit_small_patch14_dinov2", checkpoint_path=None)
    model = DinoV2FreqFusionSegmentationModel(
        encoder,
        freeze_encoder=True,
        freq=FreqFusionSpec(mode="logmag", freq_channels=8),
    )
    x = torch.rand(2, 3, 56, 56)
    y = model(x)
    assert y.shape == (2, 1, 56, 56)


def test_dino_freq_fusion_forward_shape_low_high() -> None:
    encoder = DinoV2EncoderSpec(model_name="vit_small_patch14_dinov2", checkpoint_path=None)
    model = DinoV2FreqFusionSegmentationModel(
        encoder,
        freeze_encoder=True,
        freq=FreqFusionSpec(mode="lp_hp", freq_channels=8, hp_radius_fraction=0.2),
    )
    x = torch.rand(1, 3, 56, 56)
    y = model(x)
    assert y.shape == (1, 1, 56, 56)


def test_dino_freq_fusion_forward_shape_reg_tokens() -> None:
    encoder = DinoV2EncoderSpec(model_name="vit_small_patch14_reg4_dinov2", checkpoint_path=None)
    model = DinoV2FreqFusionSegmentationModel(
        encoder,
        freeze_encoder=True,
        freq=FreqFusionSpec(mode="logmag", freq_channels=8),
    )
    x = torch.rand(1, 3, 56, 56)
    y = model(x)
    assert y.shape == (1, 1, 56, 56)
