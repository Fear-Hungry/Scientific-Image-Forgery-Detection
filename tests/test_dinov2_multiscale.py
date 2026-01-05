from __future__ import annotations

import torch

from forgeryseg.models.dinov2_decoder import DinoV2EncoderSpec
from forgeryseg.models.dinov2_multiscale import DinoV2MultiScaleSegmentationModel, MultiScaleSpec


def test_dinov2_multiscale_forward_shape() -> None:
    encoder = DinoV2EncoderSpec(model_name="vit_small_patch14_reg4_dinov2", checkpoint_path=None)
    model = DinoV2MultiScaleSegmentationModel(
        encoder,
        freeze_encoder=True,
        decoder_hidden_channels=64,
        multiscale=MultiScaleSpec(layers=[2, 5, 8, 11], proj_channels=64, fuse="concat", decoder_depth=2),
    )
    x = torch.rand(1, 3, 56, 56)
    y = model(x)
    assert y.shape == (1, 1, 56, 56)

