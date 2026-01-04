from __future__ import annotations

import torch

from forgeryseg.models.dinov2_decoder import DinoV2EncoderSpec, DinoV2SegmentationModel


def test_dinov2_decoder_forward_shape_reg_tokens() -> None:
    encoder = DinoV2EncoderSpec(model_name="vit_small_patch14_reg4_dinov2", checkpoint_path=None)
    model = DinoV2SegmentationModel(
        encoder,
        freeze_encoder=True,
        decoder_hidden_channels=64,
    )
    x = torch.rand(1, 3, 56, 56)
    y = model(x)
    assert y.shape == (1, 1, 56, 56)
