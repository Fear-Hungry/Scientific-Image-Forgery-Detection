from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
except ImportError:  # pragma: no cover - optional dependency
    smp = None
    UnetDecoder = None

from .dinov2 import _DinoV2Base, _parse_torch_dtype, _split_tokens


class HybridDinoUNet(nn.Module):
    """
    Hybrid architecture:
    - CNN encoder provides multi-scale local features (skip connections).
    - DINOv2 provides global semantic tokens.
    - Tokens are reshaped and concatenated to the deepest CNN feature map.
    - U-Net decoder reconstructs the segmentation mask.
    """

    def __init__(
        self,
        *,
        cnn_encoder_name: str,
        cnn_encoder_weights: str | None,
        dino_model_id: str,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 1,
        attention_type: str | None = "scse",
        freeze_dino: bool = True,
        dino_pretrained: bool = True,
        dino_cache_dir: str | None = None,
        dino_local_files_only: bool = False,
        dino_revision: str | None = None,
        dino_trust_remote_code: bool = False,
        dino_torch_dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if smp is None or UnetDecoder is None:
            raise ImportError("segmentation_models_pytorch is required for HybridDinoUNet")

        in_channels = int(in_channels)
        classes = int(classes)
        decoder_channels = tuple(int(c) for c in decoder_channels)
        if not decoder_channels:
            raise ValueError("decoder_channels must not be empty")

        self.cnn_encoder = smp.encoders.get_encoder(
            cnn_encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=cnn_encoder_weights,
        )

        self.dino = _DinoV2Base(
            model_id=dino_model_id,
            pretrained=bool(dino_pretrained),
            freeze_encoder=bool(freeze_dino),
            cache_dir=dino_cache_dir,
            local_files_only=bool(dino_local_files_only),
            revision=dino_revision,
            trust_remote_code=bool(dino_trust_remote_code),
            torch_dtype=_parse_torch_dtype(dino_torch_dtype),
        )

        cnn_channels = list(self.cnn_encoder.out_channels)
        dino_dim = int(self.dino.hidden_size)

        encoder_channels = list(cnn_channels)
        encoder_channels[-1] = int(encoder_channels[-1]) + dino_dim

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=True,
            center=False if str(cnn_encoder_name).startswith("vgg") else True,
            attention_type=attention_type,
        )

        self.segmentation_head = nn.Conv2d(int(decoder_channels[-1]), classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_hw = (int(x.shape[-2]), int(x.shape[-1]))

        cnn_features = self.cnn_encoder(x)

        tokens, grid_hw, _orig_hw, _padded_hw = self.dino._encode(x)
        expected = int(grid_hw[0] * grid_hw[1])
        _, patch_tokens = _split_tokens(tokens, expected)

        dino_dim = int(patch_tokens.shape[-1])
        dino_map = patch_tokens.transpose(1, 2).reshape(x.shape[0], dino_dim, grid_hw[0], grid_hw[1])

        c5 = cnn_features[-1]
        if dino_map.shape[-2:] != c5.shape[-2:]:
            dino_map = F.interpolate(dino_map, size=c5.shape[-2:], mode="bilinear", align_corners=False)

        cnn_features[-1] = torch.cat([c5, dino_map], dim=1)

        decoded = self.decoder(*cnn_features)
        logits = self.segmentation_head(decoded)

        if logits.shape[-2:] != orig_hw:
            logits = F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)

        return logits


def build_hybrid_model(
    *,
    encoder_name: str,
    encoder_weights: str | None,
    in_channels: int,
    classes: int = 1,
    dino_model_id: str = "facebook/dinov2-base",
    decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    freeze_dino: bool = True,
    dino_pretrained: bool = True,
    dino_cache_dir: str | None = None,
    dino_local_files_only: bool = False,
    dino_revision: str | None = None,
    dino_trust_remote_code: bool = False,
    dino_torch_dtype: str | torch.dtype | None = None,
    attention_type: str | None = "scse",
) -> HybridDinoUNet:
    return HybridDinoUNet(
        cnn_encoder_name=str(encoder_name),
        cnn_encoder_weights=encoder_weights,
        dino_model_id=str(dino_model_id),
        decoder_channels=decoder_channels,
        in_channels=int(in_channels),
        classes=int(classes),
        attention_type=attention_type,
        freeze_dino=bool(freeze_dino),
        dino_pretrained=bool(dino_pretrained),
        dino_cache_dir=dino_cache_dir,
        dino_local_files_only=bool(dino_local_files_only),
        dino_revision=dino_revision,
        dino_trust_remote_code=bool(dino_trust_remote_code),
        dino_torch_dtype=dino_torch_dtype,
    )
