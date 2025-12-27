from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover - optional dependency
    smp = None


def available_encoders() -> list[str]:
    if smp is None:
        return []
    return list(smp.encoders.get_encoder_names())


def _safe_init(builder, *, encoder_weights: Optional[str], strict_weights: bool, **kwargs):
    if smp is None:
        raise ImportError("segmentation_models_pytorch is required for model builders")
    if encoder_weights is None or strict_weights:
        return builder(encoder_weights=encoder_weights, **kwargs)

    try:
        return builder(encoder_weights=encoder_weights, **kwargs)
    except Exception as exc:
        try:
            return builder(encoder_weights=None, **kwargs)
        except Exception:
            raise exc


def build_unet(
    encoder_name: str = "efficientnet-b7",
    encoder_weights: str | None = "imagenet",
    encoder_depth: int = 5,
    decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    decoder_attention_type: str | None = "scse",
    in_channels: int = 3,
    classes: int = 1,
    strict_weights: bool = False,
    **kwargs: Any,
):
    if smp is None:
        raise ImportError("segmentation_models_pytorch is required for Unet models")
    return _safe_init(
        smp.Unet,
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        decoder_attention_type=decoder_attention_type,
        in_channels=in_channels,
        classes=classes,
        activation=None,
        strict_weights=strict_weights,
        encoder_weights=encoder_weights,
        **kwargs,
    )


def build_segformer(
    encoder_name: str = "mit_b2",
    encoder_weights: str | None = "imagenet",
    encoder_depth: int = 5,
    decoder_segmentation_channels: int = 256,
    in_channels: int = 3,
    classes: int = 1,
    upsampling: int = 4,
    strict_weights: bool = False,
    **kwargs: Any,
):
    if smp is None:
        raise ImportError("segmentation_models_pytorch is required for SegFormer models")
    return _safe_init(
        smp.Segformer,
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        decoder_segmentation_channels=decoder_segmentation_channels,
        in_channels=in_channels,
        classes=classes,
        activation=None,
        upsampling=upsampling,
        strict_weights=strict_weights,
        encoder_weights=encoder_weights,
        **kwargs,
    )

