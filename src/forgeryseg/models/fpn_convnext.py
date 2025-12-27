from __future__ import annotations

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover - optional dependency
    smp = None


def build_model(
    encoder_name: str = "convnext_tiny",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
):
    if smp is None:
        raise ImportError("segmentation_models_pytorch is required for FPN models")
    return smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
