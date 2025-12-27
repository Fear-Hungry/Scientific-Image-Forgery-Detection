from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover - optional dependency
    smp = None


def _as_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got bool")
    try:
        value_int = int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}") from exc
    if value_int <= 0:
        raise ValueError(f"{name} must be >= 1, got {value_int}")
    return value_int


def _as_positive_int_sequence(value: Any, *, name: str, expected_len: int | None = None) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of ints, got {type(value).__name__}: {value!r}")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of ints, got {type(value).__name__}: {value!r}") from exc

    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(values)}: {values!r}")

    out: list[int] = []
    for idx, v in enumerate(values):
        out.append(_as_positive_int(v, name=f"{name}[{idx}]"))
    return tuple(out)


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
    encoder_depth = _as_positive_int(encoder_depth, name="encoder_depth")
    in_channels = _as_positive_int(in_channels, name="in_channels")
    classes = _as_positive_int(classes, name="classes")
    decoder_channels = _as_positive_int_sequence(decoder_channels, name="decoder_channels", expected_len=encoder_depth)
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
    encoder_depth = _as_positive_int(encoder_depth, name="encoder_depth")
    decoder_segmentation_channels = _as_positive_int(decoder_segmentation_channels, name="decoder_segmentation_channels")
    in_channels = _as_positive_int(in_channels, name="in_channels")
    classes = _as_positive_int(classes, name="classes")
    upsampling = _as_positive_int(upsampling, name="upsampling")
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


def build_unetplusplus(
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
        raise ImportError("segmentation_models_pytorch is required for Unet++ models")
    encoder_depth = _as_positive_int(encoder_depth, name="encoder_depth")
    in_channels = _as_positive_int(in_channels, name="in_channels")
    classes = _as_positive_int(classes, name="classes")
    decoder_channels = _as_positive_int_sequence(decoder_channels, name="decoder_channels", expected_len=encoder_depth)
    return _safe_init(
        smp.UnetPlusPlus,
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


def build_deeplabv3plus(
    encoder_name: str = "resnet101",
    encoder_weights: str | None = "imagenet",
    encoder_depth: int = 5,
    decoder_channels: int = 256,
    atrous_rates: Sequence[int] = (12, 24, 36),
    in_channels: int = 3,
    classes: int = 1,
    strict_weights: bool = False,
    **kwargs: Any,
):
    if smp is None:
        raise ImportError("segmentation_models_pytorch is required for DeepLabV3+ models")
    encoder_depth = _as_positive_int(encoder_depth, name="encoder_depth")
    decoder_channels = _as_positive_int(decoder_channels, name="decoder_channels")
    in_channels = _as_positive_int(in_channels, name="in_channels")
    classes = _as_positive_int(classes, name="classes")
    atrous_rates = _as_positive_int_sequence(atrous_rates, name="atrous_rates")
    return _safe_init(
        smp.DeepLabV3Plus,
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        atrous_rates=atrous_rates,
        in_channels=in_channels,
        classes=classes,
        activation=None,
        strict_weights=strict_weights,
        encoder_weights=encoder_weights,
        **kwargs,
    )
