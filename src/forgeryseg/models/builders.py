from __future__ import annotations

from typing import Any, Optional, Sequence

from torch import nn

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover - optional dependency
    smp = None

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None


def _is_transformer_style_reduction(reductions: Sequence[int]) -> bool:
    if not reductions:
        return False
    # Transformer-style downsampling: (4, 8, 16, 32, ...)
    expected = [2 ** (i + 2) for i in range(len(reductions))]
    return list(reductions) == expected


class _TimmEncoder(nn.Module):
    """
    timm `features_only` encoder compatible with SMP decoders.

    Some timm backbones (e.g. Swin/ConvNeXt) start feature extraction at stride=4.
    SMP decoders assume a stride=2 stage exists; to keep output resolution consistent
    we synthesize a stride=2 feature by upsampling the first stride=4 feature map.
    """

    def __init__(self, model_name: str, *, in_channels: int, depth: int, pretrained: bool) -> None:
        super().__init__()
        if timm is None:  # pragma: no cover - defensive
            raise ImportError("timm is required for `tu-` encoders")
        self.model_name = str(model_name)
        self.in_channels = int(in_channels)
        self.depth = int(depth)
        self.pretrained = bool(pretrained)

        backbone_kwargs = dict(
            pretrained=self.pretrained,
            features_only=True,
            in_chans=self.in_channels,
        )
        # Many transformer backbones (e.g. Swin) default to strict 224x224 inputs.
        # Disable strictness when supported; fallback for models that don't accept it.
        try:
            self.backbone = timm.create_model(self.model_name, strict_img_size=False, **backbone_kwargs)
        except TypeError:
            self.backbone = timm.create_model(self.model_name, **backbone_kwargs)

        out_fmt = getattr(self.backbone, "output_fmt", None)
        out_fmt_s = str(out_fmt).upper() if out_fmt is not None else ""
        self._is_channel_last = "NHWC" in out_fmt_s

        channels = list(self.backbone.feature_info.channels())
        reductions = list(self.backbone.feature_info.reduction())
        self._is_transformer_style = _is_transformer_style_reduction(reductions)

        if self._is_transformer_style and self.depth < 2:
            raise ValueError(f"encoder_depth must be >= 2 for transformer-style encoders, got {self.depth}")

        features_needed = (self.depth - 1) if self._is_transformer_style else self.depth
        if features_needed <= 0:
            raise ValueError(f"encoder_depth must be >= 1, got {self.depth}")
        if features_needed > len(channels):
            raise ValueError(
                f"Requested encoder_depth={self.depth} but timm model {self.model_name!r} provides only "
                f"{len(channels)} feature stages (reduction={reductions})."
            )

        self._features_needed = int(features_needed)
        out_channels: list[int] = [self.in_channels]
        if self._is_transformer_style:
            out_channels.append(int(channels[0]))
        out_channels.extend(int(c) for c in channels[: self._features_needed])
        self._out_channels = out_channels

    @property
    def out_channels(self) -> list[int]:
        return list(self._out_channels)

    def forward(self, x):
        import torch.nn.functional as F

        feats = list(self.backbone(x))[: self._features_needed]
        if self._is_channel_last:
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        out = [x]
        if self._is_transformer_style:
            if not feats:
                raise RuntimeError("timm returned no features; cannot synthesize stride-2 stage")
            target_h = max(int(x.shape[-2]) // 2, 1)
            target_w = max(int(x.shape[-1]) // 2, 1)
            stride2 = F.interpolate(feats[0], size=(target_h, target_w), mode="bilinear", align_corners=False)
            out.append(stride2)
        out.extend(feats)
        return out


class _TimmUnet(nn.Module):
    def __init__(
        self,
        *,
        encoder_name: str,
        encoder_weights: str | None,
        encoder_depth: int,
        decoder_channels: Sequence[int],
        decoder_attention_type: str | None,
        in_channels: int,
        classes: int,
        decoder_use_norm: bool | str | dict[str, Any] = "batchnorm",
        decoder_interpolation: str = "nearest",
    ) -> None:
        super().__init__()
        if smp is None:  # pragma: no cover - defensive
            raise ImportError("segmentation_models_pytorch is required for Unet models")

        encoder = _TimmEncoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            pretrained=encoder_weights is not None,
        )
        self.encoder = encoder
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=int(encoder_depth),
            use_norm=decoder_use_norm,
            attention_type=decoder_attention_type,
            add_center_block=False,
            interpolation_mode=decoder_interpolation,
        )
        self.segmentation_head = smp.base.heads.SegmentationHead(
            in_channels=int(decoder_channels[-1]),
            out_channels=int(classes),
            activation=None,
            kernel_size=3,
        )
        smp.base.initialization.initialize_decoder(self.decoder)
        smp.base.initialization.initialize_head(self.segmentation_head)

    def forward(self, x):
        feats = self.encoder(x)
        dec = self.decoder(feats)
        return self.segmentation_head(dec)


class _TimmUnetPlusPlus(nn.Module):
    def __init__(
        self,
        *,
        encoder_name: str,
        encoder_weights: str | None,
        encoder_depth: int,
        decoder_channels: Sequence[int],
        decoder_attention_type: str | None,
        in_channels: int,
        classes: int,
        decoder_use_norm: bool | str | dict[str, Any] = "batchnorm",
        decoder_interpolation: str = "nearest",
    ) -> None:
        super().__init__()
        if smp is None:  # pragma: no cover - defensive
            raise ImportError("segmentation_models_pytorch is required for Unet++ models")

        encoder = _TimmEncoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            pretrained=encoder_weights is not None,
        )
        self.encoder = encoder
        self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
            encoder_channels=encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=int(encoder_depth),
            use_norm=decoder_use_norm,
            center=False,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )
        self.segmentation_head = smp.base.heads.SegmentationHead(
            in_channels=int(decoder_channels[-1]),
            out_channels=int(classes),
            activation=None,
            kernel_size=3,
        )
        smp.base.initialization.initialize_decoder(self.decoder)
        smp.base.initialization.initialize_head(self.segmentation_head)

    def forward(self, x):
        feats = self.encoder(x)
        dec = self.decoder(feats)
        return self.segmentation_head(dec)


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
    return builder(encoder_weights=encoder_weights, **kwargs)


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
    if str(encoder_name).startswith("tu-"):
        return _TimmUnet(
            encoder_name=str(encoder_name)[3:],
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
        )
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
    if str(encoder_name).startswith("tu-"):
        return _TimmUnetPlusPlus(
            encoder_name=str(encoder_name)[3:],
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
        )
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
    # Avoid leaking atrous rate kwargs into SMP's `get_encoder(**kwargs)` call.
    # Be strict: callers must use the `atrous_rates=` argument of this function.
    if "decoder_atrous_rates" in kwargs:
        raise TypeError("build_deeplabv3plus() got an unexpected keyword argument 'decoder_atrous_rates' (use atrous_rates=...)")
    atrous_rates = _as_positive_int_sequence(atrous_rates, name="atrous_rates")
    common_kwargs = dict(
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        in_channels=in_channels,
        classes=classes,
        activation=None,
        **kwargs,
    )
    return _safe_init(
        smp.DeepLabV3Plus,
        decoder_atrous_rates=atrous_rates,
        strict_weights=strict_weights,
        encoder_weights=encoder_weights,
        **common_kwargs,
    )
