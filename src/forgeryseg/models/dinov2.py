from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel
except ImportError:  # pragma: no cover - optional dependency
    AutoConfig = None
    AutoImageProcessor = None
    AutoModel = None


def _require_transformers() -> None:
    if AutoModel is None or AutoConfig is None:
        raise ImportError("transformers is required for DINOv2 models. Install it with `pip install transformers`.")


def _require_image_processor() -> None:
    _require_transformers()
    if AutoImageProcessor is None:
        raise ImportError("transformers AutoImageProcessor is required for DinoSeg. Install `transformers>=4.26`.")


def _as_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got bool")
    try:
        value_int = int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}") from exc
    if value_int <= 0:
        raise ValueError(f"{name} must be >= 1, got {value_int}")
    return value_int


def _as_positive_int_sequence(value: object, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of ints, got {type(value).__name__}: {value!r}")
    if isinstance(value, int):
        return (_as_positive_int(value, name=name),)
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of ints, got {type(value).__name__}: {value!r}") from exc

    out: list[int] = []
    for idx, v in enumerate(values):
        out.append(_as_positive_int(v, name=f"{name}[{idx}]"))
    if not out:
        raise ValueError(f"{name} must have at least one entry")
    return tuple(out)


def _parse_torch_dtype(value: object) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return getattr(torch, text)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype: {value!r}") from exc
    raise TypeError(f"torch_dtype must be torch.dtype or str, got {type(value).__name__}")


def _resolve_patch_size(model: nn.Module) -> int:
    patch_size = None
    config = getattr(model, "config", None)
    if config is not None:
        patch_size = getattr(config, "patch_size", None)
    if patch_size is None and hasattr(model, "patch_embed"):
        patch_size = getattr(model.patch_embed, "patch_size", None)
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]
    if patch_size is None:
        raise ValueError("Unable to resolve patch_size from DINOv2 model/config")
    return _as_positive_int(patch_size, name="patch_size")


def _resolve_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    candidates = []
    if config is not None:
        candidates.extend(
            [
                getattr(config, "hidden_size", None),
                getattr(config, "embed_dim", None),
                getattr(config, "dim", None),
            ]
        )
    candidates.append(getattr(model, "embed_dim", None))
    for val in candidates:
        if val is not None:
            return _as_positive_int(val, name="hidden_size")
    raise ValueError("Unable to resolve hidden size from DINOv2 model/config")


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = int(x.shape[-2]), int(x.shape[-1])
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h))
    return x_pad, (h + pad_h, w + pad_w)


def _split_tokens(tokens: torch.Tensor, expected_patches: int) -> tuple[torch.Tensor | None, torch.Tensor]:
    if tokens.shape[1] < expected_patches:
        raise ValueError(
            f"Token count ({tokens.shape[1]}) smaller than expected patches ({expected_patches}). "
            "Check input size and patch size."
        )
    if tokens.shape[1] == expected_patches:
        return None, tokens
    cls_token = tokens[:, 0, :]
    patch_tokens = tokens[:, -expected_patches:, :]
    return cls_token, patch_tokens


def _load_encoder(
    model_id: str,
    *,
    pretrained: bool,
    cache_dir: str | None,
    local_files_only: bool,
    revision: str | None,
    trust_remote_code: bool,
    torch_dtype: torch.dtype | None,
) -> nn.Module:
    _require_transformers()
    if pretrained:
        return AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    return AutoModel.from_config(config)


class _ConvDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        decoder_channels: Sequence[int],
        out_channels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = _as_positive_int(in_channels, name="decoder_in_channels")
        for idx, ch in enumerate(decoder_channels):
            ch = _as_positive_int(ch, name=f"decoder_channels[{idx}]")
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            prev = ch
        layers.append(nn.Conv2d(prev, _as_positive_int(out_channels, name="out_channels"), kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DinoV2Base(nn.Module):
    def __init__(
        self,
        *,
        model_id: str,
        pretrained: bool,
        freeze_encoder: bool,
        cache_dir: str | None,
        local_files_only: bool,
        revision: str | None,
        trust_remote_code: bool,
        torch_dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.freeze_encoder = bool(freeze_encoder)
        self.encoder = _load_encoder(
            self.model_id,
            pretrained=bool(pretrained),
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
            revision=revision,
            trust_remote_code=bool(trust_remote_code),
            torch_dtype=torch_dtype,
        )
        self.patch_size = _resolve_patch_size(self.encoder)
        self.hidden_size = _resolve_hidden_size(self.encoder)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int], tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError("Expected BCHW tensor")
        orig_hw = (int(x.shape[-2]), int(x.shape[-1]))
        x_pad, padded_hw = _pad_to_multiple(x, self.patch_size)
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(pixel_values=x_pad)
        else:
            outputs = self.encoder(pixel_values=x_pad)
        tokens = outputs.last_hidden_state
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor of shape (B, N, C), got {tuple(tokens.shape)}")
        grid_h = int(padded_hw[0] // self.patch_size)
        grid_w = int(padded_hw[1] // self.patch_size)
        return tokens, (grid_h, grid_w), orig_hw, padded_hw


class DinoV2Segmentation(_DinoV2Base):
    def __init__(
        self,
        *,
        model_id: str,
        decoder_channels: Sequence[int],
        decoder_dropout: float = 0.0,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        decoder_channels = _as_positive_int_sequence(decoder_channels, name="decoder_channels")
        self.decoder = _ConvDecoder(self.hidden_size, decoder_channels, out_channels=1, dropout=decoder_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, grid_hw, orig_hw, padded_hw = self._encode(x)
        expected = int(grid_hw[0] * grid_hw[1])
        _, patch_tokens = _split_tokens(tokens, expected)
        feat = patch_tokens.transpose(1, 2).reshape(x.shape[0], self.hidden_size, grid_hw[0], grid_hw[1])
        logits = self.decoder(feat)
        logits = F.interpolate(logits, size=padded_hw, mode="bilinear", align_corners=False)
        return logits[..., : orig_hw[0], : orig_hw[1]]


class DinoV2Classifier(_DinoV2Base):
    def __init__(
        self,
        *,
        model_id: str,
        num_classes: int = 1,
        classifier_hidden: int = 0,
        classifier_dropout: float = 0.0,
        use_cls_token: bool = True,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.use_cls_token = bool(use_cls_token)
        num_classes = _as_positive_int(num_classes, name="num_classes")
        classifier_hidden = int(classifier_hidden)
        if classifier_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, classifier_hidden),
                nn.GELU(),
                nn.Dropout(float(classifier_dropout)),
                nn.Linear(classifier_hidden, num_classes),
            )
        else:
            self.head = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, grid_hw, _orig_hw, _padded_hw = self._encode(x)
        expected = int(grid_hw[0] * grid_hw[1])
        cls_token, patch_tokens = _split_tokens(tokens, expected)
        if self.use_cls_token and cls_token is not None:
            pooled = cls_token
        else:
            pooled = patch_tokens.mean(dim=1)
        return self.head(pooled)


def build_dinov2_segmenter(
    *,
    model_id: str,
    decoder_channels: Sequence[int] = (256, 128, 64),
    decoder_dropout: float = 0.0,
    pretrained: bool = True,
    freeze_encoder: bool = True,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: str | torch.dtype | None = None,
) -> DinoV2Segmentation:
    return DinoV2Segmentation(
        model_id=model_id,
        decoder_channels=decoder_channels,
        decoder_dropout=decoder_dropout,
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=_parse_torch_dtype(torch_dtype),
    )


def build_dinov2_classifier(
    *,
    model_id: str,
    num_classes: int = 1,
    classifier_hidden: int = 0,
    classifier_dropout: float = 0.0,
    use_cls_token: bool = True,
    pretrained: bool = True,
    freeze_encoder: bool = True,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: str | torch.dtype | None = None,
) -> DinoV2Classifier:
    return DinoV2Classifier(
        model_id=model_id,
        num_classes=num_classes,
        classifier_hidden=classifier_hidden,
        classifier_dropout=classifier_dropout,
        use_cls_token=use_cls_token,
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=_parse_torch_dtype(torch_dtype),
    )


class DinoSeg(nn.Module):
    """
    Compatibility model for the notebook "DINO-only" path.

    It wraps a frozen DINOv2 encoder loaded from `transformers` and trains only a lightweight conv head.
    The forward expects inputs in [0, 1] (B, C, H, W) and internally uses the HF image processor
    (no resize, no center crop) to match the encoder's preprocessing.
    """

    def __init__(
        self,
        dino_path: str,
        out_ch: int = 1,
        *,
        decoder_dropout: float = 0.0,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _require_image_processor()
        self.dino_path = str(dino_path)
        self.freeze_encoder = bool(freeze_encoder)

        self.processor = AutoImageProcessor.from_pretrained(  # type: ignore[misc]
            self.dino_path,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
            revision=revision,
            trust_remote_code=bool(trust_remote_code),
        )
        if pretrained:
            self.encoder = AutoModel.from_pretrained(  # type: ignore[misc]
                self.dino_path,
                cache_dir=cache_dir,
                local_files_only=bool(local_files_only),
                revision=revision,
                trust_remote_code=bool(trust_remote_code),
                torch_dtype=torch_dtype,
            )
        else:
            config = AutoConfig.from_pretrained(  # type: ignore[misc]
                self.dino_path,
                cache_dir=cache_dir,
                local_files_only=bool(local_files_only),
                revision=revision,
                trust_remote_code=bool(trust_remote_code),
            )
            self.encoder = AutoModel.from_config(config)  # type: ignore[misc]

        if self.freeze_encoder:
            self.encoder.requires_grad_(False)

        self.patch_size = _resolve_patch_size(self.encoder)
        self.hidden_size = _resolve_hidden_size(self.encoder)

        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_size, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=float(decoder_dropout)),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, int(out_ch), 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError("DinoSeg expects a floating point tensor in [0, 1]")

        imgs = (x * 255.0).clamp(0, 255).to(torch.uint8)
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(  # type: ignore[misc]
            images=list(imgs),
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).to(x.device)

        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(**inputs)
        else:
            outputs = self.encoder(**inputs)

        tokens = outputs.last_hidden_state
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor (B, N, C), got shape {tuple(tokens.shape)}")

        h, w = int(inputs["pixel_values"].shape[-2]), int(inputs["pixel_values"].shape[-1])
        grid_h = h // int(self.patch_size)
        grid_w = w // int(self.patch_size)
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"Input too small for patch_size={self.patch_size}: got {h}x{w}")
        expected = int(grid_h * grid_w)
        _, patch_tokens = _split_tokens(tokens, expected)
        return patch_tokens.transpose(1, 2).reshape(x.shape[0], self.hidden_size, grid_h, grid_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.forward_features(x)
        fmap = F.interpolate(fmap, size=(int(x.shape[2]), int(x.shape[3])), mode="bilinear", align_corners=False)
        return self.head(fmap)
