from __future__ import annotations

from typing import Any

from torch import nn

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None


class _TimmEncoderClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool,
        num_classes: int,
        feature_index: int = -1,
        pool: str = "avg",
        classifier_hidden: int = 0,
        classifier_dropout: float = 0.0,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for backend='timm_encoder'")
        self.backbone = timm.create_model(model_name, pretrained=bool(pretrained), features_only=True)
        channels = list(self.backbone.feature_info.channels())
        if not channels:
            raise ValueError("timm features_only returned no channels")
        if feature_index < 0:
            feature_index = len(channels) + int(feature_index)
        if feature_index < 0 or feature_index >= len(channels):
            raise ValueError(f"feature_index out of range: {feature_index}")
        self.feature_index = int(feature_index)
        self.pool = str(pool).lower()
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            self.backbone.requires_grad_(False)
        in_ch = int(channels[self.feature_index])
        hidden = int(classifier_hidden)
        if hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(in_ch, hidden),
                nn.GELU(),
                nn.Dropout(float(classifier_dropout)),
                nn.Linear(hidden, int(num_classes)),
            )
        else:
            self.head = nn.Linear(in_ch, int(num_classes))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.backbone.eval()
        return self

    def _pool(self, feat):
        import torch.nn.functional as F

        if self.pool == "max":
            return F.adaptive_max_pool2d(feat, 1)
        return F.adaptive_avg_pool2d(feat, 1)

    def forward(self, x):
        feats = self.backbone(x)
        feat = feats[self.feature_index]
        pooled = self._pool(feat).flatten(1)
        return self.head(pooled)


def build_classifier(
    model_name: str = "tf_efficientnet_b4_ns",
    pretrained: bool = False,
    num_classes: int = 1,
    backend: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Build a simple image classifier.

    - Prefer `timm` when available.
    - Fallback to torchvision ResNet-50.

    Notes (Kaggle offline):
    - Use `pretrained=False` to avoid downloading weights when internet is OFF.
    """
    backend = (backend or "timm").lower()
    if backend in {"dinov2", "hf"}:
        from .dinov2 import build_dinov2_classifier

        model_id = kwargs.pop("hf_model_id", None) or kwargs.pop("model_id", None) or model_name
        cache_dir = kwargs.pop("hf_cache_dir", None) or kwargs.pop("cache_dir", None)
        revision = kwargs.pop("hf_revision", None) or kwargs.pop("revision", None)
        return build_dinov2_classifier(
            model_id=str(model_id),
            num_classes=int(num_classes),
            classifier_hidden=int(kwargs.pop("classifier_hidden", 0)),
            classifier_dropout=float(kwargs.pop("classifier_dropout", 0.0)),
            use_cls_token=bool(kwargs.pop("use_cls_token", True)),
            pretrained=bool(pretrained),
            freeze_encoder=bool(kwargs.pop("freeze_encoder", True)),
            cache_dir=cache_dir,
            local_files_only=bool(kwargs.pop("local_files_only", False)),
            revision=str(revision) if revision else None,
            trust_remote_code=bool(kwargs.pop("trust_remote_code", False)),
            torch_dtype=kwargs.pop("torch_dtype", None),
        )

    if backend == "timm_encoder":
        return _TimmEncoderClassifier(
            model_name=model_name,
            pretrained=bool(pretrained),
            num_classes=int(num_classes),
            feature_index=int(kwargs.pop("feature_index", -1)),
            pool=str(kwargs.pop("pool", "avg")),
            classifier_hidden=int(kwargs.pop("classifier_hidden", 0)),
            classifier_dropout=float(kwargs.pop("classifier_dropout", 0.0)),
            freeze_encoder=bool(kwargs.pop("freeze_encoder", False)),
        )

    if backend == "timm":
        if timm is None:
            raise ImportError("timm is required for backend='timm'")
        return timm.create_model(model_name, pretrained=bool(pretrained), num_classes=int(num_classes), **kwargs)

    if backend == "torchvision":
        try:
            from torchvision.models import resnet50

            try:
                model = resnet50(weights=None)
            except TypeError:
                model = resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, int(num_classes))
            return model
        except Exception as exc:  # pragma: no cover
            raise ImportError("torchvision is required for backend='torchvision'") from exc

    raise ValueError(f"Unknown classifier backend: {backend!r}")


def compute_pos_weight(labels) -> float:
    """
    Compute a BCEWithLogits `pos_weight` value (neg/pos) from a 1D list/array of 0/1 labels.
    Returns 1.0 when there are no positive labels.
    """
    import numpy as np

    y = np.asarray(labels).astype(np.int64).reshape(-1)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg / max(pos, 1.0))
