from __future__ import annotations

from typing import Any

from torch import nn

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None


def build_classifier(model_name: str = "tf_efficientnet_b4_ns", pretrained: bool = False, num_classes: int = 1, **kwargs: Any) -> nn.Module:
    """
    Build a simple image classifier.

    - Prefer `timm` when available.
    - Fallback to torchvision ResNet-50.

    Notes (Kaggle offline):
    - Use `pretrained=False` to avoid downloading weights when internet is OFF.
    """
    if timm is not None:
        return timm.create_model(model_name, pretrained=bool(pretrained), num_classes=int(num_classes), **kwargs)

    try:
        from torchvision.models import resnet50

        try:
            model = resnet50(weights=None)
        except TypeError:
            model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, int(num_classes))
        return model
    except Exception as exc:  # pragma: no cover
        raise ImportError("Neither timm nor torchvision is available to build a classifier") from exc


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

