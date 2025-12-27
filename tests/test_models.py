import pytest
import torch

from forgeryseg.models.correlation import CorrelationConfig, SelfCorrelationBlock, SmpCorrelationWrapper
from forgeryseg.models.ensemble import SegmentationEnsemble


def test_self_correlation_block_preserves_shape_and_identity_at_init():
    x = torch.randn(2, 32, 16, 16)
    block = SelfCorrelationBlock(in_channels=32, embed_channels=8, max_tokens=256)
    y = block(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x)


def test_smp_wrapper_forward_shape():
    smp = pytest.importorskip("segmentation_models_pytorch")
    base = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    wrapped = SmpCorrelationWrapper(base, CorrelationConfig(feature_index=-1, max_tokens=256))

    x = torch.randn(2, 3, 64, 64)
    y = wrapped(x)
    assert tuple(y.shape) == (2, 1, 64, 64)


def test_segmentation_ensemble_logits_and_probs_shapes():
    smp = pytest.importorskip("segmentation_models_pytorch")
    m1 = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1, activation=None)
    m2 = smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=1, activation=None)

    x = torch.randn(1, 3, 64, 64)
    ens_logits = SegmentationEnsemble([m1, m2], output="logits")
    ens_probs = SegmentationEnsemble([m1, m2], output="probs")

    y_logits = ens_logits(x)
    y_probs = ens_probs(x)

    assert tuple(y_logits.shape) == (1, 1, 64, 64)
    assert tuple(y_probs.shape) == (1, 1, 64, 64)
