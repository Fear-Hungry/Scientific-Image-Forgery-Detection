import pytest


def test_build_unetplusplus_rejects_zero_decoder_channels():
    pytest.importorskip("segmentation_models_pytorch")
    from forgeryseg.models import builders

    with pytest.raises(ValueError, match=r"decoder_channels"):
        builders.build_unetplusplus(
            encoder_name="resnet34",
            encoder_weights=None,
            decoder_channels=(64, 32, 16, 8, 0),
        )


def test_build_unetplusplus_tu_swin_tiny_runs_forward():
    pytest.importorskip("segmentation_models_pytorch")
    pytest.importorskip("timm")
    import torch

    from forgeryseg.models import builders

    model = builders.build_unetplusplus(
        encoder_name="tu-swin_tiny_patch4_window7_224",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    empty = [name for name, p in model.named_parameters() if p.numel() == 0]
    assert not empty

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 128, 128)
