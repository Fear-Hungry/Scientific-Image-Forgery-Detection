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


def test_build_deeplabv3plus_uses_decoder_atrous_rates_kwarg(monkeypatch):
    from forgeryseg.models import builders

    calls = []

    def deeplab_builder(*, encoder_weights=None, **kwargs):
        calls.append((encoder_weights, dict(kwargs)))
        return {"encoder_weights": encoder_weights, **kwargs}

    monkeypatch.setattr(builders, "smp", type("SMP", (), {"DeepLabV3Plus": deeplab_builder}))

    out = builders.build_deeplabv3plus(
        encoder_name="tu-resnest101e",
        encoder_weights=None,
        atrous_rates=(6, 12, 18),
        in_channels=3,
        classes=1,
    )
    assert out["decoder_atrous_rates"] == (6, 12, 18)
    assert "atrous_rates" not in out
    assert calls and "decoder_atrous_rates" in calls[0][1]


def test_build_deeplabv3plus_does_not_fallback_on_bad_kwarg(monkeypatch):
    from forgeryseg.models import builders

    def deeplab_builder(*, encoder_weights=None, **kwargs):
        raise TypeError("got an unexpected keyword argument 'decoder_atrous_rates'")

    monkeypatch.setattr(builders, "smp", type("SMP", (), {"DeepLabV3Plus": deeplab_builder}))

    with pytest.raises(TypeError, match=r"decoder_atrous_rates"):
        builders.build_deeplabv3plus(
            encoder_name="tu-resnest101e",
            encoder_weights=None,
            atrous_rates=(6, 12, 18),
            in_channels=3,
            classes=1,
        )
