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

