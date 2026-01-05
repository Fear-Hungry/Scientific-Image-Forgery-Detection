from __future__ import annotations

import torch

from forgeryseg.inference import build_tta
from forgeryseg.tta import HFlipTTA, IdentityTTA, Rot90TTA, VFlipTTA, ZoomInTTA, ZoomOutTTA


def test_tta_invert_roundtrip_for_flips_and_rot() -> None:
    x = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    for tta in [IdentityTTA(), HFlipTTA(), VFlipTTA(), Rot90TTA(k=1), Rot90TTA(k=2), Rot90TTA(k=3)]:
        xr = tta.invert(tta.apply(x))
        assert torch.equal(xr, x)


def test_build_tta_custom_modes_and_weight_fallback() -> None:
    tta, w = build_tta(modes=["identity", "hflip", "vflip", "rot90", "zoom_in"], zoom_in_scale=1.2, weights=[1.0])
    assert len(tta) == 5
    assert len(w) == 5  # fallback to equal weights


def test_zoom_tta_shapes() -> None:
    x = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    for tta in [ZoomOutTTA(scale=0.9), ZoomInTTA(scale=1.1)]:
        y = tta.apply(x)
        assert y.shape == x.shape
        z = tta.invert(y)
        assert z.shape == x.shape

