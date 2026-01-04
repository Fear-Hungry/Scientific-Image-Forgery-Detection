import numpy as np
import torch

from forgeryseg.inference import TilingParams, predict_prob_map, predict_prob_map_tiled


class _ZeroLogitsModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)


def test_predict_prob_map_tiled_constant_matches_global() -> None:
    rng = np.random.default_rng(0)
    image = (rng.random((123, 245, 3)) * 255).astype(np.uint8)
    model = _ZeroLogitsModel()
    device = torch.device("cpu")

    p_global = predict_prob_map(model, image, input_size=64, device=device)
    p_tiled = predict_prob_map_tiled(
        model,
        image,
        input_size=64,
        device=device,
        tiling=TilingParams(tile_size=80, overlap=20, batch_size=2),
    )

    assert p_global.shape == (123, 245)
    assert p_tiled.shape == (123, 245)
    assert np.allclose(p_global, 0.5, atol=1e-6)
    assert np.allclose(p_tiled, 0.5, atol=1e-6)
    assert np.allclose(p_global, p_tiled, atol=1e-6)
