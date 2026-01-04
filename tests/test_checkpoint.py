from __future__ import annotations

import torch

from forgeryseg.checkpoint import load_flexible_state_dict


def test_load_flexible_state_dict_model_key(tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    ckpt = tmp_path / "ckpt.pth"
    torch.save({"model": model.state_dict()}, ckpt)

    missing, unexpected = load_flexible_state_dict(model, ckpt)
    assert missing == []
    assert unexpected == []
