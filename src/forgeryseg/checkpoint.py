from __future__ import annotations

from pathlib import Path

import torch


def load_flexible_state_dict(model: torch.nn.Module, checkpoint: str | Path) -> tuple[list[str], list[str]]:
    path = Path(checkpoint)
    obj = torch.load(path, map_location="cpu")

    state = None
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "weights"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    if state is None:
        state = obj

    missing, unexpected = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)

