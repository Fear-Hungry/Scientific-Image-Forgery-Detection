from __future__ import annotations

import os
from pathlib import Path


def configure_cache_dirs(cache_root: str | Path | None) -> None:
    """Configure cache directories for offline Kaggle runs.

    When Kaggle submissions run with internet OFF, pretrained weights must already be present in local caches.
    This helper points common libraries to a user-provided cache root.

    Expected structure (suggested):
    - <cache_root>/torch    -> torch hub checkpoints (TORCH_HOME)
    - <cache_root>/hf       -> Hugging Face cache root (HF_HOME)
    """
    if cache_root is None:
        return
    root = Path(cache_root)
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hf" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "hf" / "transformers"))

