from __future__ import annotations

import os
from pathlib import Path


def configure_cache_dirs(cache_root: str | Path | None, *, force: bool = True) -> None:
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
    torch_home = root / "torch"
    hf_home = root / "hf"
    hub_cache = hf_home / "hub"

    if force:
        torch_home.mkdir(parents=True, exist_ok=True)
        hub_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
        return

    wrote_torch = "TORCH_HOME" not in os.environ
    wrote_hf = "HF_HOME" not in os.environ
    wrote_hub = "HUGGINGFACE_HUB_CACHE" not in os.environ

    if wrote_torch:
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)
    if wrote_hf or wrote_hub:
        hub_cache.mkdir(parents=True, exist_ok=True)
    if wrote_hf:
        os.environ["HF_HOME"] = str(hf_home)
    if wrote_hub:
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
