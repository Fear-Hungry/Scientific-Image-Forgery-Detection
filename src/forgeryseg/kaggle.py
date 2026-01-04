from __future__ import annotations

import shutil
from pathlib import Path

from .typing import Pathish


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".ipynb_checkpoints",
            ".pytest_cache",
            ".venv",
            ".git",
            "data",
            "outputs",
            "runs",
            "logs",
        ),
    )


def package_kaggle_dataset(
    *,
    out_dir: Pathish = "kaggle_bundle",
    include_models: bool = False,
    models_dir: Pathish = "outputs/models",
    repo_root: Path | None = None,
) -> Path:
    """
    Create a folder ready to upload as a Kaggle Dataset (useful for offline notebooks).

    Copies a minimal subset of this repo (src/scripts/configs/notebooks/docs + key files).
    Optionally includes model checkpoints under outputs/models.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for name in ("src", "scripts", "configs", "notebooks", "docs"):
        src_dir = repo_root / name
        if src_dir.exists():
            _copytree(src_dir, out_root / name)

    for file_name in ("README.md", "pyproject.toml", "requirements.txt", "requirements-kaggle.txt"):
        src_file = repo_root / file_name
        if src_file.exists():
            shutil.copy2(src_file, out_root / file_name)

    if include_models:
        models_src = Path(models_dir)
        if not models_src.is_absolute():
            models_src = repo_root / models_src
        if models_src.exists():
            dst = out_root / "outputs" / "models"
            dst.parent.mkdir(parents=True, exist_ok=True)
            _copytree(models_src, dst)
        else:
            print(f"[warn] models dir not found: {models_src}")

    return out_root

