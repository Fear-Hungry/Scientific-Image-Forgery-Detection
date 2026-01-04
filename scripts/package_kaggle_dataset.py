from __future__ import annotations

import argparse
import shutil
from pathlib import Path


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Package a Kaggle Dataset folder for offline notebooks.")
    ap.add_argument("--out-dir", type=Path, default=Path("kaggle_bundle"))
    ap.add_argument("--include-models", action="store_true")
    ap.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for name in ("src", "scripts", "configs", "notebooks", "docs"):
        src_dir = repo_root / name
        if not src_dir.exists():
            continue
        _copytree(src_dir, out_root / name)

    for file_name in ("README.md", "pyproject.toml", "requirements.txt", "requirements-kaggle.txt"):
        src_file = repo_root / file_name
        if not src_file.exists():
            continue
        shutil.copy2(src_file, out_root / file_name)

    if args.include_models:
        models_src = args.models_dir
        if not models_src.is_absolute():
            models_src = repo_root / models_src
        if models_src.exists():
            dst = out_root / "outputs" / "models"
            dst.parent.mkdir(parents=True, exist_ok=True)
            _copytree(models_src, dst)
        else:
            print(f"[warn] models dir not found: {models_src}")

    print(f"Wrote Kaggle bundle at: {out_root.resolve()}")
    print("Upload this folder as a Kaggle Dataset and attach it to your notebook.")


if __name__ == "__main__":
    main()

