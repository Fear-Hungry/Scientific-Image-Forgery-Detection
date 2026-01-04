from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from forgeryseg.kaggle import package_kaggle_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Package a Kaggle Dataset folder for offline notebooks.")
    ap.add_argument("--out-dir", type=Path, default=Path("kaggle_bundle"))
    ap.add_argument("--include-models", action="store_true")
    ap.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
    args = ap.parse_args()

    out_root = package_kaggle_dataset(
        out_dir=args.out_dir,
        include_models=bool(args.include_models),
        models_dir=args.models_dir,
    )
    print(f"Wrote Kaggle bundle at: {out_root.resolve()}")
    print("Upload this folder as a Kaggle Dataset and attach it to your notebook.")


if __name__ == "__main__":
    main()
