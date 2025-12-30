from __future__ import annotations

import argparse
from pathlib import Path


def _parse_model_list(text: str) -> list[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def download_weights(model_name: str, out_dir: Path) -> Path:
    import timm
    import torch

    print(f"[timm] baixando pesos: {model_name}")
    model = timm.create_model(model_name, pretrained=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.pth"
    torch.save(model.state_dict(), out_path)
    print(f"[timm] pesos salvos em: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download timm pretrained weights and save as <model_name>.pth (state_dict).")
    parser.add_argument("--models", default="tf_efficientnet_b4_ns", help="Comma-separated timm model names")
    parser.add_argument("--out-dir", default="weights_cache", help="Output directory")
    parser.add_argument(
        "--also-copy-to-bundle",
        action="store_true",
        help="Also copy weights to recodai_bundle/weights_cache/ (useful when packaging a Kaggle Dataset).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    bundle_dir = Path("recodai_bundle") / "weights_cache"

    for model_name in _parse_model_list(args.models):
        out_path = download_weights(model_name, out_dir)
        if args.also_copy_to_bundle:
            import shutil

            bundle_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = bundle_dir / out_path.name
            shutil.copy2(out_path, bundle_path)
            print(f"[bundle] copiado para: {bundle_path}")

    print("Pronto. Para Kaggle (internet OFF), anexe um Dataset contendo `weights_cache/` ou `recodai_bundle/weights_cache/`.")


if __name__ == "__main__":
    main()
