#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.dataset import build_test_index, load_image
from forgeryseg.inference import predict_image
from forgeryseg.models.fpn_convnext import build_model


def _load_checkpoint(path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("config", {})
    return ckpt, {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict test masks")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root")
    parser.add_argument("--output-dir", default="outputs", help="Output root for predictions")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--encoder-name", default="", help="Override encoder name")
    parser.add_argument("--encoder-weights", default="", help="Override encoder weights")
    parser.add_argument("--tile-size", type=int, default=0, help="Tile size for inference")
    parser.add_argument("--overlap", type=int, default=0, help="Tile overlap")
    parser.add_argument("--max-size", type=int, default=0, help="Resize long side to this")
    parser.add_argument("--device", default="", help="Device override")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    samples = build_test_index(args.data_root)

    state, cfg = _load_checkpoint(Path(args.checkpoint))
    encoder_name = args.encoder_name or cfg.get("encoder_name", "convnext_tiny")
    encoder_weights = args.encoder_weights or cfg.get("encoder_weights", "imagenet")
    if encoder_weights == "":
        encoder_weights = None

    model = build_model(encoder_name=encoder_name, encoder_weights=encoder_weights)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    preds_root = Path(args.output_dir) / "preds" / "test"
    for sample in samples:
        image = load_image(sample.image_path)
        pred = predict_image(
            model,
            image,
            device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            max_size=args.max_size,
        )
        pred_path = (preds_root / sample.rel_path).with_suffix(".npy")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, pred)

    print(f"Wrote predictions to {preds_root}")


if __name__ == "__main__":
    main()
