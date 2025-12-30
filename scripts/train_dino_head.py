#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.dino_oof import train_dino_head


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINOv2 lightweight head for copy-move segmentation")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root")
    parser.add_argument("--output-dir", default="outputs/models_dino", help="Output directory for checkpoints")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to train")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split")
    parser.add_argument("--dino-path", default="facebook/dinov2-base", help="DINOv2 model id or local path")
    parser.add_argument("--image-size", type=int, default=512, help="Train image size")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--decoder-dropout", type=float, default=0.0, help="Decoder dropout")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--device", default="", help="Device override (cpu/cuda)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--local-files-only", action="store_true", help="Disable HF downloads (use local cache)")
    parser.add_argument("--cache-dir", default="", help="HF cache dir override")
    args = parser.parse_args()

    train_dino_head(
        data_root=args.data_root,
        output_dir=args.output_dir,
        folds=args.folds,
        fold=args.fold,
        seed=args.seed,
        dino_path=args.dino_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        decoder_dropout=args.decoder_dropout,
        patience=args.patience,
        device=args.device or None,
        num_workers=args.num_workers,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir.strip() or None,
    )


if __name__ == "__main__":
    main()
