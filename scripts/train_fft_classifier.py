from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from forgeryseg.training.fft_classifier import train_fft_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="none")
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--max-lr", type=float, default=0.0)
    ap.add_argument("--pct-start", type=float, default=0.1)
    args = ap.parse_args()

    train_fft_classifier(
        config_path=args.config,
        data_root=args.data_root,
        out_path=args.out,
        device=args.device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        folds=int(args.folds),
        fold=int(args.fold),
        no_aug=bool(args.no_aug),
        scheduler=args.scheduler,  # type: ignore[arg-type]
        lr_min=float(args.lr_min),
        max_lr=float(args.max_lr),
        pct_start=float(args.pct_start),
    )


if __name__ == "__main__":
    main()

