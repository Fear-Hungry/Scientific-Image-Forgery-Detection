from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from forgeryseg.training.dino_decoder import train_dino_decoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--aug", choices=["none", "basic", "robust"], default=None)
    ap.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default=None)
    ap.add_argument("--lr-min", type=float, default=None)
    ap.add_argument("--max-lr", type=float, default=None)
    ap.add_argument("--pct-start", type=float, default=None)
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config keys (ex.: --set train.epochs=10)",
    )
    args = ap.parse_args()

    train_dino_decoder(
        config_path=args.config,
        data_root=args.data_root,
        out_path=args.out,
        device=args.device,
        overrides=list(args.overrides) if args.overrides else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
        folds=args.folds,
        fold=args.fold,
        aug=args.aug,  # type: ignore[arg-type]
        scheduler=args.scheduler,  # type: ignore[arg-type]
        lr_min=args.lr_min,
        max_lr=args.max_lr,
        pct_start=args.pct_start,
    )


if __name__ == "__main__":
    main()
