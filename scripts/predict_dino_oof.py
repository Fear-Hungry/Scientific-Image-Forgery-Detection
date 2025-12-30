#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.dino_oof import DEFAULT_TTA, predict_dino_oof


def _parse_csv_list(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict DINOv2 OOF masks and score RecodAI F1")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root")
    parser.add_argument("--preds-root", default="outputs/preds_dino", help="Root directory for .npy predictions")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--fold", type=int, default=-1, help="Run a single fold index")
    parser.add_argument("--seed", type=int, default=42, help="Seed for folds")
    parser.add_argument("--dino-path", default="", help="DINOv2 model id or local path (empty = use ckpt config)")
    parser.add_argument("--head-ckpt", default="", help="Head checkpoint path override")
    parser.add_argument("--head-ckpt-dir", default="outputs/models_dino", help="Directory with fold checkpoints")
    parser.add_argument("--image-size", type=int, default=0, help="Inference image size (0 = use ckpt config)")
    parser.add_argument("--decoder-dropout", type=float, default=-1.0, help="Decoder dropout (-1 = use ckpt config)")
    parser.add_argument("--device", default="", help="Device override")
    parser.add_argument("--tta", default=",".join(DEFAULT_TTA), help="Comma-separated TTA modes")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (debug)")
    parser.add_argument("--no-save-probs", action="store_false", dest="save_probs", help="Disable saving .npy probabilities")
    parser.add_argument("--no-score", action="store_false", dest="score", help="Disable RecodAI F1 scoring")
    parser.add_argument("--run-dir", default="", help="Run directory for artifacts (default: runs/<timestamp>)")

    parser.add_argument("--threshold-factor", type=float, default=0.3, help="Adaptive threshold factor")
    parser.add_argument("--min-area", type=int, default=30, help="Minimum component area")
    parser.add_argument("--min-area-percent", type=float, default=0.0005, help="Minimum union area percent")
    parser.add_argument("--min-confidence", type=float, default=0.33, help="Minimum mean confidence")
    parser.add_argument("--closing", type=int, default=5, help="Closing kernel size")
    parser.add_argument("--opening", type=int, default=3, help="Opening kernel size")
    parser.add_argument("--morph-iters", type=int, default=1, help="Morphology iterations")
    parser.add_argument("--closing-iters", type=int, default=0, help="Closing iterations (0 = use morph-iters)")
    parser.add_argument("--opening-iters", type=int, default=0, help="Opening iterations (0 = use morph-iters)")
    parser.add_argument("--local-files-only", action="store_true", help="Disable HF downloads (use local cache)")
    parser.add_argument("--cache-dir", default="", help="HF cache dir override")

    parser.set_defaults(save_probs=True, score=True)
    args = parser.parse_args()

    tta_modes = _parse_csv_list(args.tta)

    closing_iters = args.closing_iters if int(args.closing_iters) > 0 else None
    opening_iters = args.opening_iters if int(args.opening_iters) > 0 else None

    predict_dino_oof(
        data_root=args.data_root,
        preds_root=args.preds_root,
        folds=args.folds,
        fold=args.fold,
        seed=args.seed,
        dino_path=args.dino_path or None,
        head_ckpt=args.head_ckpt or None,
        head_ckpt_dir=args.head_ckpt_dir,
        image_size=args.image_size,
        decoder_dropout=args.decoder_dropout,
        device=args.device or None,
        tta_modes=tta_modes,
        limit=args.limit,
        save_probs=bool(args.save_probs),
        score=bool(args.score),
        run_dir=args.run_dir or None,
        threshold_factor=args.threshold_factor,
        min_area=args.min_area,
        min_area_percent=args.min_area_percent,
        min_confidence=args.min_confidence,
        closing=args.closing,
        opening=args.opening,
        morph_iters=args.morph_iters,
        closing_iters=closing_iters,
        opening_iters=opening_iters,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir.strip() or None,
    )


if __name__ == "__main__":
    main()
