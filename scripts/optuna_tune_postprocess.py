from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from forgeryseg.tuning import tune_postprocess_optuna


def main() -> None:
    ap = argparse.ArgumentParser(description="Bayesian tuning of postprocess params (Optuna) with prob_map caching.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "supplemental"], default="train")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/optuna"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=1, help="If >1, tune on a specific fold validation set.")
    ap.add_argument("--fold", type=int, default=0, help="Fold id to use when --folds > 1.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of val cases (stratified).")

    ap.add_argument("--use-tta", action="store_true", help="Use config TTA during inference (slower, more faithful).")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size for non-tiling inference.")

    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=None, help="Optional timeout (seconds).")
    ap.add_argument("--objective", choices=["mean_score", "mean_forged", "combo"], default="mean_score")
    ap.add_argument("--cache-path", type=Path, default=None, help="Optional explicit .npz cache path.")

    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config keys before tuning (ex.: --set inference.fft_gate.enabled=false)",
    )
    args = ap.parse_args()

    tune_postprocess_optuna(
        config_path=args.config,
        data_root=args.data_root,
        split=args.split,  # type: ignore[arg-type]
        out_dir=args.out_dir,
        device=args.device,
        base_overrides=args.overrides,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        folds=int(args.folds),
        fold=int(args.fold),
        limit=int(args.limit),
        use_tta=bool(args.use_tta),
        batch_size=int(args.batch_size),
        n_trials=int(args.trials),
        timeout_sec=args.timeout,
        objective=args.objective,  # type: ignore[arg-type]
        cache_path=args.cache_path,
    )


if __name__ == "__main__":
    main()
