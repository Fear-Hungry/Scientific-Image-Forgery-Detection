#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.checkpoints import build_segmentation_from_config, load_checkpoint
from forgeryseg.dataset import build_train_index, load_image, load_mask_instances
from forgeryseg.inference import apply_tta, predict_image, undo_tta
from forgeryseg.metric import score_image
from forgeryseg.postprocess import prob_to_instances


def _parse_csv_list(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def _iter_stratified_folds(y: List[int], n_splits: int, seed: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in splitter.split(np.zeros(len(y)), y):
            yield train_idx, val_idx
        return
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    indices = np.arange(len(y))
    folds = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        label_indices = indices[y == label]
        rng.shuffle(label_indices)
        for i, idx in enumerate(label_indices):
            folds[i % n_splits].append(idx)

    for fold_idx in range(n_splits):
        val_idx = np.array(sorted(folds[fold_idx]))
        train_idx = np.array(sorted([i for i in indices if i not in set(val_idx)]))
        yield train_idx, val_idx


def _load_val_indices(samples, split_path: Path, fold: int, seed: int, folds: int) -> List[int]:
    if split_path.exists():
        with split_path.open("r") as f:
            payload = json.load(f)
        rel_paths = set(payload.get("val_cases", []))
        lookup = {s.rel_path.as_posix(): i for i, s in enumerate(samples)}
        return [lookup[p] for p in rel_paths if p in lookup]

    y = [0 if s.is_authentic else 1 for s in samples]
    folds_list = list(_iter_stratified_folds(y, folds, seed))
    _, val_idx = folds_list[fold]
    return list(val_idx)


def _find_ckpt(models_root: Path, model_id: str, fold: int) -> Path:
    candidates = [
        models_root / model_id / f"fold_{fold}" / "best.pt",
        models_root / model_id / f"fold_{fold}" / "last.pt",
        models_root / model_id / f"fold_{fold}" / "weights" / "best.pt",
        models_root / model_id / f"fold_{fold}" / "weights" / "last.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Checkpoint não encontrado para model_id={model_id} fold={fold} em {models_root}")


@dataclass(frozen=True)
class PostprocessCfg:
    threshold: float
    adaptive_threshold: bool
    threshold_factor: float
    min_area: int
    min_area_percent: float
    min_confidence: float
    closing: int
    closing_iters: int
    opening: int
    opening_iters: int
    fill_holes: bool
    median: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict segmentation OOF probabilities for Recod.ai/LUC and score with oF1")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root")
    parser.add_argument("--output-dir", default="outputs", help="Repo outputs root (contains models_seg/ and splits_seg/)")
    parser.add_argument("--model-id", required=True, help="Segmentation model_id (dir name under outputs/models_seg)")
    parser.add_argument("--preds-root", default="", help="Where to write probabilities (default: outputs/oof/<model_id>)")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--fold", type=int, default=-1, help="Run a single fold index")
    parser.add_argument("--seed", type=int, default=42, help="Seed for folds when split file is missing")
    parser.add_argument("--device", default="", help="Device override (e.g. cuda:0)")
    parser.add_argument("--tta", default="none,hflip,vflip", help="Comma-separated TTA modes")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size for inference (0 disables tiling)")
    parser.add_argument("--overlap", type=int, default=128, help="Tile overlap")
    parser.add_argument("--max-size", type=int, default=0, help="Optional resize long side")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples per fold (debug)")
    parser.add_argument("--no-save-probs", action="store_false", dest="save_probs", help="Disable saving .npy probabilities")
    parser.add_argument("--no-score", action="store_false", dest="score", help="Disable RecodAI oF1 scoring")

    # Postprocess knobs (for quick scoring). For full tuning use scripts/tune_thresholds.py afterwards.
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--adaptive-threshold", action="store_true", help="Use adaptive threshold (mean + factor * std)")
    parser.add_argument("--threshold-factor", type=float, default=0.3, help="Factor for adaptive threshold")
    parser.add_argument("--min-area", type=int, default=30, help="Minimum component area")
    parser.add_argument("--min-area-percent", type=float, default=0.0005, help="Minimum union area percent")
    parser.add_argument("--min-confidence", type=float, default=0.33, help="Minimum mean confidence")
    parser.add_argument("--closing", type=int, default=5, help="Morphological closing kernel size")
    parser.add_argument("--closing-iters", type=int, default=1, help="Morphological closing iterations")
    parser.add_argument("--opening", type=int, default=3, help="Morphological opening kernel size")
    parser.add_argument("--opening-iters", type=int, default=1, help="Morphological opening iterations")
    parser.add_argument("--fill-holes", action="store_true", help="Fill holes in binary mask")
    parser.add_argument("--median", type=int, default=0, help="Median smoothing kernel size (0=disabled, odd>=3)")

    parser.set_defaults(save_probs=True, score=True)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tta_modes = _parse_csv_list(args.tta)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    model_id = str(args.model_id)

    models_root = output_dir / "models_seg"
    splits_root = output_dir / "splits_seg" / model_id
    preds_root = Path(args.preds_root) if args.preds_root else (output_dir / "oof" / model_id)
    preds_root.mkdir(parents=True, exist_ok=True)

    samples = build_train_index(data_root)

    post_cfg = PostprocessCfg(
        threshold=float(args.threshold),
        adaptive_threshold=bool(args.adaptive_threshold),
        threshold_factor=float(args.threshold_factor),
        min_area=int(args.min_area),
        min_area_percent=float(args.min_area_percent),
        min_confidence=float(args.min_confidence),
        closing=int(args.closing),
        closing_iters=int(args.closing_iters),
        opening=int(args.opening),
        opening_iters=int(args.opening_iters),
        fill_holes=bool(args.fill_holes),
        median=int(args.median),
    )

    folds = list(range(int(args.folds)))
    if int(args.fold) >= 0:
        folds = [int(args.fold)]

    all_scores: list[float] = []
    for fold in folds:
        split_path = splits_root / f"fold_{fold}.json"
        val_idx = _load_val_indices(samples, split_path, fold, int(args.seed), int(args.folds))
        val_samples = [samples[i] for i in val_idx]
        if int(args.limit) > 0:
            val_samples = val_samples[: int(args.limit)]

        ckpt_path = _find_ckpt(models_root, model_id, fold)
        state, cfg = load_checkpoint(ckpt_path)
        model = build_segmentation_from_config(cfg)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        in_channels = int(cfg.get("in_channels", 3))
        use_freq_channels = bool(in_channels == 4)

        fold_scores: list[float] = []
        fold_out = preds_root / f"fold_{fold}"
        for sample in val_samples:
            image = load_image(sample.image_path)

            probs_sum: np.ndarray | None = None
            for mode in tta_modes:
                img_aug = apply_tta(image, mode, axes=(0, 1))
                prob = predict_image(
                    model,
                    img_aug,
                    device,
                    tile_size=int(args.tile_size),
                    overlap=int(args.overlap),
                    max_size=int(args.max_size),
                    use_freq_channels=use_freq_channels,
                )
                prob = undo_tta(prob, mode, axes=(0, 1))
                probs_sum = prob if probs_sum is None else (probs_sum + prob)

            assert probs_sum is not None
            prob_ens = probs_sum / float(max(len(tta_modes), 1))

            if bool(args.save_probs):
                out_path = (fold_out / sample.rel_path).with_suffix(".npy")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, prob_ens.astype(np.float32))

            if bool(args.score):
                gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
                pred_instances = prob_to_instances(
                    prob_ens,
                    threshold=post_cfg.threshold,
                    adaptive_threshold=post_cfg.adaptive_threshold,
                    threshold_factor=post_cfg.threshold_factor,
                    min_area=post_cfg.min_area,
                    min_area_percent=post_cfg.min_area_percent,
                    min_confidence=post_cfg.min_confidence,
                    closing_ksize=post_cfg.closing,
                    closing_iters=post_cfg.closing_iters,
                    opening_ksize=post_cfg.opening,
                    opening_iters=post_cfg.opening_iters,
                    fill_holes_enabled=post_cfg.fill_holes,
                    median_ksize=post_cfg.median,
                )
                fold_scores.append(score_image(gt_instances, pred_instances))

        if fold_scores:
            mean_score = float(np.mean(fold_scores))
            all_scores.extend(fold_scores)
            print(f"fold {fold}: mean RecodAI oF1 {mean_score:.6f} | n={len(fold_scores)} | ckpt={ckpt_path}")
        else:
            print(f"fold {fold}: (score disabled) | n={len(val_samples)} | ckpt={ckpt_path}")

    if all_scores:
        print(f"overall: mean RecodAI oF1 {float(np.mean(all_scores)):.6f} | n={len(all_scores)}")
    print("preds_root:", preds_root)


if __name__ == "__main__":
    main()

