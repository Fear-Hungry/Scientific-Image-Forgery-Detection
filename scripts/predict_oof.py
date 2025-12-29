#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.dataset import build_train_index, load_image, load_mask_instances
from forgeryseg.inference import predict_image
from forgeryseg.metric import score_image
from forgeryseg.models.fpn_convnext import build_model
from forgeryseg.postprocess import prob_to_instances


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


def _load_checkpoint(path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("config", {})
    return ckpt, {}


def _load_val_indices(samples, splits_dir: Path, fold: int, seed: int, folds: int) -> List[int]:
    split_path = splits_dir / f"fold_{fold}.json"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict OOF masks and score RecodAI F1")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root")
    parser.add_argument("--output-dir", default="outputs", help="Output root for predictions")
    parser.add_argument("--splits-dir", default="outputs/splits", help="Directory with CV splits")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--fold", type=int, default=-1, help="Run a single fold index")
    parser.add_argument("--seed", type=int, default=42, help="Seed for folds when no split file")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path override")
    parser.add_argument("--encoder-name", default="", help="Override encoder name")
    parser.add_argument("--encoder-weights", default="", help="Override encoder weights")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--adaptive-threshold", action="store_true", help="Use adaptive threshold (mean + factor * std)")
    parser.add_argument("--threshold-factor", type=float, default=0.3, help="Factor for adaptive threshold")
    parser.add_argument("--min-area", type=int, default=30, help="Minimum component area")
    parser.add_argument("--min-area-percent", type=float, default=0.0, help="Minimum mask area fraction (0 disables)")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum mean prob inside mask (0 disables)")
    parser.add_argument("--closing", type=int, default=0, help="Morphological closing kernel size (0=disabled)")
    parser.add_argument("--closing-iters", type=int, default=1, help="Morphological closing iterations")
    parser.add_argument("--opening", type=int, default=0, help="Morphological opening kernel size (0=disabled)")
    parser.add_argument("--opening-iters", type=int, default=1, help="Morphological opening iterations")
    parser.add_argument("--fill-holes", action="store_true", help="Fill holes in binary mask")
    parser.add_argument("--median", type=int, default=0, help="Median smoothing kernel size (0=disabled, odd>=3)")
    parser.add_argument("--tile-size", type=int, default=0, help="Tile size for inference")
    parser.add_argument("--overlap", type=int, default=0, help="Tile overlap")
    parser.add_argument("--max-size", type=int, default=0, help="Resize long side to this")
    parser.add_argument("--device", default="", help="Device override")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    samples = build_train_index(args.data_root)

    folds = list(range(args.folds))
    if args.fold >= 0:
        folds = [args.fold]

    for fold in folds:
        split_dir = Path(args.splits_dir)
        val_idx = _load_val_indices(samples, split_dir, fold, args.seed, args.folds)
        val_samples = [samples[i] for i in val_idx]

        ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(args.output_dir) / "models" / f"fold_{fold}" / "best.pt"
        state, cfg = _load_checkpoint(ckpt_path)

        encoder_name = args.encoder_name or cfg.get("encoder_name", "convnext_tiny")
        encoder_weights = args.encoder_weights or cfg.get("encoder_weights", "imagenet")
        if encoder_weights == "":
            encoder_weights = None

        model = build_model(encoder_name=encoder_name, encoder_weights=encoder_weights)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        preds_root = Path(args.output_dir) / "preds" / f"fold_{fold}"
        scores = []
        for sample in val_samples:
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

            gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
            pred_instances = prob_to_instances(
                pred,
                threshold=args.threshold,
                adaptive_threshold=args.adaptive_threshold,
                threshold_factor=args.threshold_factor,
                min_area=args.min_area,
                min_area_percent=args.min_area_percent,
                min_confidence=args.min_confidence,
                closing_ksize=args.closing,
                closing_iters=args.closing_iters,
                opening_ksize=args.opening,
                opening_iters=args.opening_iters,
                fill_holes_enabled=args.fill_holes,
                median_ksize=args.median,
            )
            scores.append(score_image(gt_instances, pred_instances))

        mean_score = float(np.mean(scores)) if scores else 0.0
        print(f"fold {fold}: mean RecodAI F1 {mean_score:.6f}")


if __name__ == "__main__":
    main()
