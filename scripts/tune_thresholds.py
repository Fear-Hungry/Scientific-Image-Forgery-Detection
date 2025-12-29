#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.dataset import build_train_index, load_mask_instances
from forgeryseg.metric import score_image
from forgeryseg.postprocess import adaptive_threshold_value, prob_to_instances


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip() != ""]


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip() != ""]


def _iter_stratified_folds(y: list[int], n_splits: int, seed: int):
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


def _prediction_path(preds_root: Path, sample_rel_path: Path, *, fold: int | None, use_folds: bool) -> Path:
    if use_folds and fold is not None:
        return (preds_root / f"fold_{fold}" / sample_rel_path).with_suffix(".npy")
    return (preds_root / sample_rel_path).with_suffix(".npy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune postprocess thresholds with stratified CV")
    parser.add_argument("--data-root", default="data/recodai", help="Path to dataset root")
    parser.add_argument("--preds-root", required=True, help="Root directory with .npy predictions")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified folds")
    parser.add_argument("--seed", type=int, default=42, help="Seed for folds")
    parser.add_argument("--fold", type=int, default=-1, help="Evaluate a single fold index")
    parser.add_argument("--adaptive-threshold", action="store_true", help="Use adaptive threshold (mean + factor * std)")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7", help="Fixed threshold grid (comma-separated)")
    parser.add_argument("--threshold-factors", default="0.2,0.3,0.4", help="Adaptive threshold factors grid (comma-separated)")
    parser.add_argument("--min-areas", default="0,30,64,128", help="Min area grid (comma-separated)")
    parser.add_argument("--min-area-percents", default="0.0002,0.0005,0.001", help="Min area percent grid")
    parser.add_argument("--min-confidences", default="0.30,0.33,0.36,0.40", help="Min confidence grid")
    parser.add_argument("--closing", type=int, default=0, help="Morphological closing kernel size (0=disabled)")
    parser.add_argument("--closing-iters", type=int, default=1, help="Morphological closing iterations")
    parser.add_argument("--opening", type=int, default=0, help="Morphological opening kernel size (0=disabled)")
    parser.add_argument("--opening-iters", type=int, default=1, help="Morphological opening iterations")
    parser.add_argument("--fill-holes", action="store_true", help="Fill holes in binary mask")
    parser.add_argument("--median", type=int, default=0, help="Median smoothing kernel size (0=disabled, odd>=3)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (debug)")
    parser.add_argument("--out-config", default="configs/thresholds.json", help="Output JSON config")
    args = parser.parse_args()

    thresholds = _parse_floats(args.thresholds)
    threshold_factors = _parse_floats(args.threshold_factors)
    min_areas = _parse_ints(args.min_areas)
    min_area_percents = _parse_floats(args.min_area_percents)
    min_confidences = _parse_floats(args.min_confidences)

    data_root = Path(args.data_root)
    preds_root = Path(args.preds_root)

    samples = build_train_index(data_root)
    if args.limit:
        samples = samples[: args.limit]

    if args.folds <= 1:
        raise ValueError("--folds must be >= 2 for cross-validation")

    has_fold_dirs = (preds_root / "fold_0").exists()
    if not has_fold_dirs:
        raise FileNotFoundError("Expected OOF predictions under preds_root/fold_*/ for cross-validation.")

    labels = [0 if s.is_authentic else 1 for s in samples]
    folds = list(_iter_stratified_folds(labels, args.folds, args.seed))
    fold_ids = list(range(len(folds)))
    if args.fold >= 0:
        fold_id = int(args.fold)
        folds = [folds[fold_id]]
        fold_ids = [fold_id]

    if args.adaptive_threshold and not threshold_factors:
        raise ValueError("--threshold-factors must be provided when --adaptive-threshold is set")
    if not args.adaptive_threshold and not thresholds:
        raise ValueError("--thresholds must be provided when adaptive threshold is disabled")

    best = {
        "score": -1.0,
        "adaptive_threshold": bool(args.adaptive_threshold),
        "threshold": None,
        "threshold_factor": None,
        "min_area": None,
        "min_area_percent": None,
        "min_confidence": None,
    }

    thr_grid = [None] if args.adaptive_threshold else thresholds
    factor_grid = threshold_factors if args.adaptive_threshold else [0.0]

    for thr in thr_grid:
        for factor in factor_grid:
            for min_area in min_areas:
                for min_area_percent in min_area_percents:
                    for min_conf in min_confidences:
                        scores = []
                        for fold_id, (_, val_idx) in zip(fold_ids, folds):
                            val_samples = [samples[i] for i in val_idx]
                            for sample in val_samples:
                                pred_path = _prediction_path(
                                    preds_root, sample.rel_path, fold=fold_id, use_folds=True
                                )
                                if not pred_path.exists():
                                    raise FileNotFoundError(f"Missing prediction: {pred_path}")

                                pred = np.load(pred_path)
                                if pred.ndim == 2:
                                    pred_instances = prob_to_instances(
                                        pred,
                                        threshold=float(thr if thr is not None else 0.5),
                                        adaptive_threshold=bool(args.adaptive_threshold),
                                        threshold_factor=float(factor),
                                        min_area=int(min_area),
                                        min_area_percent=float(min_area_percent),
                                        min_confidence=float(min_conf),
                                        closing_ksize=args.closing,
                                        closing_iters=args.closing_iters,
                                        opening_ksize=args.opening,
                                        opening_iters=args.opening_iters,
                                        fill_holes_enabled=args.fill_holes,
                                        median_ksize=args.median,
                                    )
                                elif pred.ndim == 3:
                                    pred_instances = []
                                    for p in pred:
                                        if args.adaptive_threshold:
                                            thr_val = adaptive_threshold_value(p, factor=float(factor))
                                        else:
                                            thr_val = float(thr if thr is not None else 0.5)
                                        inst = (p >= thr_val).astype(np.uint8)
                                        if int(inst.sum()) >= int(min_area):
                                            pred_instances.append(inst)
                                else:
                                    raise ValueError(f"Unsupported prediction shape: {pred.shape}")

                                gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
                                scores.append(score_image(gt_instances, pred_instances))

                        mean_score = float(np.mean(scores)) if scores else 0.0
                        thr_label = "adaptive" if args.adaptive_threshold else f"{float(thr):.2f}"
                        print(
                            f"thr={thr_label} factor={float(factor):.2f} min_area={min_area} "
                            f"min_area_percent={min_area_percent} min_conf={min_conf} score={mean_score:.6f}"
                        )
                        if mean_score > best["score"]:
                            best = {
                                "score": mean_score,
                                "adaptive_threshold": bool(args.adaptive_threshold),
                                "threshold": None if args.adaptive_threshold else float(thr),
                                "threshold_factor": float(factor),
                                "min_area": int(min_area),
                                "min_area_percent": float(min_area_percent),
                                "min_confidence": float(min_conf),
                            }

    out_path = Path(args.out_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "adaptive_threshold": bool(best["adaptive_threshold"]),
                "threshold": best["threshold"],
                "threshold_factor": best["threshold_factor"],
                "min_area": best["min_area"],
                "min_area_percent": best["min_area_percent"],
                "min_confidence": best["min_confidence"],
                "closing": int(args.closing),
                "closing_iters": int(args.closing_iters),
                "opening": int(args.opening),
                "opening_iters": int(args.opening_iters),
                "fill_holes": bool(args.fill_holes),
                "median": int(args.median),
            },
            f,
            indent=2,
        )

    print(
        "Best:",
        f"score={best['score']:.6f}",
        f"adaptive={best['adaptive_threshold']}",
        f"threshold={best['threshold']}",
        f"factor={best['threshold_factor']}",
        f"min_area={best['min_area']}",
        f"min_area_percent={best['min_area_percent']}",
        f"min_confidence={best['min_confidence']}",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
