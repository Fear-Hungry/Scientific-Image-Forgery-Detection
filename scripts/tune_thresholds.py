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
from forgeryseg.postprocess import binarize, extract_components


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip() != ""]


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip() != ""]


def _prediction_path(preds_root: Path, sample_rel_path: Path) -> Path:
    return (preds_root / sample_rel_path).with_suffix(".npy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune threshold/min_area for RecodAI F1")
    parser.add_argument("--data-root", default="data/recodai", help="Path to dataset root")
    parser.add_argument("--preds-root", required=True, help="Root directory with .npy predictions")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7", help="Comma-separated")
    parser.add_argument("--min-areas", default="0,32,64,128", help="Comma-separated")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (debug)")
    parser.add_argument("--out-config", default="configs/thresholds.json", help="Output JSON config")
    args = parser.parse_args()

    thresholds = _parse_floats(args.thresholds)
    min_areas = _parse_ints(args.min_areas)

    data_root = Path(args.data_root)
    preds_root = Path(args.preds_root)

    samples = build_train_index(data_root)
    if args.limit:
        samples = samples[: args.limit]

    best = {"score": -1.0, "threshold": None, "min_area": None}

    for threshold in thresholds:
        for min_area in min_areas:
            scores = []
            for sample in samples:
                pred_path = _prediction_path(preds_root, sample.rel_path)
                if not pred_path.exists():
                    raise FileNotFoundError(f"Missing prediction: {pred_path}")

                pred = np.load(pred_path)
                if pred.ndim == 2:
                    bin_mask = binarize(pred, threshold)
                    pred_instances = extract_components(bin_mask, min_area=min_area)
                elif pred.ndim == 3:
                    pred_instances = [(p >= threshold).astype(np.uint8) for p in pred]
                    if min_area:
                        pred_instances = [p for p in pred_instances if int(p.sum()) >= min_area]
                else:
                    raise ValueError(f"Unsupported prediction shape: {pred.shape}")

                gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
                scores.append(score_image(gt_instances, pred_instances))

            mean_score = float(np.mean(scores)) if scores else 0.0
            print(f"threshold={threshold:.2f} min_area={min_area} score={mean_score:.6f}")
            if mean_score > best["score"]:
                best = {"score": mean_score, "threshold": threshold, "min_area": min_area}

    out_path = Path(args.out_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"threshold": best["threshold"], "min_area": best["min_area"]}, f, indent=2)

    print(f"Best: score={best['score']:.6f} threshold={best['threshold']} min_area={best['min_area']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
