#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.dataset import build_test_index, build_train_index
from forgeryseg.postprocess import prob_to_instances
from forgeryseg.rle import encode_instances


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("r") as f:
        return json.load(f)


def _prediction_path(preds_root: Path, sample_rel_path: Path) -> Path:
    return (preds_root / sample_rel_path).with_suffix(".npy")


def _filter_instances(instances, min_area: int):
    if not min_area:
        return instances
    filtered = []
    for inst in instances:
        if int(inst.sum()) >= min_area:
            filtered.append(inst)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission CSV from predictions")
    parser.add_argument("--data-root", default="data/recodai", help="Path to dataset root")
    parser.add_argument("--preds-root", required=True, help="Root directory with .npy predictions")
    parser.add_argument("--out-csv", default="outputs/submission.csv", help="Output CSV path")
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
    parser.add_argument("--config", default="", help="JSON config with threshold/min_area")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    threshold = float(cfg.get("threshold", args.threshold))
    adaptive_threshold = bool(cfg.get("adaptive_threshold", args.adaptive_threshold))
    threshold_factor = float(cfg.get("threshold_factor", args.threshold_factor))
    min_area = int(cfg.get("min_area", args.min_area))
    min_area_percent = float(cfg.get("min_area_percent", args.min_area_percent))
    min_confidence = float(cfg.get("min_confidence", args.min_confidence))
    closing = int(cfg.get("closing", args.closing))
    closing_iters = int(cfg.get("closing_iters", args.closing_iters))
    opening = int(cfg.get("opening", args.opening))
    opening_iters = int(cfg.get("opening_iters", args.opening_iters))
    fill_holes = bool(cfg.get("fill_holes", args.fill_holes))
    median = int(cfg.get("median", args.median))

    data_root = Path(args.data_root)
    preds_root = Path(args.preds_root)

    if args.split == "train":
        samples = build_train_index(data_root)
    else:
        samples = build_test_index(data_root)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in samples:
        pred_path = _prediction_path(preds_root, sample.rel_path)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        pred = np.load(pred_path)
        if pred.ndim == 2:
            instances = prob_to_instances(
                pred,
                threshold=threshold,
                adaptive_threshold=adaptive_threshold,
                threshold_factor=threshold_factor,
                min_area=min_area,
                min_area_percent=min_area_percent,
                min_confidence=min_confidence,
                closing_ksize=closing,
                closing_iters=closing_iters,
                opening_ksize=opening,
                opening_iters=opening_iters,
                fill_holes_enabled=fill_holes,
                median_ksize=median,
            )
        elif pred.ndim == 3:
            instances = [(p >= threshold).astype(np.uint8) for p in pred]
            instances = _filter_instances(instances, min_area)
        else:
            raise ValueError(f"Unsupported prediction shape: {pred.shape}")

        annotation = encode_instances(instances)
        rows.append({"case_id": sample.case_id, "annotation": annotation})

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
