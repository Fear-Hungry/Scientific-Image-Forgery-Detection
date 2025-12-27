#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.dataset import build_test_index, build_train_index, load_mask_instances
from forgeryseg.metric import score_image
from forgeryseg.rle import encode_instances


def _case_id_for_csv(sample, unique_case_ids: bool) -> str:
    if not unique_case_ids:
        return sample.case_id
    if sample.label:
        return f"{sample.label}-{sample.case_id}"
    return sample.case_id


def _write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sanity submissions")
    parser.add_argument("--data-root", default="data/recodai", help="Path to dataset root")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--out-dir", default="outputs/sanity", help="Directory for CSVs")
    parser.add_argument("--unique-case-ids", action="store_true", help="Avoid duplicate case_id in train")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "train":
        samples = build_train_index(args.data_root)
    else:
        samples = build_test_index(args.data_root)

    modes = ["authentic"]
    if args.split == "train":
        modes.append("gt")

    for mode in modes:
        rows = []
        scores = []
        for sample in samples:
            if mode == "authentic":
                pred_instances = []
            else:
                pred_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []

            annotation = encode_instances(pred_instances)
            rows.append({
                "case_id": _case_id_for_csv(sample, args.unique_case_ids),
                "annotation": annotation,
            })

            if args.split == "train":
                gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
                scores.append(score_image(gt_instances, pred_instances))

        out_path = out_dir / f"sanity_{args.split}_{mode}.csv"
        _write_csv(rows, out_path)

        if scores:
            print(f"{mode}: mean score {float(np.mean(scores)):.6f}")
        else:
            print(f"{mode}: wrote {out_path}")


if __name__ == "__main__":
    main()
