#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.dataset import build_train_index, load_mask_instances
from forgeryseg.metric import score_image
from forgeryseg.rle import encode_instances


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


def _case_id_for_csv(sample, unique_case_ids: bool) -> str:
    if not unique_case_ids:
        return sample.case_id
    if sample.label:
        return f"{sample.label}-{sample.case_id}"
    return sample.case_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validation sanity checks for RecodAI F1")
    parser.add_argument("--data-root", default="data/recodai", help="Path to dataset root")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode",
        choices=["gt", "authentic"],
        default="gt",
        help="Prediction mode: gt (use GT) or authentic (predict none)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (debug)")
    parser.add_argument("--out-dir", default="", help="Write per-fold CSVs to this directory")
    parser.add_argument(
        "--unique-case-ids",
        action="store_true",
        help="Append label to case_id to avoid duplicates in train CSVs",
    )
    args = parser.parse_args()

    samples = build_train_index(args.data_root)
    if args.limit:
        samples = samples[: args.limit]

    y = [0 if s.is_authentic else 1 for s in samples]

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    fold_scores = []
    for fold_idx, (_, val_idx) in enumerate(_iter_stratified_folds(y, args.folds, args.seed)):
        fold_rows = []
        scores = []
        for idx in val_idx:
            sample = samples[int(idx)]
            gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
            if args.mode == "gt":
                pred_instances = gt_instances
            else:
                pred_instances = []

            score = score_image(gt_instances, pred_instances)
            scores.append(score)

            if out_dir:
                annotation = encode_instances(pred_instances)
                fold_rows.append({
                    "case_id": _case_id_for_csv(sample, args.unique_case_ids),
                    "annotation": annotation,
                })

        fold_mean = float(np.mean(scores)) if scores else 0.0
        fold_scores.append(fold_mean)
        print(f"fold {fold_idx}: {fold_mean:.6f}")

        if out_dir:
            out_path = out_dir / f"cv_fold_{fold_idx}_{args.mode}.csv"
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
                writer.writeheader()
                writer.writerows(fold_rows)

    overall = float(np.mean(fold_scores)) if fold_scores else 0.0
    print(f"mean: {overall:.6f}")


if __name__ == "__main__":
    main()
