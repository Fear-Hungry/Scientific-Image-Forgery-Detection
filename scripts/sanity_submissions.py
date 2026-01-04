from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from tqdm import tqdm

from forgeryseg.dataset import list_cases, load_mask_instances
from forgeryseg.metric import of1_score
from forgeryseg.rle import annotation_to_masks, masks_to_annotation


def _score_rows(
    rows: dict[str, str],
    *,
    data_root: Path,
    split: str,
) -> float:
    cases = list_cases(data_root, split, include_authentic=True, include_forged=True)
    scores: list[float] = []
    for case in cases:
        pred_ann = rows.get(case.case_id, "authentic")
        if case.mask_path is None:
            scores.append(1.0 if pred_ann == "authentic" else 0.0)
            continue

        gt_masks = load_mask_instances(case.mask_path)
        if pred_ann == "authentic":
            scores.append(0.0)
            continue

        # shape-safe decode using ground-truth H/W
        h, w = gt_masks[0].shape
        pred_masks = annotation_to_masks(pred_ann, (h, w))
        scores.append(of1_score(pred_masks, gt_masks))

    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "supplemental"], default="train")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = list_cases(args.data_root, args.split, include_authentic=True, include_forged=True)

    oracle_rows: list[dict[str, str]] = []
    for case in tqdm(cases, desc="Oracle"):
        if case.mask_path is None:
            oracle_rows.append({"case_id": case.case_id, "annotation": "authentic"})
            continue
        gt_masks = load_mask_instances(case.mask_path)
        ann = masks_to_annotation(gt_masks)
        oracle_rows.append({"case_id": case.case_id, "annotation": ann})

    oracle_path = args.out_dir / f"oracle_{args.split}.csv"
    pd.DataFrame(oracle_rows).to_csv(oracle_path, index=False)
    oracle_score = _score_rows(dict(zip([r["case_id"] for r in oracle_rows], [r["annotation"] for r in oracle_rows])) , data_root=args.data_root, split=args.split)
    print(f"oracle: {oracle_path} score={oracle_score:.6f}")

    all_auth_path = args.out_dir / f"all_authentic_{args.split}.csv"
    with all_auth_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()
        writer.writerows({"case_id": c.case_id, "annotation": "authentic"} for c in cases)
    all_auth_score = _score_rows({c.case_id: "authentic" for c in cases}, data_root=args.data_root, split=args.split)
    print(f"all_authentic: {all_auth_path} score={all_auth_score:.6f}")


if __name__ == "__main__":
    main()
