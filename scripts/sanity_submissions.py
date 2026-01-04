from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from tqdm import tqdm

from forgeryseg.dataset import list_cases, load_mask_instances
from forgeryseg.eval import score_submission_annotations
from forgeryseg.rle import masks_to_annotation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "supplemental"], default="train")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = list_cases(args.data_root, args.split, include_authentic=True, include_forged=True)

    oracle_rows: list[dict[str, str]] = []
    oracle_pred: dict[str, str] = {}
    for case in tqdm(cases, desc="Oracle"):
        if case.mask_path is None:
            ann = "authentic"
        else:
            gt_masks = load_mask_instances(case.mask_path)
            ann = masks_to_annotation(gt_masks)
        oracle_rows.append({"case_id": case.case_id, "annotation": ann})
        oracle_pred[case.case_id] = ann

    oracle_path = args.out_dir / f"oracle_{args.split}.csv"
    pd.DataFrame(oracle_rows).to_csv(oracle_path, index=False)
    oracle_score = score_submission_annotations(oracle_pred, data_root=args.data_root, split=args.split)
    print(f"oracle: {oracle_path} score={oracle_score.mean_score:.6f}")

    all_auth_path = args.out_dir / f"all_authentic_{args.split}.csv"
    with all_auth_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()
        writer.writerows({"case_id": c.case_id, "annotation": "authentic"} for c in cases)

    all_auth_pred = {c.case_id: "authentic" for c in cases}
    all_auth_score = score_submission_annotations(all_auth_pred, data_root=args.data_root, split=args.split)
    print(f"all_authentic: {all_auth_path} score={all_auth_score.mean_score:.6f}")


if __name__ == "__main__":
    main()

