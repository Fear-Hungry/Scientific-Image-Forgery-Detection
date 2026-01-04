from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from tqdm import tqdm

from forgeryseg.ensemble import ensemble_annotations, rank_weights_by_score
from forgeryseg.submission import list_ordered_cases


@dataclass(frozen=True)
class SubInput:
    path: Path
    score: float | None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "test", "supplemental"], default="test")
    ap.add_argument("--subs", type=Path, nargs="+", required=True)
    ap.add_argument("--scores", type=float, nargs="*")
    ap.add_argument("--method", choices=["weighted", "majority", "union", "intersection"], default="weighted")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    subs: list[SubInput] = []
    for idx, p in enumerate(args.subs):
        score = None
        if args.scores and idx < len(args.scores):
            score = float(args.scores[idx])
        subs.append(SubInput(path=p, score=score))

    sub_tables: list[dict[str, str]] = []
    for s in subs:
        df = pd.read_csv(s.path)
        if "case_id" not in df.columns or "annotation" not in df.columns:
            raise ValueError(f"{s.path} must have columns case_id,annotation")
        sub_tables.append(dict(zip(df["case_id"].astype(str), df["annotation"], strict=True)))

    if args.method == "weighted":
        if any(s.score is None for s in subs):
            weights = [1.0 / len(subs)] * len(subs)
        else:
            weights = rank_weights_by_score([float(s.score) for s in subs])
        print(f"weights={weights}")

    cases = list_ordered_cases(args.data_root, args.split)
    rows: list[dict[str, str]] = []
    for case in tqdm(cases, desc="Ensemble"):
        import cv2

        h, w = cv2.imread(str(case.image_path), cv2.IMREAD_UNCHANGED).shape[:2]
        anns = [table.get(case.case_id, "authentic") for table in sub_tables]
        ann_out = ensemble_annotations(
            anns,
            shape=(h, w),
            method=args.method,
            weights=weights if args.method == "weighted" else None,
            threshold=float(args.threshold),
        )
        rows.append({"case_id": case.case_id, "annotation": ann_out})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()
        writer.writerows(rows)
    n_auth = sum(1 for r in rows if r["annotation"] == "authentic")
    print(f"Wrote {args.out} ({n_auth}/{len(rows)} authentic)")


if __name__ == "__main__":
    main()
