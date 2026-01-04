from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from forgeryseg.ensemble_io import ensemble_submissions_from_csvs


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

    scores = list(args.scores) if args.scores else None
    ensemble_submissions_from_csvs(
        sub_paths=args.subs,
        data_root=args.data_root,
        split=args.split,  # type: ignore[arg-type]
        out_path=args.out,
        method=str(args.method),
        scores=scores,
        threshold=float(args.threshold),
    )


if __name__ == "__main__":
    main()

