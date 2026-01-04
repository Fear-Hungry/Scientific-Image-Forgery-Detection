from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from forgeryseg.eval import load_submission_csv, score_submission_detailed, validate_submission_format


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate and (optionally) score a submission.csv locally.")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "supplemental", "test"], default="test")
    ap.add_argument("--csv", type=Path, required=True, help="Path to submission.csv")
    ap.add_argument("--no-decode-check", action="store_true", help="Skip decode validation (faster).")
    ap.add_argument("--show-worst", type=int, default=20, help="Show N worst cases (train/supplemental only).")
    ap.add_argument("--out-cases", type=Path, default=None, help="Optional CSV with per-case scores.")
    args = ap.parse_args()

    fmt = validate_submission_format(
        args.csv,
        data_root=args.data_root,
        split=args.split,  # type: ignore[arg-type]
        validate_decode=not bool(args.no_decode_check),
    )
    print("[Format check]")
    print(json.dumps(fmt, indent=2, ensure_ascii=False))

    if args.split == "test":
        print("\nSplit=test => não há ground truth; não dá para calcular score real aqui.")
        return

    print("\n[Local score]")
    pred = load_submission_csv(args.csv)
    try:
        summary, per_case = score_submission_detailed(
            pred,
            data_root=args.data_root,
            split=args.split,  # type: ignore[arg-type]
            progress=True,
        )
    except ImportError as e:
        print("\n[ERRO] Para calcular o oF1, precisa de SciPy (Hungarian matching).")
        print("Detalhe:", e)
        return

    print(json.dumps(summary.as_dict(csv_path=args.csv, split=args.split), indent=2, ensure_ascii=False))

    if args.out_cases is not None:
        args.out_cases.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dataclasses.asdict(x) for x in per_case]).to_csv(args.out_cases, index=False)
        print(f"Wrote per-case scores to {args.out_cases}")

    n = int(max(0, args.show_worst))
    if n:
        worst = sorted(per_case, key=lambda x: x.score)[:n]
        df = pd.DataFrame([dataclasses.asdict(x) for x in worst])
        print(f"\n[Worst {len(df)}/{len(per_case)} cases]")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

