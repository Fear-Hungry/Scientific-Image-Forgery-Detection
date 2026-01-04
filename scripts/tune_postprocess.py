from __future__ import annotations

import argparse
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from forgeryseg.eval import score_submission_csv
from forgeryseg.submission import write_submission_csv


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    out: list[float] = []
    x = float(start)
    # include stop (with float tolerance)
    while x <= float(stop) + 1e-12:
        out.append(float(x))
        x += float(step)
    return out


def _slug_float(x: float) -> str:
    # 0.35 -> 0p35
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep postprocess thresholds using local oF1 (train/supplemental).")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "supplemental"], default="train")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/tune"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of cases for faster iteration.")

    ap.add_argument("--thresholds", type=float, nargs="*", default=None, help="Explicit list of prob_threshold values.")
    ap.add_argument("--thr-start", type=float, default=0.30)
    ap.add_argument("--thr-stop", type=float, default=0.80)
    ap.add_argument("--thr-step", type=float, default=0.05)

    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config keys (ex.: --set inference.postprocess.min_area=200)",
    )
    args = ap.parse_args()

    thresholds = list(args.thresholds) if args.thresholds else _frange(args.thr_start, args.thr_stop, args.thr_step)
    if not thresholds:
        raise SystemExit("No thresholds to evaluate")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    best_thr = None
    best_score = -math.inf
    best_csv: Path | None = None

    print(f"split={args.split} n_thresholds={len(thresholds)} limit={int(args.limit)} out_dir={args.out_dir}")
    for thr in thresholds:
        thr = float(thr)
        out_csv = args.out_dir / f"submission_thr{_slug_float(thr)}.csv"
        overrides = list(args.overrides) if args.overrides else []
        overrides.append(f"inference.postprocess.prob_threshold={thr}")

        write_submission_csv(
            config_path=args.config,
            data_root=args.data_root,
            split=args.split,  # type: ignore[arg-type]
            out_path=out_csv,
            device=device,
            limit=int(args.limit),
            overrides=overrides,
            path_roots=[args.config.parent, Path.cwd(), Path(__file__).resolve().parents[1]],
        )

        try:
            summary = score_submission_csv(out_csv, data_root=args.data_root, split=args.split)  # type: ignore[arg-type]
        except ImportError as e:
            print("\n[ERRO] Para calcular o oF1, precisa de SciPy (Hungarian matching).")
            print("Detalhe:", e)
            raise SystemExit(2) from e

        score = float(summary.mean_score)
        print(f"prob_threshold={thr:.4f} mean_score={score:.6f} mean_forged={summary.mean_forged:.6f}")
        if score > best_score:
            best_score = score
            best_thr = thr
            best_csv = out_csv

    assert best_thr is not None
    assert best_csv is not None
    print(f"\nBEST prob_threshold={best_thr:.4f} mean_score={best_score:.6f} csv={best_csv}")


if __name__ == "__main__":
    main()
