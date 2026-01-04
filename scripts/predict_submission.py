from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from forgeryseg.submission import write_submission_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "test", "supplemental"], default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config keys (ex.: --set inference.postprocess.min_area=150)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    write_submission_csv(
        config_path=args.config,
        data_root=args.data_root,
        split=args.split,
        out_path=args.out,
        device=device,
        limit=int(args.limit),
        overrides=list(args.overrides) if args.overrides else None,
        path_roots=[args.config.parent, Path.cwd(), Path(__file__).resolve().parents[1]],
    )


if __name__ == "__main__":
    main()
