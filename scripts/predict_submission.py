from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

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
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = json.loads(args.config.read_text())
    _ = cfg  # keep early JSON parse errors in this script for fast feedback
    write_submission_csv(
        config_path=args.config,
        data_root=args.data_root,
        split=args.split,
        out_path=args.out,
        device=device,
        limit=int(args.limit),
        path_roots=[args.config.parent, Path.cwd(), Path(__file__).resolve().parents[1]],
    )


if __name__ == "__main__":
    main()
