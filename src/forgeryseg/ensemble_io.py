from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .ensemble import ensemble_annotations, rank_weights_by_score
from .submission import list_ordered_cases
from .typing import Pathish, Split


@dataclass(frozen=True)
class EnsembleResult:
    out_path: Path
    n_rows: int
    n_authentic: int
    weights: list[float] | None


def ensemble_submissions_from_csvs(
    *,
    sub_paths: list[Pathish],
    data_root: Pathish,
    split: Split,
    out_path: Pathish,
    method: str = "weighted",
    weights: list[float] | None = None,
    scores: list[float] | None = None,
    threshold: float = 0.5,
) -> EnsembleResult:
    if not sub_paths:
        raise ValueError("sub_paths cannot be empty")

    subs: list[Path] = [Path(p) for p in sub_paths]
    out_path = Path(out_path)

    sub_tables: list[dict[str, str]] = []
    for p in subs:
        df = pd.read_csv(p)
        if "case_id" not in df.columns or "annotation" not in df.columns:
            raise ValueError(f"{p} precisa ter colunas case_id,annotation")
        sub_tables.append(dict(zip(df["case_id"].astype(str), df["annotation"], strict=True)))

    if method == "weighted":
        if weights is None:
            if scores is None:
                weights = [1.0 / len(subs)] * len(subs)
            else:
                weights = rank_weights_by_score(scores)
        if len(weights) != len(subs):
            raise ValueError("weights precisa ter o mesmo tamanho de sub_paths")
        print(f"ensemble weights={weights}")

    cases = list_ordered_cases(Path(data_root), split)

    import cv2

    rows: list[dict[str, str]] = []
    for case in tqdm(cases, desc="Ensemble"):
        h, w = cv2.imread(str(case.image_path), cv2.IMREAD_UNCHANGED).shape[:2]
        anns = [t.get(case.case_id, "authentic") for t in sub_tables]
        ann_out = ensemble_annotations(
            anns,
            shape=(h, w),
            method=method,  # type: ignore[arg-type]
            weights=weights if method == "weighted" else None,
            threshold=float(threshold),
        )
        rows.append({"case_id": case.case_id, "annotation": ann_out})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()
        writer.writerows(rows)

    n_auth = sum(1 for r in rows if r["annotation"] == "authentic")
    print(f"Wrote {out_path} ({n_auth}/{len(rows)} authentic)")
    return EnsembleResult(out_path=out_path, n_rows=len(rows), n_authentic=n_auth, weights=weights)

