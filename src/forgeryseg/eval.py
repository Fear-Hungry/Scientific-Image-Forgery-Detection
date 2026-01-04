from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import list_cases, load_mask_instances
from .metric import of1_score
from .rle import annotation_to_masks
from .typing import Pathish, Split

try:  # optional in some minimal environments
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def is_authentic_annotation(annotation: str | float | None) -> bool:
    if annotation is None:
        return True
    if isinstance(annotation, float) and np.isnan(annotation):
        return True
    s = str(annotation).strip().lower()
    return (s == "") or (s == "authentic")


def load_submission_csv(csv_path: Pathish) -> dict[str, str | float | None]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if "case_id" not in df.columns or "annotation" not in df.columns:
        raise ValueError(f"{csv_path} precisa ter colunas: case_id, annotation")
    df["case_id"] = df["case_id"].astype(str)
    if df["case_id"].duplicated().any():
        dup = df.loc[df["case_id"].duplicated(), "case_id"].iloc[:5].tolist()
        raise ValueError(f"{csv_path} tem case_id duplicado (ex.: {dup})")
    return dict(zip(df["case_id"], df["annotation"], strict=True))


def validate_submission_format(
    csv_path: Pathish,
    *,
    data_root: Pathish,
    split: Split,
    validate_decode: bool = True,
) -> dict[str, Any]:
    """
    Validates:
      - CSV has columns case_id, annotation
      - case_id set matches expected (for test: sample_submission.csv order)
      - annotations decode without errors (shape read from image) when non-empty
    """
    csv_path = Path(csv_path)
    data_root = Path(data_root)

    pred = load_submission_csv(csv_path)
    cases = list_cases(data_root, split, include_authentic=True, include_forged=True)
    case_by_id = {c.case_id: c for c in cases}

    if split == "test":
        sample_path = data_root / "sample_submission.csv"
        expected_ids = (
            pd.read_csv(sample_path)["case_id"].astype(str).tolist() if sample_path.exists() else list(case_by_id.keys())
        )
    else:
        expected_ids = list(case_by_id.keys())

    missing_in_csv = [cid for cid in expected_ids if cid not in pred]
    extra_in_csv = [cid for cid in pred.keys() if cid not in case_by_id]

    decode_errors: list[tuple[str, str]] = []
    decoded_non_empty = 0

    if validate_decode and Image is None:
        validate_decode = False

    if validate_decode:
        for cid, ann in tqdm(pred.items(), desc="Validating RLE"):
            if cid not in case_by_id:
                continue
            if is_authentic_annotation(ann):
                continue

            case = case_by_id[cid]
            try:
                assert Image is not None  # for type checkers
                with Image.open(case.image_path) as im:
                    w, h = im.size
                masks = annotation_to_masks(ann, (h, w))
                if masks and any(np.any(m) for m in masks):
                    decoded_non_empty += 1
            except Exception as e:
                decode_errors.append((cid, str(e)))

    return {
        "csv_path": str(csv_path),
        "split": split,
        "n_cases_in_split": len(cases),
        "n_rows_in_csv": len(pred),
        "missing_case_ids_in_csv": len(missing_in_csv),
        "extra_case_ids_in_csv": len(extra_in_csv),
        "n_decode_errors": len(decode_errors),
        "n_non_empty_decoded": int(decoded_non_empty),
        "decode_validated": bool(validate_decode),
        "sample_missing_ids": missing_in_csv[:5],
        "sample_extra_ids": extra_in_csv[:5],
        "sample_decode_errors": decode_errors[:5],
    }


@dataclass(frozen=True)
class ScoreSummary:
    mean_score: float
    mean_authentic: float
    mean_forged: float
    n_cases: int
    n_authentic: int
    n_forged: int
    auth_pred_as_forged: int
    forg_pred_as_auth: int
    decode_errors_scoring: int

    def as_dict(self, *, csv_path: Path | None = None, split: Split | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mean_score": float(self.mean_score),
            "mean_authentic": float(self.mean_authentic),
            "mean_forged": float(self.mean_forged),
            "n_cases": int(self.n_cases),
            "n_authentic": int(self.n_authentic),
            "n_forged": int(self.n_forged),
            "auth_pred_as_forged": int(self.auth_pred_as_forged),
            "forg_pred_as_auth": int(self.forg_pred_as_auth),
            "decode_errors_scoring": int(self.decode_errors_scoring),
        }
        if csv_path is not None:
            out["csv_path"] = str(csv_path)
        if split is not None:
            out["split"] = split
        return out


def score_submission_annotations(
    pred: dict[str, str | float | None],
    *,
    data_root: Pathish,
    split: Split,
) -> ScoreSummary:
    cases = list_cases(data_root, split, include_authentic=True, include_forged=True)

    scores_all: list[float] = []
    scores_auth: list[float] = []
    scores_forg: list[float] = []
    n_auth_pred_as_forged = 0
    n_forg_pred_as_auth = 0
    decode_errors = 0

    for case in tqdm(cases, desc=f"Scoring {split}"):
        ann = pred.get(case.case_id, "authentic")

        if case.mask_path is None:
            s = 1.0 if is_authentic_annotation(ann) else 0.0
            if s == 0.0:
                n_auth_pred_as_forged += 1
            scores_all.append(s)
            scores_auth.append(s)
            continue

        if is_authentic_annotation(ann):
            n_forg_pred_as_auth += 1
            s = 0.0
            scores_all.append(s)
            scores_forg.append(s)
            continue

        gt_masks = load_mask_instances(case.mask_path)
        h, w = gt_masks[0].shape
        try:
            pred_masks = annotation_to_masks(ann, (h, w))
            s = float(of1_score(pred_masks, gt_masks))
        except ImportError:
            raise
        except Exception:
            decode_errors += 1
            s = 0.0

        scores_all.append(s)
        scores_forg.append(s)

    def _mean(x: list[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    return ScoreSummary(
        mean_score=_mean(scores_all),
        mean_authentic=_mean(scores_auth),
        mean_forged=_mean(scores_forg),
        n_cases=len(scores_all),
        n_authentic=len(scores_auth),
        n_forged=len(scores_forg),
        auth_pred_as_forged=n_auth_pred_as_forged,
        forg_pred_as_auth=n_forg_pred_as_auth,
        decode_errors_scoring=decode_errors,
    )


def score_submission_csv(
    csv_path: Pathish,
    *,
    data_root: Pathish,
    split: Split,
) -> ScoreSummary:
    csv_path = Path(csv_path)
    pred = load_submission_csv(csv_path)
    return score_submission_annotations(pred, data_root=data_root, split=split)
