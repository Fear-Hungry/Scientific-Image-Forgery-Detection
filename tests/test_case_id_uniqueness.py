from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from forgeryseg.dataset import list_cases
from forgeryseg.eval import validate_submission_format


def _write_png(path: Path, *, size: int = 8) -> None:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def test_list_cases_namespaces_overlapping_stems_in_train(tmp_path: Path) -> None:
    data_root = tmp_path / "recodai"
    (data_root / "train_images" / "authentic").mkdir(parents=True, exist_ok=True)
    (data_root / "train_images" / "forged").mkdir(parents=True, exist_ok=True)
    (data_root / "train_masks").mkdir(parents=True, exist_ok=True)

    _write_png(data_root / "train_images" / "authentic" / "10.png")
    _write_png(data_root / "train_images" / "forged" / "10.png")
    np.save(data_root / "train_masks" / "10.npy", np.zeros((1, 8, 8), dtype=np.uint8))

    cases = list_cases(data_root, "train", include_authentic=True, include_forged=True)
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"authentic_10", "forged_10"}

    csv_path = tmp_path / "submission_train.csv"
    pd.DataFrame({"case_id": ids, "annotation": ["authentic"] * len(ids)}).to_csv(csv_path, index=False)
    fmt = validate_submission_format(csv_path, data_root=data_root, split="train", validate_decode=False)
    assert fmt["missing_case_ids_in_csv"] == 0
    assert fmt["extra_case_ids_in_csv"] == 0
