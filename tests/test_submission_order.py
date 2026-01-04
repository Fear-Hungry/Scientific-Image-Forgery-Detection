from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from forgeryseg.submission import list_ordered_cases


def _write_png(path: Path, *, size: int = 8) -> None:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def test_list_ordered_cases_uses_sample_submission_order(tmp_path: Path) -> None:
    data_root = tmp_path / "recodai"
    (data_root / "test_images").mkdir(parents=True, exist_ok=True)

    _write_png(data_root / "test_images" / "10.png")
    _write_png(data_root / "test_images" / "2.png")
    _write_png(data_root / "test_images" / "1.png")

    pd.DataFrame({"case_id": ["2", "1", "10"], "annotation": ["authentic"] * 3}).to_csv(
        data_root / "sample_submission.csv", index=False
    )

    ordered = list_ordered_cases(data_root, "test")
    assert [c.case_id for c in ordered] == ["2", "1", "10"]
