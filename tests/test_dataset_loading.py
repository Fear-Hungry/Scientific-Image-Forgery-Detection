from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from forgeryseg.dataset import RecodaiDataset


def _write_png(path: Path, *, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


def test_recodai_dataset_loads_images_and_masks(tmp_path: Path) -> None:
    data_root = tmp_path / "recodai"
    (data_root / "train_images" / "authentic").mkdir(parents=True, exist_ok=True)
    (data_root / "train_images" / "forged").mkdir(parents=True, exist_ok=True)
    (data_root / "train_masks").mkdir(parents=True, exist_ok=True)

    _write_png(data_root / "train_images" / "authentic" / "a1.png")
    _write_png(data_root / "train_images" / "authentic" / "a2.png")

    _write_png(data_root / "train_images" / "forged" / "f1.png")
    _write_png(data_root / "train_images" / "forged" / "f2.png")

    m1 = np.zeros((8, 8), dtype=np.uint8)
    m1[0:2, 0:2] = 1
    m2 = np.zeros((8, 8), dtype=np.uint8)
    m2[6:8, 6:8] = 1
    np.save(data_root / "train_masks" / "f1.npy", np.stack([m1, m2], axis=0))

    m3 = np.zeros((8, 8), dtype=np.uint8)
    m3[3:5, 1:4] = 1
    np.save(data_root / "train_masks" / "f2.npy", np.stack([m3], axis=0))

    ds = RecodaiDataset(data_root, "train", training=True, transforms=None)
    assert len(ds) == 4

    s_f1 = next(s for s in ds if s.case_id == "f1")
    assert s_f1.is_forged is True
    assert s_f1.mask.shape == (1, 8, 8)
    union = np.clip(m1 + m2, 0, 1).astype(np.uint8)
    assert np.array_equal(s_f1.mask[0].numpy().astype(np.uint8), union)

    s_a1 = next(s for s in ds if s.case_id == "a1")
    assert s_a1.is_forged is False
    assert int(s_a1.mask.sum().item()) == 0


def test_recodai_dataset_training_false_returns_zero_masks(tmp_path: Path) -> None:
    data_root = tmp_path / "recodai"
    (data_root / "train_images" / "authentic").mkdir(parents=True, exist_ok=True)
    (data_root / "train_images" / "forged").mkdir(parents=True, exist_ok=True)
    (data_root / "train_masks").mkdir(parents=True, exist_ok=True)

    _write_png(data_root / "train_images" / "authentic" / "a1.png")
    _write_png(data_root / "train_images" / "forged" / "f1.png")

    m = np.zeros((8, 8), dtype=np.uint8)
    m[0:2, 0:2] = 1
    np.save(data_root / "train_masks" / "f1.npy", np.stack([m], axis=0))

    ds = RecodaiDataset(data_root, "train", training=False, transforms=None)
    s_f1 = next(s for s in ds if s.case_id == "f1")
    assert int(s_f1.mask.sum().item()) == 0
