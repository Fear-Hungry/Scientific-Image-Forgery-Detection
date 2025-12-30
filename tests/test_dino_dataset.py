from pathlib import Path

import numpy as np
import torch
from PIL import Image

from forgeryseg.dataset import DinoSegDataset, Sample


def _make_sample(tmp_path: Path) -> Sample:
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[:, :15] = (255, 0, 0)
    img[:, 15:] = (0, 255, 0)
    img_path = tmp_path / "img.png"
    Image.fromarray(img).save(img_path)

    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[:, :15] = 1
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)

    return Sample(
        case_id="1",
        image_path=img_path,
        mask_path=mask_path,
        is_authentic=False,
        split="train",
        label="forged",
        rel_path=Path("img.png"),
    )


def test_dino_seg_dataset_shapes_and_types(tmp_path: Path):
    sample = _make_sample(tmp_path)
    ds = DinoSegDataset([sample], image_size=16, train=False)
    x, y = ds[0]
    assert tuple(x.shape) == (3, 16, 16)
    assert tuple(y.shape) == (1, 16, 16)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dino_seg_dataset_no_aug_matches_eval(tmp_path: Path):
    sample = _make_sample(tmp_path)
    ds_eval = DinoSegDataset([sample], image_size=16, train=False)
    ds_train = DinoSegDataset([sample], image_size=16, train=True, seed=123, p_hflip=0.0, p_vflip=0.0, p_rot90=0.0)
    x_eval, y_eval = ds_eval[0]
    x_train, y_train = ds_train[0]
    assert torch.allclose(x_train, x_eval)
    assert torch.allclose(y_train, y_eval)


def test_dino_seg_dataset_forced_hflip(tmp_path: Path):
    sample = _make_sample(tmp_path)
    ds_eval = DinoSegDataset([sample], image_size=16, train=False)
    ds_hflip = DinoSegDataset([sample], image_size=16, train=True, seed=123, p_hflip=1.0, p_vflip=0.0, p_rot90=0.0)
    x_eval, y_eval = ds_eval[0]
    x_flip, y_flip = ds_hflip[0]
    assert torch.allclose(x_flip, torch.flip(x_eval, dims=[2]))
    assert torch.allclose(y_flip, torch.flip(y_eval, dims=[2]))
