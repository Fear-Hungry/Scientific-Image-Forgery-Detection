from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

import forgeryseg.training.trainer as trainer_mod
from forgeryseg.config import (
    EncoderConfig,
    PostprocessConfig,
    SegmentationExperimentConfig,
    SegmentationInferenceConfig,
    SegmentationModelConfig,
    SegmentationTrainConfig,
    TTAConfig,
)
from forgeryseg.dataset import RecodaiDataset
from forgeryseg.training.trainer import Trainer


def _write_png(path: Path, *, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


class TinySeg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_trainer_fit_saves_checkpoint(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "recodai"
    (data_root / "train_images" / "authentic").mkdir(parents=True, exist_ok=True)
    (data_root / "train_images" / "forged").mkdir(parents=True, exist_ok=True)
    (data_root / "train_masks").mkdir(parents=True, exist_ok=True)

    _write_png(data_root / "train_images" / "authentic" / "a1.png")
    _write_png(data_root / "train_images" / "authentic" / "a2.png")
    _write_png(data_root / "train_images" / "forged" / "f1.png")
    _write_png(data_root / "train_images" / "forged" / "f2.png")

    m = np.zeros((8, 8), dtype=np.uint8)
    m[0:2, 0:2] = 1
    np.save(data_root / "train_masks" / "f1.npy", np.stack([m], axis=0))
    np.save(data_root / "train_masks" / "f2.npy", np.stack([m], axis=0))

    train_ds = RecodaiDataset(data_root, "train", training=True, transforms=None)
    val_ds = RecodaiDataset(data_root, "train", training=True, transforms=None)

    cfg = SegmentationExperimentConfig(
        name="unit_trainer",
        model=SegmentationModelConfig(
            type="dinov2",
            input_size=8,
            encoder=EncoderConfig(model_name="vit_small_patch14_dinov2", pretrained=False),
            checkpoint=None,
            decoder_hidden_channels=8,
            decoder_dropout=0.0,
            freeze_encoder=False,
        ),
        inference=SegmentationInferenceConfig(
            batch_size=1,
            tta=TTAConfig(zoom_scale=0.9, weights=[1.0]),
            tiling=None,
            postprocess=PostprocessConfig(prob_threshold=0.5),
            fft_gate=None,
        ),
        train=SegmentationTrainConfig(
            epochs=1,
            batch_size=2,
            lr=1e-3,
            weight_decay=0.0,
            num_workers=0,
            val_fraction=0.5,
            seed=123,
            folds=1,
            fold=-1,
            aug="none",
            scheduler="none",
            patience=0,
            min_delta=0.0,
        ),
    )

    monkeypatch.setattr(trainer_mod, "_build_model", lambda _cfg: TinySeg())

    out_path = tmp_path / "outputs" / "models" / "tiny.pth"
    trainer = Trainer(
        config=cfg,
        data_root=data_root,
        out_path=out_path,
        device="cpu",
        split="train",
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    res = trainer.fit()
    assert res.fold_results
    assert out_path.exists()
