from __future__ import annotations

from pathlib import Path

from forgeryseg.config import load_fft_classifier_config, load_segmentation_config


def test_load_all_segmentation_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "configs"
    paths = sorted(cfg_dir.glob("dino_*.json"))
    assert paths, "expected segmentation configs under configs/dino_*.json"

    for p in paths:
        cfg = load_segmentation_config(p)
        assert cfg.name
        assert cfg.model.input_size > 0
        assert 0.0 <= cfg.inference.postprocess.prob_threshold <= 1.0


def test_load_all_fft_classifier_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "configs"
    paths = sorted(cfg_dir.glob("fft_classifier_*.json"))
    assert paths, "expected FFT classifier configs under configs/fft_classifier_*.json"

    for p in paths:
        cfg = load_fft_classifier_config(p)
        assert cfg.name
        assert cfg.fft.input_size > 0
        assert cfg.model.backbone


def test_config_overrides_work() -> None:
    root = Path(__file__).resolve().parents[1]
    p = root / "configs" / "dino_v3_518_r69.json"
    cfg = load_segmentation_config(
        p,
        overrides=[
            "model.input_size=123",
            "inference.postprocess.min_area=321",
            "train.epochs=7",
        ],
    )
    assert cfg.model.input_size == 123
    assert cfg.inference.postprocess.min_area == 321
    assert cfg.train.epochs == 7
