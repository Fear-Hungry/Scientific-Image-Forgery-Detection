from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..config import load_segmentation_config
from ..typing import Pathish, Split
from .trainer import Trainer, TrainResult

AugMode = Literal["none", "basic", "robust"]
SchedulerMode = Literal["none", "cosine", "onecycle"]


def train_dino_decoder(
    *,
    config_path: Pathish,
    data_root: Pathish,
    out_path: Pathish,
    device: str,
    split: Split = "train",
    overrides: list[str] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    num_workers: int | None = None,
    val_fraction: float | None = None,
    seed: int | None = None,
    folds: int | None = None,
    fold: int | None = None,
    aug: AugMode | None = None,
    scheduler: SchedulerMode | None = None,
    lr_min: float | None = None,
    max_lr: float | None = None,
    pct_start: float | None = None,
    patience: int | None = None,
    min_delta: float | None = None,
) -> TrainResult:
    config_path = Path(config_path)
    cfg = load_segmentation_config(config_path, overrides=overrides)

    if epochs is not None:
        cfg.train.epochs = int(epochs)
    if batch_size is not None:
        cfg.train.batch_size = int(batch_size)
    if lr is not None:
        cfg.train.lr = float(lr)
    if weight_decay is not None:
        cfg.train.weight_decay = float(weight_decay)
    if num_workers is not None:
        cfg.train.num_workers = int(num_workers)
    if val_fraction is not None:
        cfg.train.val_fraction = float(val_fraction)
    if seed is not None:
        cfg.train.seed = int(seed)
    if folds is not None:
        cfg.train.folds = int(folds)
    if fold is not None:
        cfg.train.fold = int(fold)
    if aug is not None:
        cfg.train.aug = str(aug)  # type: ignore[assignment]
    if scheduler is not None:
        cfg.train.scheduler = str(scheduler)  # type: ignore[assignment]
    if lr_min is not None:
        cfg.train.lr_min = float(lr_min)
    if max_lr is not None:
        cfg.train.max_lr = float(max_lr)
    if pct_start is not None:
        cfg.train.pct_start = float(pct_start)
    if patience is not None:
        cfg.train.patience = int(patience)
    if min_delta is not None:
        cfg.train.min_delta = float(min_delta)

    trainer = Trainer(
        config=cfg,
        data_root=data_root,
        out_path=out_path,
        device=device,
        split=split,
        path_roots=[config_path.parent, Path.cwd()],
    )
    return trainer.fit()
