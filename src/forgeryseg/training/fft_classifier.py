from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..config import load_fft_classifier_config
from ..dataset_fft import RecodaiFFTDataset
from ..frequency import FFTParams
from ..models.fft_classifier import FFTClassifier
from ..typing import Pathish
from .utils import fold_out_path, seed_everything, stratified_splits

SchedulerMode = Literal["none", "cosine", "onecycle"]


def augment(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        x = torch.flip(x, dims=(-1,))
    if random.random() < 0.25:
        x = torch.flip(x, dims=(-2,))
    return x


@torch.no_grad()
def eval_classifier(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses: list[float] = []
    correct = 0
    total = 0
    for batch in loader:
        x = batch.x.to(device)
        y = batch.y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.detach().cpu().item()))

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += int((preds == y).sum().detach().cpu().item())
        total += int(y.numel())

    return (float(np.mean(losses)) if losses else 0.0), (correct / total if total else 0.0)


def train_fft_classifier(
    *,
    config_path: Pathish,
    data_root: Pathish,
    out_path: Pathish,
    device: str | torch.device,
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
    no_aug: bool | None = None,
    scheduler: SchedulerMode | None = None,
    lr_min: float | None = None,
    max_lr: float | None = None,
    pct_start: float | None = None,
) -> list[Path]:
    config_path = Path(config_path)
    out_path = Path(out_path)
    device = torch.device(device)

    cfg = load_fft_classifier_config(config_path, overrides=overrides)
    train_cfg = cfg.train

    epochs = int(train_cfg.epochs) if epochs is None else int(epochs)
    batch_size = int(train_cfg.batch_size) if batch_size is None else int(batch_size)
    lr = float(train_cfg.lr) if lr is None else float(lr)
    weight_decay = float(train_cfg.weight_decay) if weight_decay is None else float(weight_decay)
    num_workers = int(train_cfg.num_workers) if num_workers is None else int(num_workers)
    val_fraction = float(train_cfg.val_fraction) if val_fraction is None else float(val_fraction)
    seed = int(train_cfg.seed) if seed is None else int(seed)
    folds = int(train_cfg.folds) if folds is None else int(folds)
    fold = int(train_cfg.fold) if fold is None else int(fold)
    no_aug = bool(train_cfg.no_aug) if no_aug is None else bool(no_aug)
    scheduler = train_cfg.scheduler if scheduler is None else scheduler
    lr_min = float(train_cfg.lr_min) if lr_min is None else float(lr_min)
    max_lr = float(train_cfg.max_lr) if max_lr is None else float(max_lr)
    pct_start = float(train_cfg.pct_start) if pct_start is None else float(pct_start)

    folds = int(folds)
    if folds < 1:
        raise ValueError("folds must be >= 1")

    if folds == 1 and not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be between 0 and 1 (quando folds=1)")

    fft_percentiles = tuple(float(x) for x in cfg.fft.normalize_percentiles)
    if len(fft_percentiles) != 2:
        raise ValueError("fft.normalize_percentiles must have 2 values")
    fft_params = FFTParams(
        mode=cfg.fft.mode,  # type: ignore[arg-type]
        input_size=int(cfg.fft.input_size),
        hp_radius_fraction=float(cfg.fft.hp_radius_fraction),
        normalize_percentiles=fft_percentiles,  # type: ignore[arg-type]
    )

    backbone = cfg.model.backbone
    dropout = float(cfg.model.dropout)

    seed_everything(int(seed))
    ds_train = RecodaiFFTDataset(
        data_root,
        "train",
        fft_params=fft_params,
        include_authentic=True,
        include_forged=True,
        image_transforms=None if no_aug else augment,
    )
    ds_val = RecodaiFFTDataset(
        data_root,
        "train",
        fft_params=fft_params,
        include_authentic=True,
        include_forged=True,
        image_transforms=None,
    )

    labels = np.asarray([1 if c.mask_path is not None else 0 for c in ds_train.cases], dtype=np.int64)
    splits = stratified_splits(labels, folds=folds, val_fraction=float(val_fraction), seed=int(seed))

    saved: list[Path] = []
    target_fold = int(fold)
    for fold_id, train_idx, val_idx in splits:
        if target_fold >= 0 and int(fold_id) != target_fold:
            continue

        print(f"fold={fold_id}/{folds - 1 if folds > 1 else 0}")
        seed_everything(int(seed) + int(fold_id))

        train_loader = DataLoader(
            Subset(ds_train, train_idx.tolist()),
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=False,
        )
        val_loader = DataLoader(
            Subset(ds_val, val_idx.tolist()),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=False,
        )

        model = FFTClassifier(backbone=backbone, in_chans=1, dropout=dropout).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        scheduler_obj = None
        step_per_batch = False
        if scheduler == "cosine":
            scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=int(epochs), eta_min=float(lr_min)
            )
        elif scheduler == "onecycle":
            max_lr_eff = float(max_lr) if float(max_lr) > 0 else float(lr)
            scheduler_obj = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=max_lr_eff,
                total_steps=int(epochs) * max(1, len(train_loader)),
                pct_start=float(pct_start),
                anneal_strategy="cos",
            )
            step_per_batch = True
        elif scheduler != "none":
            raise ValueError(f"Unknown scheduler: {scheduler}")

        best_val = float("inf")
        best_path = fold_out_path(out_path, fold=int(fold_id), folds=folds)
        for epoch in range(1, int(epochs) + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Train fold {fold_id} epoch {epoch}/{epochs}")
            for batch in pbar:
                x = batch.x.to(device)
                y = batch.y.to(device)

                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if scheduler_obj is not None and step_per_batch:
                    scheduler_obj.step()
                pbar.set_postfix(loss=float(loss.detach().cpu().item()), lr=float(opt.param_groups[0]["lr"]))

            if scheduler_obj is not None and not step_per_batch:
                scheduler_obj.step()

            val_loss, val_acc = eval_classifier(model, val_loader, device)
            print(f"fold={fold_id} epoch={epoch} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": dataclasses.asdict(cfg),
                        "config_path": str(config_path),
                        "fold": fold_id,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    best_path,
                )
                print(f"saved best checkpoint to {best_path}")

        saved.append(best_path)

    return saved
