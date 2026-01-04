from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

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
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
    folds: int = 1,
    fold: int = -1,
    no_aug: bool = False,
    scheduler: SchedulerMode = "none",
    lr_min: float = 1e-6,
    max_lr: float = 0.0,
    pct_start: float = 0.1,
) -> list[Path]:
    config_path = Path(config_path)
    out_path = Path(out_path)
    device = torch.device(device)

    folds = int(folds)
    if folds < 1:
        raise ValueError("folds must be >= 1")

    if folds == 1 and not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be between 0 and 1 (quando folds=1)")

    cfg = json.loads(config_path.read_text())
    fft_params = FFTParams(**cfg.get("fft", {}))

    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "resnet18")
    dropout = float(model_cfg.get("dropout", 0.0))

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
                    {"model": model.state_dict(), "config": cfg, "fold": fold_id, "epoch": epoch, "val_loss": val_loss},
                    best_path,
                )
                print(f"saved best checkpoint to {best_path}")

        saved.append(best_path)

    return saved

