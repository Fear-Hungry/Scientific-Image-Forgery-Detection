from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from forgeryseg.dataset_fft import RecodaiFFTDataset
from forgeryseg.frequency import FFTParams
from forgeryseg.models.fft_classifier import FFTClassifier


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _augment(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        x = torch.flip(x, dims=(-1,))
    if random.random() < 0.25:
        x = torch.flip(x, dims=(-2,))
    return x


@torch.no_grad()
def _eval(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
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


def _fold_out_path(base: Path, *, fold: int, folds: int) -> Path:
    if folds <= 1:
        return base
    return base.with_name(f"{base.stem}_fold{fold}{base.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="none")
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--max-lr", type=float, default=0.0)
    ap.add_argument("--pct-start", type=float, default=0.1)
    args = ap.parse_args()

    folds = int(args.folds)
    if folds < 1:
        raise ValueError("--folds must be >= 1")
    target_fold = int(args.fold)

    cfg = json.loads(args.config.read_text())
    fft_params = FFTParams(**cfg.get("fft", {}))

    model_cfg = cfg.get("model", {})

    val_fraction = float(args.val_fraction)
    if folds == 1 and not (0.0 < val_fraction < 1.0):
        raise ValueError("--val-fraction must be between 0 and 1 (quando --folds=1)")

    _seed_everything(int(args.seed))
    ds_train = RecodaiFFTDataset(
        args.data_root,
        "train",
        fft_params=fft_params,
        include_authentic=True,
        include_forged=True,
        image_transforms=None if args.no_aug else _augment,
    )
    ds_val = RecodaiFFTDataset(
        args.data_root,
        "train",
        fft_params=fft_params,
        include_authentic=True,
        include_forged=True,
        image_transforms=None,
    )

    labels = np.asarray([1 if c.mask_path is not None else 0 for c in ds_train.cases], dtype=np.int64)
    device = torch.device(args.device)

    if folds == 1:
        try:
            from sklearn.model_selection import StratifiedShuffleSplit

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=int(args.seed))
            train_idx_np, val_idx_np = next(splitter.split(np.zeros(len(labels)), labels))
            splits = [(0, train_idx_np, val_idx_np)]
        except Exception as e:
            print(f"[warn] stratified split failed ({type(e).__name__}: {e}); falling back to random split")
            indices = np.random.permutation(len(labels))
            n_val = int(round(len(indices) * val_fraction))
            splits = [(0, indices[n_val:], indices[:n_val])]
    else:
        try:
            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(args.seed))
            splits = [(i, tr, va) for i, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels))]
        except Exception as e:
            raise RuntimeError(f"StratifiedKFold unavailable ({type(e).__name__}: {e})") from e

    for fold, train_idx_np, val_idx_np in splits:
        if target_fold >= 0 and fold != target_fold:
            continue

        print(f"fold={fold}/{folds - 1 if folds > 1 else 0}")
        _seed_everything(int(args.seed) + int(fold))

        train_ds = Subset(ds_train, train_idx_np.tolist())
        val_ds = Subset(ds_val, val_idx_np.tolist())

        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=False,
        )

        model = FFTClassifier(
            backbone=model_cfg.get("backbone", "resnet18"),
            in_chans=1,
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        scheduler = None
        step_per_batch = False
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=int(args.epochs), eta_min=float(args.lr_min)
            )
        elif args.scheduler == "onecycle":
            max_lr = float(args.max_lr) if float(args.max_lr) > 0 else float(args.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=max_lr,
                total_steps=int(args.epochs) * max(1, len(train_loader)),
                pct_start=float(args.pct_start),
                anneal_strategy="cos",
            )
            step_per_batch = True
        elif args.scheduler != "none":
            raise ValueError(f"Unknown scheduler: {args.scheduler}")

        best_val = float("inf")
        out_path = _fold_out_path(args.out, fold=fold, folds=folds)
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Train fold {fold} epoch {epoch}/{args.epochs}")
            for batch in pbar:
                x = batch.x.to(device)
                y = batch.y.to(device)

                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if scheduler is not None and step_per_batch:
                    scheduler.step()
                pbar.set_postfix(loss=float(loss.detach().cpu().item()), lr=float(opt.param_groups[0]["lr"]))

            if scheduler is not None and not step_per_batch:
                scheduler.step()

            val_loss, val_acc = _eval(model, val_loader, device)
            print(f"fold={fold} epoch={epoch} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "config": cfg, "fold": fold, "epoch": epoch, "val_loss": val_loss},
                    out_path,
                )
                print(f"saved best checkpoint to {out_path}")


if __name__ == "__main__":
    main()
