from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import random
from pathlib import Path

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
    ap.add_argument("--no-aug", action="store_true")
    args = ap.parse_args()

    _seed_everything(int(args.seed))

    cfg = json.loads(args.config.read_text())
    fft_params = FFTParams(**cfg.get("fft", {}))

    model_cfg = cfg.get("model", {})
    model = FFTClassifier(
        backbone=model_cfg.get("backbone", "resnet18"),
        in_chans=1,
        dropout=float(model_cfg.get("dropout", 0.0)),
    )

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

    val_fraction = float(args.val_fraction)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("--val-fraction must be between 0 and 1")

    labels = np.asarray([1 if c.mask_path is not None else 0 for c in ds_train.cases], dtype=np.int64)
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=int(args.seed))
        train_idx_np, val_idx_np = next(splitter.split(np.zeros(len(labels)), labels))
        train_idx = train_idx_np.tolist()
        val_idx = val_idx_np.tolist()
    except Exception as e:
        print(f"[warn] stratified split failed ({type(e).__name__}: {e}); falling back to random split")
        indices = np.random.permutation(len(labels))
        n_val = int(round(len(indices) * val_fraction))
        val_idx = indices[:n_val].tolist()
        train_idx = indices[n_val:].tolist()

    train_ds = Subset(ds_train, train_idx)
    val_ds = Subset(ds_val, val_idx)

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

    device = torch.device(args.device)
    model = model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch.x.to(device)
            y = batch.y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        val_loss, val_acc = _eval(model, val_loader, device)
        print(f"epoch={epoch} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_loss": val_loss}, args.out)
            print(f"saved best checkpoint to {args.out}")


if __name__ == "__main__":
    main()
