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

from forgeryseg.dataset import RecodaiDataset
from forgeryseg.losses import bce_dice_loss
from forgeryseg.models.dinov2_decoder import DinoV2EncoderSpec, DinoV2SegmentationModel
from forgeryseg.models.dinov2_freq_fusion import DinoV2FreqFusionSegmentationModel, FreqFusionSpec


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_transforms(input_size: int, *, train: bool):
    import albumentations as A
    import cv2

    base = [
        A.LongestMaxSize(max_size=input_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(
            min_height=input_size,
            min_width=input_size,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
    ]
    if train:
        base.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.25),
            ]
        )
    aug = A.Compose(base)

    def _apply(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = aug(image=img, mask=mask)
        return out["image"], out["mask"]

    return _apply


@torch.no_grad()
def _eval_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        x = batch.image.to(device)
        y = batch.mask.to(device)
        logits = model(x)
        loss = bce_dice_loss(logits, y)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _seed_everything(int(args.seed))

    cfg = json.loads(args.config.read_text())
    input_size = int(cfg["input_size"])

    enc_cfg = cfg.get("encoder", {})
    encoder = DinoV2EncoderSpec(
        model_name=enc_cfg.get("model_name", "vit_base_patch14_dinov2"),
        checkpoint_path=enc_cfg.get("checkpoint_path"),
        pretrained=bool(enc_cfg.get("pretrained", False)),
    )

    model_type = str(cfg.get("model_type", "dinov2"))
    freeze_encoder = bool(cfg.get("freeze_encoder", True))
    if freeze_encoder and encoder.checkpoint_path is None and not encoder.pretrained:
        raise ValueError(
            "freeze_encoder=true mas encoder não está pré-treinado. "
            "Defina encoder.pretrained=true (requer internet/cache) ou forneça encoder.checkpoint_path."
        )
    if model_type == "dinov2_freq_fusion":
        freq = FreqFusionSpec(**cfg.get("freq_fusion", {}))
        model = DinoV2FreqFusionSegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.get("decoder_hidden_channels", 256)),
            decoder_dropout=float(cfg.get("decoder_dropout", 0.0)),
            freeze_encoder=freeze_encoder,
            freq=freq,
        )
    else:
        model = DinoV2SegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.get("decoder_hidden_channels", 256)),
            decoder_dropout=float(cfg.get("decoder_dropout", 0.0)),
            freeze_encoder=freeze_encoder,
        )

    ds_train = RecodaiDataset(
        args.data_root,
        "train",
        include_authentic=True,
        include_forged=True,
        transforms=_make_transforms(input_size, train=True),
    )
    ds_val = RecodaiDataset(
        args.data_root,
        "train",
        include_authentic=True,
        include_forged=True,
        transforms=_make_transforms(input_size, train=False),
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
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch.image.to(device)
            y = batch.mask.to(device)
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        val_loss = _eval_loss(model, val_loader, device)
        print(f"epoch={epoch} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_loss": val_loss}, args.out)
            print(f"saved best checkpoint to {args.out}")


if __name__ == "__main__":
    main()
