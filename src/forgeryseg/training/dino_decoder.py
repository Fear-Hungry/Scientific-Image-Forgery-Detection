from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..config import SegmentationModelConfig, load_segmentation_config
from ..dataset import RecodaiDataset
from ..losses import bce_dice_loss
from ..models.dinov2_decoder import DinoV2EncoderSpec, DinoV2SegmentationModel
from ..models.dinov2_freq_fusion import DinoV2FreqFusionSegmentationModel, FreqFusionSpec
from ..paths import resolve_existing_path
from ..typing import Pathish
from .utils import fold_out_path, seed_everything, stratified_splits

AugMode = Literal["none", "basic", "robust"]
SchedulerMode = Literal["none", "cosine", "onecycle"]

TransformFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def make_transforms(input_size: int, *, train: bool, aug: AugMode) -> TransformFn:
    import albumentations as A
    import cv2

    base = [
        A.LongestMaxSize(max_size=int(input_size), interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(
            min_height=int(input_size),
            min_width=int(input_size),
            border_mode=cv2.BORDER_REFLECT_101,
        ),
    ]

    if train:
        if aug == "none":
            pass
        elif aug in {"basic", "robust"}:
            base.append(A.HorizontalFlip(p=0.5))
            base.append(A.VerticalFlip(p=0.25))
            base.append(A.RandomBrightnessContrast(p=0.25))
        else:
            raise ValueError(f"Unknown aug mode: {aug}")

        if aug == "robust":
            base.extend(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.15,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.MotionBlur(blur_limit=7, p=1.0),
                        ],
                        p=0.2,
                    ),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                ]
            )

    aug_tf = A.Compose(base)

    def _apply(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = aug_tf(image=img, mask=mask)
        return out["image"], out["mask"]

    return _apply


@torch.no_grad()
def eval_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        x = batch.image.to(device)
        y = batch.mask.to(device)
        logits = model(x)
        loss = bce_dice_loss(logits, y)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def build_model(cfg: SegmentationModelConfig, encoder: DinoV2EncoderSpec, *, freeze_encoder: bool) -> torch.nn.Module:
    model_type = str(cfg.type)
    if model_type == "dinov2_freq_fusion":
        freq = FreqFusionSpec(**cfg.freq_fusion)
        return DinoV2FreqFusionSegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.decoder_hidden_channels),
            decoder_dropout=float(cfg.decoder_dropout),
            freeze_encoder=freeze_encoder,
            freq=freq,
        )
    return DinoV2SegmentationModel(
        encoder,
        decoder_hidden_channels=int(cfg.decoder_hidden_channels),
        decoder_dropout=float(cfg.decoder_dropout),
        freeze_encoder=freeze_encoder,
    )


def train_dino_decoder(
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
    aug: AugMode | None = None,
    scheduler: SchedulerMode | None = None,
    lr_min: float | None = None,
    max_lr: float | None = None,
    pct_start: float | None = None,
) -> list[Path]:
    config_path = Path(config_path)
    out_path = Path(out_path)
    device = torch.device(device)

    cfg = load_segmentation_config(config_path, overrides=overrides)
    input_size = int(cfg.model.input_size)

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
    aug = train_cfg.aug if aug is None else aug
    scheduler = train_cfg.scheduler if scheduler is None else scheduler
    lr_min = float(train_cfg.lr_min) if lr_min is None else float(lr_min)
    max_lr = float(train_cfg.max_lr) if max_lr is None else float(max_lr)
    pct_start = float(train_cfg.pct_start) if pct_start is None else float(pct_start)

    enc_cfg = cfg.model.encoder
    enc_ckpt = enc_cfg.checkpoint_path
    if enc_ckpt:
        enc_ckpt = str(resolve_existing_path(enc_ckpt, roots=[config_path.parent, Path.cwd()], search_kaggle_input=True))
    encoder = DinoV2EncoderSpec(
        model_name=enc_cfg.model_name,
        checkpoint_path=enc_ckpt,
        pretrained=enc_cfg.pretrained,
    )

    freeze_encoder = bool(cfg.model.freeze_encoder)
    if freeze_encoder and encoder.checkpoint_path is None and not encoder.pretrained:
        raise ValueError(
            "freeze_encoder=true mas encoder não está pré-treinado. "
            "Defina encoder.pretrained=true (requer internet/cache) ou forneça encoder.checkpoint_path."
        )

    seed_everything(int(seed))
    ds_train = RecodaiDataset(
        data_root,
        "train",
        include_authentic=True,
        include_forged=True,
        transforms=make_transforms(input_size, train=True, aug=aug),
    )
    ds_val = RecodaiDataset(
        data_root,
        "train",
        include_authentic=True,
        include_forged=True,
        transforms=make_transforms(input_size, train=False, aug="none"),
    )

    labels = np.asarray([1 if c.mask_path is not None else 0 for c in ds_train.cases], dtype=np.int64)
    splits = stratified_splits(labels, folds=int(folds), val_fraction=float(val_fraction), seed=int(seed))

    saved: list[Path] = []
    target_fold = int(fold)
    for fold_id, train_idx, val_idx in splits:
        if target_fold >= 0 and int(fold_id) != target_fold:
            continue

        print(f"fold={fold_id}/{int(folds) - 1 if int(folds) > 1 else 0}")
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

        model = build_model(cfg.model, encoder, freeze_encoder=freeze_encoder).to(device)
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
        best_path = fold_out_path(out_path, fold=int(fold_id), folds=int(folds))
        for epoch in range(1, int(epochs) + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Train fold {fold_id} epoch {epoch}/{epochs}")
            for batch in pbar:
                x = batch.image.to(device)
                y = batch.mask.to(device)
                logits = model(x)
                loss = bce_dice_loss(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if scheduler_obj is not None and step_per_batch:
                    scheduler_obj.step()
                pbar.set_postfix(loss=float(loss.detach().cpu().item()), lr=float(opt.param_groups[0]["lr"]))

            if scheduler_obj is not None and not step_per_batch:
                scheduler_obj.step()

            val_loss = eval_loss(model, val_loader, device)
            print(f"fold={fold_id} epoch={epoch} val_loss={val_loss:.6f}")

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
