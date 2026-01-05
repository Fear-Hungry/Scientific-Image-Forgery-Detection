from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..config import SegmentationExperimentConfig
from ..dataset import RecodaiDataset
from ..losses import bce_dice_loss
from ..models.dinov2_decoder import DinoV2EncoderSpec, DinoV2SegmentationModel
from ..models.dinov2_freq_fusion import DinoV2FreqFusionSegmentationModel, FreqFusionSpec
from ..models.dinov2_multiscale import DinoV2MultiScaleSegmentationModel, MultiScaleSpec
from ..paths import resolve_existing_path
from ..postprocess import PostprocessParams
from ..typing import Case, Pathish, Split
from .callbacks import Callback, CSVLoggerCallback, JSONLoggerCallback
from .eval_of1 import evaluate_of1
from .utils import apply_cutmix, fold_out_path, seed_everything, stratified_splits


@dataclass(frozen=True)
class FoldResult:
    fold: int
    checkpoint_path: Path
    best_epoch: int
    best_val_of1: float
    best_val_loss: float
    log_csv: Path | None = None
    log_json: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": int(self.fold),
            "checkpoint_path": str(self.checkpoint_path),
            "best_epoch": int(self.best_epoch),
            "best_val_of1": float(self.best_val_of1),
            "best_val_loss": float(self.best_val_loss),
            "log_csv": str(self.log_csv) if self.log_csv is not None else None,
            "log_json": str(self.log_json) if self.log_json is not None else None,
        }


@dataclass(frozen=True)
class TrainResult:
    fold_results: list[FoldResult]

    def mean_best_val_of1(self) -> float:
        xs = [fr.best_val_of1 for fr in self.fold_results]
        return float(np.mean(xs)) if xs else 0.0

    def mean_best_val_loss(self) -> float:
        xs = [fr.best_val_loss for fr in self.fold_results]
        return float(np.mean(xs)) if xs else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "folds": [fr.as_dict() for fr in self.fold_results],
            "mean_best_val_of1": float(self.mean_best_val_of1()),
            "mean_best_val_loss": float(self.mean_best_val_loss()),
        }


def _build_model(cfg: SegmentationExperimentConfig) -> torch.nn.Module:
    enc_cfg = cfg.model.encoder
    encoder_ckpt = enc_cfg.checkpoint_path
    if encoder_ckpt:
        encoder_ckpt = str(resolve_existing_path(encoder_ckpt, roots=None, search_kaggle_input=True))

    encoder = DinoV2EncoderSpec(
        model_name=enc_cfg.model_name,
        checkpoint_path=encoder_ckpt,
        pretrained=bool(enc_cfg.pretrained),
    )

    if cfg.model.freeze_encoder and encoder.checkpoint_path is None and not encoder.pretrained:
        raise ValueError(
            "freeze_encoder=true mas encoder não está pré-treinado. "
            "Defina encoder.pretrained=true (requer internet/cache) ou forneça encoder.checkpoint_path."
        )

    if cfg.model.type == "dinov2_freq_fusion":
        freq = FreqFusionSpec(**cfg.model.freq_fusion)
        return DinoV2FreqFusionSegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.model.decoder_hidden_channels),
            decoder_dropout=float(cfg.model.decoder_dropout),
            freeze_encoder=bool(cfg.model.freeze_encoder),
            freq=freq,
        )
    if cfg.model.type == "dinov2_multiscale":
        multiscale = MultiScaleSpec(**cfg.model.multiscale)
        return DinoV2MultiScaleSegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.model.decoder_hidden_channels),
            decoder_dropout=float(cfg.model.decoder_dropout),
            freeze_encoder=bool(cfg.model.freeze_encoder),
            multiscale=multiscale,
        )
    return DinoV2SegmentationModel(
        encoder,
        decoder_hidden_channels=int(cfg.model.decoder_hidden_channels),
        decoder_dropout=float(cfg.model.decoder_dropout),
        freeze_encoder=bool(cfg.model.freeze_encoder),
    )


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


class Trainer:
    """
    Segmentation trainer with optional stratified cross-validation.

    Designed to be reused from scripts and notebooks (no CLI-only logic).
    """

    def __init__(
        self,
        *,
        config: SegmentationExperimentConfig,
        data_root: Pathish,
        out_path: Pathish,
        device: str | torch.device,
        split: Split = "train",
        callbacks: Iterable[Callback] | None = None,
        path_roots: list[Path] | None = None,
        train_dataset: RecodaiDataset | None = None,
        val_dataset: RecodaiDataset | None = None,
    ) -> None:
        self.config = config
        self.data_root = Path(data_root)
        self.out_path = Path(out_path)
        self.device = torch.device(device)
        self.split: Split = split
        self.callbacks = list(callbacks) if callbacks is not None else []
        self.path_roots = path_roots
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self._post = PostprocessParams(**dataclasses.asdict(self.config.inference.postprocess))
        self._tiling = None
        if self.config.inference.tiling is not None and int(self.config.inference.tiling.tile_size) > 0:
            from ..inference import TilingParams

            self._tiling = TilingParams(
                tile_size=int(self.config.inference.tiling.tile_size),
                overlap=int(self.config.inference.tiling.overlap),
                batch_size=int(self.config.inference.tiling.batch_size),
            )

    def fit(self) -> TrainResult:
        seed_everything(int(self.config.train.seed))

        ds_train = self.train_dataset
        if ds_train is None:
            ds_train = RecodaiDataset(
                self.data_root,
                self.split,
                training=True,
                transforms=self.config_train_transforms(),
            )
        ds_val = self.val_dataset
        if ds_val is None:
            ds_val = RecodaiDataset(
                self.data_root,
                self.split,
                training=True,
                transforms=self.config_val_transforms(),
            )

        labels = np.asarray([1 if c.mask_path is not None else 0 for c in ds_train.cases], dtype=np.int64)
        splits = stratified_splits(
            labels,
            folds=int(self.config.train.folds),
            val_fraction=float(self.config.train.val_fraction),
            seed=int(self.config.train.seed),
        )

        fold_results: list[FoldResult] = []
        target_fold = int(self.config.train.fold)
        for fold_id, train_idx, val_idx in splits:
            if target_fold >= 0 and int(fold_id) != target_fold:
                continue
            fold_results.append(self._fit_fold(ds_train, ds_val, fold_id, train_idx, val_idx))

        result = TrainResult(fold_results=fold_results)
        summary = result.as_dict()
        for cb in self.callbacks:
            cb.on_train_end(summary)
        print(
            f"mean_best_val_of1={summary['mean_best_val_of1']:.6f} mean_best_val_loss={summary['mean_best_val_loss']:.6f}"
        )
        return result

    def config_train_transforms(self):
        from ..transforms import make_transforms

        return make_transforms(
            int(self.config.model.input_size),
            train=True,
            aug=self.config.train.aug,  # type: ignore[arg-type]
        )

    def config_val_transforms(self):
        from ..transforms import make_transforms

        return make_transforms(int(self.config.model.input_size), train=False, aug="none")

    def _fit_fold(
        self,
        ds_train: RecodaiDataset,
        ds_val: RecodaiDataset,
        fold_id: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> FoldResult:
        print(f"fold={fold_id}/{int(self.config.train.folds) - 1 if int(self.config.train.folds) > 1 else 0}")
        seed_everything(int(self.config.train.seed) + int(fold_id))

        train_loader = DataLoader(
            Subset(ds_train, train_idx.tolist()),
            batch_size=int(self.config.train.batch_size),
            shuffle=True,
            num_workers=int(self.config.train.num_workers),
            pin_memory=False,
        )
        val_loader = DataLoader(
            Subset(ds_val, val_idx.tolist()),
            batch_size=int(self.config.train.batch_size),
            shuffle=False,
            num_workers=int(self.config.train.num_workers),
            pin_memory=False,
        )

        if self.path_roots and self.config.model.encoder.checkpoint_path:
            self.config.model.encoder.checkpoint_path = str(
                resolve_existing_path(
                    self.config.model.encoder.checkpoint_path,
                    roots=self.path_roots,
                    search_kaggle_input=True,
                )
            )
        model = _build_model(self.config).to(self.device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.train.lr),
            weight_decay=float(self.config.train.weight_decay),
        )

        scheduler_obj = None
        step_per_batch = False
        if self.config.train.scheduler == "cosine":
            scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=int(self.config.train.epochs),
                eta_min=float(self.config.train.lr_min),
            )
        elif self.config.train.scheduler == "onecycle":
            max_lr_eff = (
                float(self.config.train.max_lr) if float(self.config.train.max_lr) > 0 else float(self.config.train.lr)
            )
            scheduler_obj = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=max_lr_eff,
                total_steps=int(self.config.train.epochs) * max(1, len(train_loader)),
                pct_start=float(self.config.train.pct_start),
                anneal_strategy="cos",
            )
            step_per_batch = True
        elif self.config.train.scheduler != "none":
            raise ValueError(f"Unknown scheduler: {self.config.train.scheduler}")

        ckpt_path = fold_out_path(self.out_path, fold=fold_id, folds=int(self.config.train.folds))
        last_ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_last{ckpt_path.suffix}")
        log_csv = fold_out_path(self.out_path.with_suffix(".csv"), fold=fold_id, folds=int(self.config.train.folds))
        log_json = fold_out_path(self.out_path.with_suffix(".json"), fold=fold_id, folds=int(self.config.train.folds))

        per_fold_callbacks: list[Callback] = [
            CSVLoggerCallback(path=log_csv),
            JSONLoggerCallback(path=log_json),
            *self.callbacks,
        ]
        for cb in per_fold_callbacks:
            cb.on_fold_start(fold=fold_id, folds=int(self.config.train.folds))

        best_val_of1 = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        bad_epochs = 0
        patience = int(self.config.train.patience)
        min_delta = float(self.config.train.min_delta)

        val_cases: list[Case] = [ds_val.cases[int(i)] for i in val_idx.tolist()]

        for epoch in range(1, int(self.config.train.epochs) + 1):
            start = time.time()
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                x = batch.image.to(self.device)
                y = batch.mask.to(self.device)
                x, y = apply_cutmix(
                    x,
                    y,
                    prob=float(self.config.train.cutmix_prob),
                    alpha=float(self.config.train.cutmix_alpha),
                )
                logits = model(x)
                loss = bce_dice_loss(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if scheduler_obj is not None and step_per_batch:
                    scheduler_obj.step()
                train_losses.append(float(loss.detach().cpu().item()))

            if scheduler_obj is not None and not step_per_batch:
                scheduler_obj.step()

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_loss = _eval_loss(model, val_loader, self.device)
            val_of1 = evaluate_of1(
                model,
                val_cases,
                device=self.device,
                input_size=int(self.config.model.input_size),
                postprocess=self._post,
                tiling=self._tiling,
                use_tta=False,
                progress=False,
            ).mean_of1

            is_best = val_of1 > (best_val_of1 + float(min_delta))
            if is_best:
                best_val_of1 = float(val_of1)
                best_val_loss = float(val_loss)
                best_epoch = int(epoch)
                bad_epochs = 0

                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": dataclasses.asdict(self.config),
                        "fold": int(fold_id),
                        "epoch": int(epoch),
                        "val_loss": float(val_loss),
                        "val_of1": float(val_of1),
                    },
                    ckpt_path,
                )
            else:
                bad_epochs += 1

            # Always save a "last" checkpoint (overwritten each epoch) so partial runs can be used.
            if bool(getattr(self.config.train, "save_last", True)):
                last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": dataclasses.asdict(self.config),
                        "fold": int(fold_id),
                        "epoch": int(epoch),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "val_of1": float(val_of1),
                        "is_best": bool(is_best),
                    },
                    last_ckpt_path,
                )

            lr = float(opt.param_groups[0]["lr"])
            elapsed = float(time.time() - start)
            metrics = {
                "fold": int(fold_id),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_of1": float(val_of1),
                "lr": lr,
                "is_best": bool(is_best),
                "bad_epochs": int(bad_epochs),
                "elapsed_sec": elapsed,
            }
            for cb in per_fold_callbacks:
                cb.on_epoch_end(metrics)

            print(
                f"fold={fold_id} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_of1={val_of1:.6f} lr={lr:.3e}"
                + (" *" if is_best else "")
            )

            if patience > 0 and bad_epochs >= patience:
                print(f"early stopping: no val_of1 improvement for {bad_epochs} epochs (patience={patience})")
                break

        fold_summary = {
            "fold": int(fold_id),
            "best_epoch": int(best_epoch),
            "best_val_of1": float(best_val_of1),
            "best_val_loss": float(best_val_loss),
            "checkpoint_path": str(ckpt_path),
        }
        for cb in per_fold_callbacks:
            cb.on_fold_end(fold_summary)

        return FoldResult(
            fold=int(fold_id),
            checkpoint_path=ckpt_path,
            best_epoch=int(best_epoch),
            best_val_of1=float(best_val_of1),
            best_val_loss=float(best_val_loss),
            log_csv=log_csv,
            log_json=log_json,
        )
