#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.augment import get_train_augment, get_val_augment
from forgeryseg.dataset import PatchDataset, build_supplemental_index, build_train_index
from forgeryseg.losses import BCEDiceLoss, BCETverskyLoss
from forgeryseg.models import builders, dinov2
from forgeryseg.models.hybrid import build_hybrid_model
from forgeryseg.offline import configure_cache_dirs
from forgeryseg.train import train_one_epoch, validate


def _iter_stratified_folds(y: List[int], n_splits: int, seed: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in splitter.split(np.zeros(len(y)), y):
            yield train_idx, val_idx
        return
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    indices = np.arange(len(y))
    folds = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        label_indices = indices[y == label]
        rng.shuffle(label_indices)
        for i, idx in enumerate(label_indices):
            folds[i % n_splits].append(idx)

    for fold_idx in range(n_splits):
        val_idx = np.array(sorted(folds[fold_idx]))
        train_idx = np.array(sorted([i for i in indices if i not in set(val_idx)]))
        yield train_idx, val_idx


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("r") as f:
        return json.load(f)


def _build_criterion(name: str, cfg: dict) -> nn.Module:
    name = name.lower()
    if name == "bce_dice":
        return BCEDiceLoss(dice_weight=float(cfg.get("dice_weight", 1.0)))
    if name == "bce_tversky":
        return BCETverskyLoss(
            alpha=float(cfg.get("tversky_alpha", 0.7)),
            beta=float(cfg.get("tversky_beta", 0.3)),
            tversky_weight=float(cfg.get("tversky_weight", 1.0)),
        )
    raise ValueError(f"Unknown loss {name!r}")


def _build_seg_model(cfg: dict) -> nn.Module:
    backend = str(cfg.get("backend", "smp")).lower()
    classes = int(cfg.get("classes", 1))
    in_channels = int(cfg.get("in_channels", 3))

    if backend == "smp":
        arch = str(cfg.get("arch", "unetplusplus")).lower()
        encoder_name = str(cfg.get("encoder_name", "tu-convnext_small"))
        encoder_weights = cfg.get("encoder_weights", "imagenet")
        if encoder_weights == "":
            encoder_weights = None
        strict_weights = bool(cfg.get("strict_weights", False))

        if arch == "unet":
            return builders.build_unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                strict_weights=strict_weights,
            )
        if arch in {"unetplusplus", "unetpp"}:
            return builders.build_unetplusplus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                strict_weights=strict_weights,
            )
        if arch in {"deeplabv3plus", "deeplabv3+", "deeplabv3p"}:
            return builders.build_deeplabv3plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                strict_weights=strict_weights,
            )
        if arch in {"segformer", "mit"}:
            return builders.build_segformer(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                strict_weights=strict_weights,
            )
        raise ValueError(f"Unknown SMP arch {arch!r}")

    if backend == "hybrid" or str(cfg.get("arch", "")).lower() == "hybrid":
        return build_hybrid_model(
            encoder_name=str(cfg.get("encoder_name", "efficientnet-b4")),
            encoder_weights=cfg.get("encoder_weights", None),
            in_channels=in_channels,
            classes=classes,
            dino_model_id=str(cfg.get("dino_model_id", "facebook/dinov2-base")),
            decoder_channels=cfg.get("decoder_channels", (256, 128, 64, 32, 16)),
            freeze_dino=bool(cfg.get("freeze_dino", True)),
            dino_pretrained=bool(cfg.get("dino_pretrained", True)),
            dino_cache_dir=cfg.get("dino_cache_dir") or cfg.get("cache_dir"),
            dino_local_files_only=bool(cfg.get("local_files_only", False)),
            dino_revision=cfg.get("dino_revision") or cfg.get("revision"),
            dino_trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            dino_torch_dtype=cfg.get("torch_dtype", None),
            attention_type=cfg.get("attention_type", "scse"),
        )

    if backend in {"dinov2", "hf"}:
        model_id = str(cfg.get("hf_model_id", cfg.get("encoder_name", "metaresearch/dinov2")))
        return dinov2.build_dinov2_segmenter(
            model_id=model_id,
            decoder_channels=cfg.get("decoder_channels", (256, 128, 64)),
            decoder_dropout=float(cfg.get("decoder_dropout", 0.0)),
            pretrained=bool(cfg.get("pretrained", True)),
            freeze_encoder=bool(cfg.get("freeze_encoder", True)),
            cache_dir=cfg.get("hf_cache_dir") or cfg.get("cache_dir"),
            local_files_only=bool(cfg.get("local_files_only", False)),
            revision=cfg.get("hf_revision") or cfg.get("revision"),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            torch_dtype=cfg.get("torch_dtype", None),
        )

    raise ValueError(f"Unknown segmentation backend {backend!r}")


def _write_split(path: Path, train_idx: List[int], val_idx: List[int], samples) -> None:
    # `json.dump` não serializa `numpy.int64` (Kaggle/NumPy), então forçamos `int`.
    train_idx = [int(i) for i in train_idx]
    val_idx = [int(i) for i in val_idx]
    payload = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_cases": [samples[i].rel_path.as_posix() for i in train_idx],
        "val_cases": [samples[i].rel_path.as_posix() for i in val_idx],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SMP segmentation models with CV (Unet++ / DeepLabV3+ / SegFormer)")
    parser.add_argument("--config", required=True, help="Config JSON")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (overrides config)")
    parser.add_argument("--fold", type=int, default=-1, help="Train a single fold index")
    parser.add_argument("--patch-size", type=int, default=0, help="Override patch size")
    parser.add_argument("--device", default="", help="Device override (e.g. cuda:0)")
    parser.add_argument("--include-supplemental", action="store_true", help="Include supplemental_images in training")
    parser.add_argument("--cache-root", default="", help="Cache root for offline pretrained weights (sets TORCH_HOME/HF_HOME)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    data_root = Path(cfg.get("data_root", args.data_root))
    output_dir = Path(cfg.get("output_dir", args.output_dir))

    cache_root = cfg.get("cache_root", args.cache_root) or None
    configure_cache_dirs(cache_root)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("use_amp", True)) and str(device).startswith("cuda")

    model_id = str(cfg.get("model_id", Path(args.config).stem))
    cfg_folds = int(cfg.get("folds", args.folds))
    folds_n = int(args.folds) if args.folds else cfg_folds
    if folds_n <= 1:
        raise ValueError("--folds must be >= 2")

    samples = build_train_index(data_root)
    supplemental_samples = []
    include_supplemental = bool(cfg.get("include_supplemental", False)) or args.include_supplemental
    if include_supplemental:
        supplemental_samples = build_supplemental_index(data_root, strict=False)

    y = [0 if s.is_authentic else 1 for s in samples]
    folds = list(_iter_stratified_folds(y, folds_n, seed))
    if args.fold >= 0:
        folds = [folds[args.fold]]

    patch_size = int(cfg.get("patch_size", 512))
    if int(args.patch_size) > 0:
        patch_size = int(args.patch_size)
        cfg["patch_size"] = int(patch_size)
    batch_size = int(cfg.get("batch_size", 8))
    epochs = int(cfg.get("epochs", 10))
    patience = int(cfg.get("patience", 3))
    lr = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    pos_prob = float(cfg.get("positive_prob", 0.7))
    min_pos_pixels = int(cfg.get("min_pos_pixels", 1))
    max_tries = int(cfg.get("max_tries", 10))
    pos_sample_weight = float(cfg.get("pos_sample_weight", 2.0))
    num_workers = int(cfg.get("num_workers", 4))
    use_freq_channels = bool(cfg.get("use_freq_channels", False))
    in_channels = int(cfg.get("in_channels", 4 if use_freq_channels else 3))
    cfg["in_channels"] = int(in_channels)
    cfg["use_freq_channels"] = bool(use_freq_channels)
    if int(in_channels) != 3 and cfg.get("encoder_weights", None):
        print("[WARN] in_channels != 3; desativando encoder_weights para evitar mismatch.")
        cfg["encoder_weights"] = None

    copy_move_scale_range = cfg.get("copy_move_scale_range", (0.9, 1.1))
    if isinstance(copy_move_scale_range, (list, tuple)) and len(copy_move_scale_range) == 2:
        copy_move_scale_range = (float(copy_move_scale_range[0]), float(copy_move_scale_range[1]))
    else:
        copy_move_scale_range = (0.9, 1.1)

    train_aug = get_train_augment(
        patch_size=patch_size,
        copy_move_prob=float(cfg.get("copy_move_prob", 0.0)),
        copy_move_min_area_frac=float(cfg.get("copy_move_min_area_frac", 0.05)),
        copy_move_max_area_frac=float(cfg.get("copy_move_max_area_frac", 0.20)),
        copy_move_rotation_limit=float(cfg.get("copy_move_rotation_limit", 15.0)),
        copy_move_scale_range=copy_move_scale_range,
        grayscale_prob=float(cfg.get("grayscale_prob", 0.1)),
        cutout_prob=float(cfg.get("cutout_prob", 0.2)),
    )
    val_aug = get_val_augment()

    loss_name = str(cfg.get("loss", "bce_tversky"))
    criterion = _build_criterion(loss_name, cfg)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_id = args.fold if args.fold >= 0 else fold_idx
        train_samples = [samples[i] for i in train_idx] + supplemental_samples
        val_samples = [samples[i] for i in val_idx]

        train_ds = PatchDataset(
            train_samples,
            patch_size=patch_size,
            train=True,
            augment=train_aug,
            positive_prob=pos_prob,
            min_pos_pixels=min_pos_pixels,
            max_tries=max_tries,
            seed=seed,
            use_freq_channels=use_freq_channels,
        )
        val_ds = PatchDataset(
            val_samples,
            patch_size=patch_size,
            train=False,
            augment=val_aug,
            seed=seed,
            use_freq_channels=use_freq_channels,
        )

        weights = [pos_sample_weight if (s.is_authentic is False) else 1.0 for s in train_samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
        )

        model_cfg = dict(cfg)
        model_cfg["classes"] = 1
        model_cfg.setdefault("backend", "smp")
        backend = str(model_cfg.get("backend", "smp")).lower()
        if backend in {"dinov2", "hf"} and "arch" not in model_cfg:
            model_cfg["arch"] = "dinov2"

        model = _build_seg_model(model_cfg).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        fold_dir = output_dir / "models_seg" / model_id / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = fold_dir / "best.pt"
        log_path = fold_dir / "train_log.csv"
        split_path = output_dir / "splits_seg" / model_id / f"fold_{fold_id}.json"
        _write_split(split_path, list(train_idx), list(val_idx), samples)

        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_dice"])
            writer.writeheader()

        best_dice = -1.0
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(
                model, train_loader, criterion, optimizer, device, use_amp=use_amp, progress=True, desc=f"train {model_id}"
            )
            val_stats, val_dice = validate(
                model, val_loader, criterion, device, progress=True, desc=f"val {model_id}"
            )

            with log_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_dice"])
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": float(tr.loss),
                        "val_loss": float(val_stats.loss),
                        "val_dice": float(val_dice),
                    }
                )

            print(
                f"[SEG {model_id}] fold={fold_id} epoch {epoch:02d}/{epochs} | "
                f"train_loss={tr.loss:.4f} | val_loss={val_stats.loss:.4f} | dice@0.5={val_dice:.4f}"
            )

            if float(val_dice) > best_dice:
                best_dice = float(val_dice)
                best_epoch = int(epoch)
                ckpt_config = dict(model_cfg)
                ckpt_config.update(
                    {
                        "backend": backend,
                        "classes": 1,
                        "model_id": model_id,
                        "patch_size": int(patch_size),
                        "in_channels": int(in_channels),
                        "fold": int(fold_id),
                        "seed": int(seed),
                    }
                )
                if backend == "smp":
                    ckpt_config.update(
                        {
                            "arch": str(model_cfg.get("arch", "unetplusplus")),
                            "encoder_name": str(model_cfg.get("encoder_name", "")),
                            "encoder_weights": model_cfg.get("encoder_weights", None),
                        }
                    )
                ckpt = {
                    "model_state": model.state_dict(),
                    "config": ckpt_config,
                    "score": float(best_dice),
                }
                torch.save(ckpt, ckpt_path)
                print("[SEG] saved best ->", ckpt_path)

            if patience and best_epoch and (int(epoch) - int(best_epoch) >= int(patience)):
                print(f"[SEG {model_id}] early stopping: sem melhora por {patience} épocas (best_epoch={best_epoch}).")
                break

        print(f"[SEG {model_id}] fold={fold_id} done. best_dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
