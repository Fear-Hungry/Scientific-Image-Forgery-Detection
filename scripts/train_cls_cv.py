#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.classification import BinaryForgeryClsDataset, build_classification_transform
from forgeryseg.dataset import build_supplemental_index, build_train_index
from forgeryseg.models.classifier import build_classifier, compute_pos_weight
from forgeryseg.offline import configure_cache_dirs


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


@torch.no_grad()
def _eval_classifier(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    for x, yb in loader:
        x = x.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(x).view(-1, 1)
        loss = criterion(logits, yb)
        losses.append(float(loss.item()))
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())

    if not losses:
        return {"loss": float("nan"), "acc@0.5": float("nan"), "auc": float("nan")}

    logits_np = np.concatenate(logits_all, axis=0).reshape(-1)
    y_np = np.concatenate(y_all, axis=0).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    acc = float(((probs >= 0.5).astype(np.int64) == y_np.astype(np.int64)).mean())

    out = {"loss": float(np.mean(losses)), "acc@0.5": acc}
    try:
        from sklearn.metrics import roc_auc_score

        out["auc"] = float(roc_auc_score(y_np, probs))
    except Exception:
        out["auc"] = float("nan")

    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1 = -1.0
    best_thr = 0.5
    for thr in thresholds:
        pred = (probs >= float(thr)).astype(np.int64)
        tp = int(((pred == 1) & (y_np == 1)).sum())
        fp = int(((pred == 1) & (y_np == 0)).sum())
        fn = int(((pred == 0) & (y_np == 1)).sum())
        denom = (2 * tp + fp + fn)
        f1 = float((2 * tp) / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    out["best_f1"] = float(best_f1)
    out["best_threshold"] = float(best_thr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary classifier (authentic vs forged) with CV")
    parser.add_argument("--config", default="", help="Config JSON (optional)")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (overrides config)")
    parser.add_argument("--fold", type=int, default=-1, help="Train a single fold index")
    parser.add_argument("--image-size", type=int, default=0, help="Override image size")
    parser.add_argument("--device", default="", help="Device override")
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

    backend = str(cfg.get("backend", "timm")).lower()
    model_name = str(cfg.get("model_name", "tf_efficientnet_b4_ns"))
    model_id = str(cfg.get("model_id", "cls"))
    pretrained = bool(cfg.get("pretrained", True))
    strict_pretrained = bool(cfg.get("strict_pretrained", False))
    image_size = int(cfg.get("image_size", 384))
    if int(args.image_size) > 0:
        image_size = int(args.image_size)
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 10))
    patience = int(cfg.get("patience", 3))
    lr = float(cfg.get("learning_rate", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))
    num_workers = int(cfg.get("num_workers", 2))

    samples = build_train_index(data_root)
    supplemental_samples = []
    include_supplemental = bool(cfg.get("include_supplemental", False)) or args.include_supplemental
    if include_supplemental:
        supplemental_samples = [s for s in build_supplemental_index(data_root, strict=False) if s.is_authentic is not None]

    labels = np.array([0 if s.is_authentic else 1 for s in samples], dtype=np.int64)
    folds_n = int(args.folds) if args.folds else int(cfg.get("folds", 5))
    if folds_n <= 1:
        raise ValueError("--folds must be >= 2")
    folds = list(_iter_stratified_folds(labels.tolist(), folds_n, seed))
    if args.fold >= 0:
        folds = [folds[args.fold]]

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_id = args.fold if args.fold >= 0 else fold_idx

        train_samples = [samples[i] for i in train_idx] + supplemental_samples
        val_samples = [samples[i] for i in val_idx]

        aug_kwargs = {
            "brightness": float(cfg.get("brightness", 0.2)),
            "contrast": float(cfg.get("contrast", 0.2)),
            "grayscale_prob": float(cfg.get("grayscale_prob", 0.1)),
            "blur_prob": float(cfg.get("blur_prob", 0.1)),
            "cutout_prob": float(cfg.get("cutout_prob", 0.2)),
        }
        tr_transform = build_classification_transform(image_size=image_size, train=True, **aug_kwargs)
        va_transform = build_classification_transform(image_size=image_size, train=False, **aug_kwargs)
        ds_train = BinaryForgeryClsDataset(train_samples, tr_transform)
        ds_val = BinaryForgeryClsDataset(val_samples, va_transform)

        dl_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
            drop_last=True,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
            drop_last=False,
        )

    build_kwargs: dict[str, object] = {}
    if backend in {"dinov2", "hf"}:
        build_kwargs = {
            "hf_model_id": cfg.get("hf_model_id", model_name),
            "hf_cache_dir": cfg.get("hf_cache_dir", None),
            "hf_revision": cfg.get("hf_revision", None),
            "local_files_only": bool(cfg.get("local_files_only", False)),
            "freeze_encoder": bool(cfg.get("freeze_encoder", True)),
            "classifier_hidden": int(cfg.get("classifier_hidden", 0)),
            "classifier_dropout": float(cfg.get("classifier_dropout", 0.0)),
            "use_cls_token": bool(cfg.get("use_cls_token", True)),
            "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
            "torch_dtype": cfg.get("torch_dtype", None),
        }
    elif backend == "timm_encoder":
        build_kwargs = {
            "feature_index": int(cfg.get("feature_index", -1)),
            "pool": cfg.get("pool", "avg"),
            "classifier_hidden": int(cfg.get("classifier_hidden", 0)),
            "classifier_dropout": float(cfg.get("classifier_dropout", 0.0)),
            "freeze_encoder": bool(cfg.get("freeze_encoder", False)),
        }
        model = build_classifier(model_name=model_name, pretrained=pretrained, num_classes=1, backend=backend, **build_kwargs).to(device)

        pos_weight = torch.tensor(compute_pos_weight(labels[train_idx]), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        fold_dir = output_dir / "models_cls" / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = fold_dir / "best.pt"
        log_path = fold_dir / "train_log.csv"
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_acc@0.5",
                    "val_auc",
                    "val_best_f1",
                    "val_best_threshold",
                ],
            )
            writer.writeheader()

        best_score = -1.0
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: list[float] = []
            for x, yb in dl_train:
                x = x.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x).view(-1, 1)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val = _eval_classifier(model, dl_val, criterion, device)

            score = float(val.get("auc", float("nan")))
            if not np.isfinite(score):
                score = -float(val["loss"])

            with log_path.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_acc@0.5",
                        "val_auc",
                        "val_best_f1",
                        "val_best_threshold",
                    ],
                )
                writer.writerow(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(train_loss),
                        "val_loss": float(val["loss"]),
                        "val_acc@0.5": float(val["acc@0.5"]),
                        "val_auc": float(val.get("auc", float("nan"))),
                        "val_best_f1": float(val.get("best_f1", float("nan"))),
                        "val_best_threshold": float(val.get("best_threshold", float("nan"))),
                    }
                )

            print(f"[CLS {model_id}] fold={fold_id} epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val={val}")

            if score > best_score:
                best_score = float(score)
                best_epoch = int(epoch)
                ckpt_config = {
                    "backend": backend,
                    "model_id": model_id,
                    "model_name": model_name,
                    "image_size": int(image_size),
                    "pretrained": bool(pretrained),
                    "cls_threshold": float(val.get("best_threshold", 0.5)),
                    "fold": int(fold_id),
                    "seed": int(seed),
                }
                if backend in {"dinov2", "hf"}:
                    ckpt_config.update(
                        {
                            "hf_model_id": cfg.get("hf_model_id", model_name),
                            "hf_cache_dir": cfg.get("hf_cache_dir", None),
                            "hf_revision": cfg.get("hf_revision", None),
                            "local_files_only": bool(cfg.get("local_files_only", False)),
                            "freeze_encoder": bool(cfg.get("freeze_encoder", True)),
                            "classifier_hidden": int(cfg.get("classifier_hidden", 0)),
                            "classifier_dropout": float(cfg.get("classifier_dropout", 0.0)),
                            "use_cls_token": bool(cfg.get("use_cls_token", True)),
                            "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
                            "torch_dtype": cfg.get("torch_dtype", None),
                        }
                    )
                elif backend == "timm_encoder":
                    ckpt_config.update(
                        {
                            "feature_index": int(cfg.get("feature_index", -1)),
                            "pool": cfg.get("pool", "avg"),
                            "classifier_hidden": int(cfg.get("classifier_hidden", 0)),
                            "classifier_dropout": float(cfg.get("classifier_dropout", 0.0)),
                            "freeze_encoder": bool(cfg.get("freeze_encoder", False)),
                        }
                    )
                ckpt = {
                    "model_state": model.state_dict(),
                    "config": ckpt_config,
                    "score": float(best_score),
                }
                torch.save(ckpt, ckpt_path)
                print("[CLS] saved best ->", ckpt_path)

            if patience and best_epoch and (int(epoch) - int(best_epoch) >= int(patience)):
                print(f"[CLS {model_id}] early stopping: sem melhora por {patience} épocas (best_epoch={best_epoch}).")
                break

        print(f"[CLS {model_id}] fold={fold_id} done. best_score={best_score:.6f}")


if __name__ == "__main__":
    main()
