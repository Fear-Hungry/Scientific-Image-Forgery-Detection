from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .dataset import DinoSegDataset, build_train_index, load_image, load_mask_instances
from .inference import apply_tta, undo_tta
from .metric import score_image
from .models import dinov2
from .postprocess import adaptive_threshold_value, dino_prob_to_instances, prob_to_instances


DEFAULT_TTA = ("none", "hflip", "vflip", "rot90", "rot180", "rot270")


@dataclass(frozen=True)
class DinoTrainResult:
    best_path: Path
    best_dice: float
    best_epoch: int


@dataclass(frozen=True)
class DinoOOFResult:
    run_dir: Path
    mean_score: float | None
    fold_scores: dict[int, float]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return float(dice.mean().item())


def _predict_prob(model, img_rgb: np.ndarray, image_size: int, device: str) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV (cv2) is required for DINO inference.") from exc

    orig_h, orig_w = img_rgb.shape[:2]
    img_rs = cv2.resize(img_rgb, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img_rs).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    if prob.shape != (orig_h, orig_w):
        prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return prob


def _tta_predict(model, img_rgb: np.ndarray, image_size: int, device: str, tta_modes: tuple[str, ...]) -> np.ndarray:
    if not tta_modes:
        return _predict_prob(model, img_rgb, image_size, device)
    preds = []
    for mode in tta_modes:
        img_t = apply_tta(img_rgb, mode)
        prob_t = _predict_prob(model, img_t, image_size, device)
        prob = undo_tta(prob_t, mode)
        preds.append(prob)
    return np.mean(preds, axis=0).astype(np.float32)


def _write_scores_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train_dino_head(
    *,
    data_root: str | Path = "data/recodai",
    output_dir: str | Path = "outputs/models_dino",
    folds: int = 5,
    fold: int = 0,
    seed: int = 42,
    dino_path: str = "facebook/dinov2-base",
    image_size: int = 512,
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    decoder_dropout: float = 0.0,
    patience: int = 3,
    device: str | None = None,
    num_workers: int = 0,
    local_files_only: bool = False,
    cache_dir: str | None = None,
) -> DinoTrainResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    samples = build_train_index(data_root, strict=False)
    labels = [0 if s.is_authentic else 1 for s in samples]
    folds_list = list(_iter_stratified_folds(labels, folds, seed))
    if fold < 0 or fold >= len(folds_list):
        raise ValueError(f"Invalid fold {fold}; must be in [0, {len(folds_list) - 1}]")

    train_idx, val_idx = folds_list[fold]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    ds_train = DinoSegDataset(train_samples, image_size, train=True, seed=seed)
    ds_val = DinoSegDataset(val_samples, image_size, train=False, seed=seed)

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = dinov2.DinoSeg(
        dino_path,
        decoder_dropout=decoder_dropout,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    ).to(device)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    output_root = Path(output_dir)
    save_dir = output_root / f"fold_{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"

    best_dice = -1.0
    best_epoch = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses = []
        for xb, yb in dl_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        model.eval()
        val_dices = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                val_dices.append(_dice_from_logits(logits, yb))

        mean_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_dice = float(np.mean(val_dices)) if val_dices else float("nan")
        print(f"[DINO] epoch {epoch:02d}/{epochs} | train_loss={mean_loss:.4f} | dice@0.5={mean_dice:.4f}")

        if float(mean_dice) > best_dice:
            best_dice = float(mean_dice)
            best_epoch = int(epoch)
            ckpt = {
                "head_state": model.head.state_dict(),
                "config": {
                    "dino_path": str(dino_path),
                    "image_size": int(image_size),
                    "decoder_dropout": float(decoder_dropout),
                    "fold": int(fold),
                    "seed": int(seed),
                },
                "score": float(best_dice),
            }
            torch.save(ckpt, best_path)
            print("[DINO] saved best ->", best_path)

        if patience and best_epoch and (int(epoch) - int(best_epoch) >= int(patience)):
            print(f"[DINO] early stopping: sem melhora por {patience} épocas (best_epoch={best_epoch}).")
            break

    meta_path = save_dir / "train_meta.json"
    with meta_path.open("w") as f:
        json.dump(
            {
                "fold": int(fold),
                "epochs": int(epochs),
                "best_epoch": int(best_epoch),
                "best_dice": float(best_dice),
                "device": str(device),
            },
            f,
            indent=2,
        )

    print("[DINO] done. best dice:", best_dice)
    print("[DINO] checkpoint:", best_path)
    return DinoTrainResult(best_path=best_path, best_dice=best_dice, best_epoch=best_epoch)


def _prediction_path(preds_root: Path, sample_rel_path: Path, *, fold: int | None, use_folds: bool) -> Path:
    if use_folds and fold is not None:
        return (preds_root / f"fold_{fold}" / sample_rel_path).with_suffix(".npy")
    return (preds_root / sample_rel_path).with_suffix(".npy")


def predict_dino_oof(
    *,
    data_root: str | Path = "data/recodai",
    preds_root: str | Path = "outputs/preds_dino",
    folds: int = 5,
    fold: int = -1,
    seed: int = 42,
    dino_path: str | None = None,
    head_ckpt: str | Path | None = None,
    head_ckpt_dir: str | Path = "outputs/models_dino",
    image_size: int = 0,
    decoder_dropout: float = -1.0,
    device: str | None = None,
    tta_modes: tuple[str, ...] = DEFAULT_TTA,
    limit: int = 0,
    save_probs: bool = True,
    score: bool = True,
    run_dir: str | Path | None = None,
    threshold_factor: float = 0.3,
    min_area: int = 30,
    min_area_percent: float = 0.0005,
    min_confidence: float = 0.33,
    closing: int = 5,
    opening: int = 3,
    morph_iters: int = 1,
    closing_iters: int | None = None,
    opening_iters: int | None = None,
    local_files_only: bool = False,
    cache_dir: str | None = None,
) -> DinoOOFResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    samples = build_train_index(data_root, strict=False)
    if limit:
        samples = samples[:limit]

    labels = [0 if s.is_authentic else 1 for s in samples]
    folds_list = list(_iter_stratified_folds(labels, folds, seed))
    fold_ids = list(range(len(folds_list)))
    if fold >= 0:
        fold_ids = [int(fold)]
        folds_list = [folds_list[int(fold)]]

    preds_root = Path(preds_root)
    run_dir_path = Path(run_dir) if run_dir else (Path("runs") / f"dino_oof_{_timestamp()}")
    run_dir_path.mkdir(parents=True, exist_ok=True)

    all_rows = []
    fold_scores: dict[int, float] = {}

    for fold_id, (_, val_idx) in zip(fold_ids, folds_list):
        if head_ckpt:
            ckpt_path = Path(head_ckpt)
        else:
            ckpt_path = Path(head_ckpt_dir) / f"fold_{fold_id}" / "best.pt"
            if not ckpt_path.exists():
                ckpt_path = Path(head_ckpt_dir) / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Head checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        dino_path_final = dino_path or ckpt_cfg.get("dino_path") or "facebook/dinov2-base"
        if dino_path and ckpt_cfg.get("dino_path") and str(dino_path) != str(ckpt_cfg.get("dino_path")):
            print(f"[WARN] dino_path override differs from ckpt config: {dino_path} vs {ckpt_cfg.get('dino_path')}")
        image_size_final = int(image_size) if int(image_size) > 0 else int(ckpt_cfg.get("image_size", 512))
        decoder_dropout_final = float(decoder_dropout) if float(decoder_dropout) >= 0 else float(ckpt_cfg.get("decoder_dropout", 0.0))

        model = dinov2.DinoSeg(
            str(dino_path_final),
            decoder_dropout=decoder_dropout_final,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).to(device)
        model.eval()

        if "head_state" in ckpt:
            model.head.load_state_dict(ckpt["head_state"])
        elif "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            raise ValueError(f"Checkpoint missing head_state/model_state: {ckpt_path}")

        val_samples = [samples[i] for i in val_idx]
        scores = []
        for sample in val_samples:
            img = load_image(sample.image_path, as_rgb=True)
            prob = _tta_predict(model, img, image_size_final, device, tta_modes)

            if save_probs:
                pred_path = _prediction_path(preds_root, sample.rel_path, fold=fold_id, use_folds=True)
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(pred_path, prob)

            if score:
                gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
                pred_instances = dino_prob_to_instances(
                    prob,
                    threshold_factor=threshold_factor,
                    min_area=min_area,
                    min_area_percent=min_area_percent,
                    min_confidence=min_confidence,
                    closing_ksize=closing,
                    opening_ksize=opening,
                    morph_iters=morph_iters,
                    closing_iters=closing_iters,
                    opening_iters=opening_iters,
                )
                score_val = float(score_image(gt_instances, pred_instances))
                scores.append(score_val)
                all_rows.append({"fold": int(fold_id), "case_id": sample.case_id, "score": score_val})

        if scores:
            fold_scores[int(fold_id)] = float(np.mean(scores))
            print(f"fold {fold_id}: mean RecodAI F1 {fold_scores[int(fold_id)]:.6f}")

    mean_score = float(np.mean(list(fold_scores.values()))) if fold_scores else None

    summary_path = run_dir_path / "summary.json"
    with summary_path.open("w") as f:
        json.dump({"mean_score": mean_score, "fold_scores": fold_scores}, f, indent=2)

    meta_path = run_dir_path / "meta.json"
    with meta_path.open("w") as f:
        json.dump(
            {
                "data_root": str(data_root),
                "preds_root": str(preds_root),
                "folds": int(folds),
                "fold": int(fold),
                "seed": int(seed),
                "dino_path": str(dino_path_final),
                "image_size": int(image_size_final),
                "device": str(device),
                "tta_modes": list(tta_modes),
                "closing": int(closing),
                "opening": int(opening),
                "closing_iters": int(closing_iters if closing_iters is not None else morph_iters),
                "opening_iters": int(opening_iters if opening_iters is not None else morph_iters),
            },
            f,
            indent=2,
        )

    _write_scores_csv(all_rows, run_dir_path / "scores.csv")
    print("[DINO OOF] run_dir:", run_dir_path)
    if mean_score is not None:
        print("[DINO OOF] mean score:", mean_score)

    return DinoOOFResult(run_dir=run_dir_path, mean_score=mean_score, fold_scores=fold_scores)


def tune_dino_thresholds(
    *,
    data_root: str | Path,
    preds_root: str | Path,
    folds: int = 5,
    seed: int = 42,
    fold: int = -1,
    adaptive_threshold: bool = True,
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    threshold_factors: Iterable[float] = (0.2, 0.3, 0.4),
    min_areas: Iterable[int] = (0, 30, 64, 128),
    min_area_percents: Iterable[float] = (0.0002, 0.0005, 0.001),
    min_confidences: Iterable[float] = (0.30, 0.33, 0.36, 0.40),
    closing: int = 0,
    closing_iters: int = 1,
    opening: int = 0,
    opening_iters: int = 1,
    fill_holes: bool = False,
    median: int = 0,
    limit: int = 0,
    out_config: str | Path | None = None,
) -> dict:
    data_root = Path(data_root)
    preds_root = Path(preds_root)

    samples = build_train_index(data_root, strict=False)
    if limit:
        samples = samples[:limit]

    if folds <= 1:
        raise ValueError("folds must be >= 2 for cross-validation")

    has_fold_dirs = (preds_root / "fold_0").exists()
    if not has_fold_dirs:
        raise FileNotFoundError("Expected OOF predictions under preds_root/fold_*/ for cross-validation.")

    labels = [0 if s.is_authentic else 1 for s in samples]
    folds_list = list(_iter_stratified_folds(labels, folds, seed))
    fold_ids = list(range(len(folds_list)))
    if fold >= 0:
        fold_ids = [int(fold)]
        folds_list = [folds_list[int(fold)]]

    thresholds = list(thresholds)
    threshold_factors = list(threshold_factors)
    min_areas = list(min_areas)
    min_area_percents = list(min_area_percents)
    min_confidences = list(min_confidences)

    if adaptive_threshold and not threshold_factors:
        raise ValueError("threshold_factors must be provided when adaptive_threshold is True")
    if not adaptive_threshold and not thresholds:
        raise ValueError("thresholds must be provided when adaptive_threshold is False")

    best = {
        "score": -1.0,
        "adaptive_threshold": bool(adaptive_threshold),
        "threshold": None,
        "threshold_factor": None,
        "min_area": None,
        "min_area_percent": None,
        "min_confidence": None,
    }

    thr_grid = [None] if adaptive_threshold else thresholds
    factor_grid = threshold_factors if adaptive_threshold else [0.0]

    for thr in thr_grid:
        for factor in factor_grid:
            for min_area in min_areas:
                for min_area_percent in min_area_percents:
                    for min_conf in min_confidences:
                        scores = []
                        for fold_id, (_, val_idx) in zip(fold_ids, folds_list):
                            val_samples = [samples[i] for i in val_idx]
                            for sample in val_samples:
                                pred_path = _prediction_path(
                                    preds_root, sample.rel_path, fold=fold_id, use_folds=True
                                )
                                if not pred_path.exists():
                                    raise FileNotFoundError(f"Missing prediction: {pred_path}")

                                pred = np.load(pred_path)
                                if pred.ndim != 2:
                                    raise ValueError(f"Unsupported prediction shape: {pred.shape}")

                                pred_instances = prob_to_instances(
                                    pred,
                                    threshold=float(thr if thr is not None else 0.5),
                                    adaptive_threshold=bool(adaptive_threshold),
                                    threshold_factor=float(factor),
                                    min_area=int(min_area),
                                    min_area_percent=float(min_area_percent),
                                    min_confidence=float(min_conf),
                                    closing_ksize=int(closing),
                                    closing_iters=int(closing_iters),
                                    opening_ksize=int(opening),
                                    opening_iters=int(opening_iters),
                                    fill_holes_enabled=bool(fill_holes),
                                    median_ksize=int(median),
                                )

                                gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
                                scores.append(score_image(gt_instances, pred_instances))

                        mean_score = float(np.mean(scores)) if scores else 0.0
                        thr_label = "adaptive" if adaptive_threshold else f"{float(thr):.2f}"
                        print(
                            f"thr={thr_label} factor={float(factor):.2f} min_area={min_area} "
                            f"min_area_percent={min_area_percent} min_conf={min_conf} score={mean_score:.6f}"
                        )
                        if mean_score > best["score"]:
                            best = {
                                "score": mean_score,
                                "adaptive_threshold": bool(adaptive_threshold),
                                "threshold": None if adaptive_threshold else float(thr),
                                "threshold_factor": float(factor),
                                "min_area": int(min_area),
                                "min_area_percent": float(min_area_percent),
                                "min_confidence": float(min_conf),
                            }

    if out_config:
        out_path = Path(out_config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(
                {
                    "adaptive_threshold": bool(best["adaptive_threshold"]),
                    "threshold": best["threshold"],
                    "threshold_factor": best["threshold_factor"],
                    "min_area": best["min_area"],
                    "min_area_percent": best["min_area_percent"],
                    "min_confidence": best["min_confidence"],
                    "closing": int(closing),
                    "closing_iters": int(closing_iters),
                    "opening": int(opening),
                    "opening_iters": int(opening_iters),
                    "fill_holes": bool(fill_holes),
                    "median": int(median),
                },
                f,
                indent=2,
            )
        print(f"Wrote {out_path}")

    print(
        "Best:",
        f"score={best['score']:.6f}",
        f"adaptive={best['adaptive_threshold']}",
        f"threshold={best['threshold']}",
        f"factor={best['threshold_factor']}",
        f"min_area={best['min_area']}",
        f"min_area_percent={best['min_area_percent']}",
        f"min_confidence={best['min_confidence']}",
    )

    return best
