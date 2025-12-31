#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - optional dependency
    minimize = None


def _center_crop_with_pad(mask: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w = mask.shape
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        h, w = mask.shape
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return mask[top : top + crop_h, left : left + crop_w]


def _load_union_mask(mask_path: Path) -> np.ndarray:
    masks = np.load(mask_path)
    if masks.ndim == 2:
        union = masks
    else:
        union = masks.max(axis=0)
    return (union > 0).astype(np.uint8)


def _collect_oof_predictions(oof_root: Path, model_ids: list[str]) -> tuple[dict[str, dict[str, Path]], list[str]]:
    case_map: dict[str, dict[str, Path]] = {}
    for mid in model_ids:
        mdir = oof_root / mid
        if not mdir.exists():
            continue
        for p in sorted(mdir.rglob("*.npy")):
            cid = p.stem
            case_map.setdefault(cid, {})[mid] = p
    valid_cases = [cid for cid, preds in case_map.items() if len(preds) == len(model_ids)]
    return case_map, sorted(valid_cases)


def _optimize_weights(stats: list[tuple[list[float], list[float], float]], model_ids: list[str]) -> dict[str, float]:
    if not stats:
        return {m: 1.0 / len(model_ids) for m in model_ids}

    if minimize is None:
        print("[WARN] scipy não disponível; usando pesos uniformes.")
        return {m: 1.0 / len(model_ids) for m in model_ids}

    def objective(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=np.float64)
        if w.sum() <= 0:
            return 1.0
        w = w / w.sum()
        dice_sum = 0.0
        for I_list, S_list, t_sum in stats:
            ens_I = sum(w[i] * I_list[i] for i in range(len(model_ids)))
            ens_S = sum(w[i] * S_list[i] for i in range(len(model_ids)))
            dice_sum += (2.0 * ens_I) / (ens_S + t_sum + 1e-6)
        return -(dice_sum / len(stats))

    w0 = np.ones(len(model_ids), dtype=np.float64) / len(model_ids)
    bounds = [(0.0, 1.0) for _ in model_ids]
    constraints = ({"type": "eq", "fun": lambda w: 1.0 - float(np.sum(w))},)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = np.asarray(res.x, dtype=np.float64)
    if w.sum() <= 0:
        w = w0
    else:
        w = w / w.sum()
    return {model_ids[i]: float(w[i]) for i in range(len(model_ids))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize ensemble weights from OOF predictions (soft Dice proxy).")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root containing train_masks")
    parser.add_argument("--oof-dir", default="outputs/oof", help="OOF root dir (outputs/oof/<model_id>/fold_*/case.npy)")
    parser.add_argument("--models", default="", help="Comma-separated model_ids (default: subdirs in oof-dir)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases (debug)")
    parser.add_argument("--out", default="outputs/ensemble_weights.json", help="Output JSON path")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    oof_root = Path(args.oof_dir)
    masks_dir = data_root / "train_masks"
    if not oof_root.exists():
        raise FileNotFoundError(oof_root)
    if not masks_dir.exists():
        raise FileNotFoundError(masks_dir)

    if args.models:
        model_ids = [t.strip() for t in str(args.models).split(",") if t.strip()]
    else:
        model_ids = sorted([p.name for p in oof_root.iterdir() if p.is_dir()])
    if not model_ids:
        raise RuntimeError("No model_ids found in oof-dir.")

    case_map, valid_cases = _collect_oof_predictions(oof_root, model_ids)
    if not valid_cases:
        raise RuntimeError("No cases common to all models.")

    if args.limit > 0:
        valid_cases = valid_cases[: int(args.limit)]

    stats: list[tuple[list[float], list[float], float]] = []
    for cid in valid_cases:
        pred_path = case_map[cid][model_ids[0]]
        pred0 = np.load(pred_path)
        if pred0.ndim != 2:
            pred0 = np.squeeze(pred0)
        h, w = pred0.shape

        mask_path = masks_dir / f"{cid}.npy"
        if mask_path.exists():
            gt = _load_union_mask(mask_path)
            gt = _center_crop_with_pad(gt, h, w)
        else:
            gt = np.zeros((h, w), dtype=np.uint8)

        I_list = []
        S_list = []
        t = gt.astype(np.float32)
        t_sum = float(t.sum())

        for mid in model_ids:
            p = np.load(case_map[cid][mid]).astype(np.float32)
            if p.ndim != 2:
                p = np.squeeze(p)
            if p.shape != (h, w):
                # Best-effort: center-crop/pad to match
                p = _center_crop_with_pad(p, h, w)
            I_list.append(float((p * t).sum()))
            S_list.append(float(p.sum()))

        stats.append((I_list, S_list, t_sum))

    weights = _optimize_weights(stats, model_ids)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(weights, f, indent=2)
    print("Saved weights to", out_path)


if __name__ == "__main__":
    main()
