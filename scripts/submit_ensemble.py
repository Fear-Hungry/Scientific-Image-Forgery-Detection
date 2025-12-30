#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.checkpoints import build_classifier_from_config, build_segmentation_from_config, load_checkpoint
from forgeryseg.constants import AUTHENTIC_LABEL
from forgeryseg.dataset import build_test_index, load_image
from forgeryseg.inference import apply_tta, normalize_image, predict_image, undo_tta
from forgeryseg.postprocess import prob_to_instances
from forgeryseg.rle import encode_instances


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()


DEFAULT_TTA = ("none", "hflip", "vflip")
DEFAULT_CLS_SKIP_THRESHOLD = 0.30
# Pesos padrão (exemplo): DINOv2 + U-Net++ + SegFormer.
DEFAULT_MODEL_WEIGHTS = {
    "dinov2_base_light": 0.5,
    "unetpp_effnet_b7": 0.3,
    "segformer_mit_b3": 0.2,
}


@dataclass(frozen=True)
class SegEntry:
    model_id: str
    ckpt_path: Path
    model: nn.Module
    weight: float
    score: float | None


def _maybe_tqdm(it: Iterable, enabled: bool, desc: str):
    if not enabled:
        return it
    try:
        from tqdm.auto import tqdm

        return tqdm(it, desc=desc)
    except Exception:
        return it


def _find_models_dir(dir_name: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            candidates: list[Path] = []
            for ds in sorted(ki.glob("*")):
                for base in (ds, ds / "recodai_bundle"):
                    cand = base / "outputs" / dir_name
                    if cand.exists():
                        candidates.append(cand)
            if candidates:
                if len(candidates) > 1:
                    print(f"[CKPT] múltiplos candidatos para outputs/{dir_name}; usando o primeiro:")
                    for c in candidates:
                        print(" -", c)
                return candidates[0]

    local = Path("outputs") / dir_name
    if local.exists():
        return local

    raise FileNotFoundError(f"Não encontrei outputs/{dir_name}. Passe --models-dir/--cls-models-dir explicitamente.")


def _parse_csv_list(text: str) -> list[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _find_checkpoint_paths(models_dir: Path) -> tuple[str, list[Path]]:
    """
    Return (pattern_used, paths) for segmentation checkpoints inside `models_dir`.

    We prefer the standard training layout used by this repo:
    - outputs/models_seg/<model_id>/fold_<k>/best.pt  -> "*/*/best.pt"

    But we also accept alternative layouts (common when users copy files around):
    - outputs/models_seg/<model_id>/best.pt           -> "*/best.pt"
    - outputs/models_seg/**/best.pt                   -> "**/best.pt"

    As a last resort we also look for "last.pt" with the same patterns.
    """
    patterns = [
        "*/*/best.pt",
        "*/best.pt",
        "**/best.pt",
        "*/*/last.pt",
        "*/last.pt",
        "**/last.pt",
    ]
    for pat in patterns:
        matches = sorted(models_dir.glob(pat))
        if matches:
            return pat, matches
    return "", []


def _infer_model_id_from_path(models_dir: Path, ckpt_path: Path) -> str:
    try:
        rel = ckpt_path.relative_to(models_dir)
        parts = rel.parts
    except Exception:
        parts = ckpt_path.parts

    if not parts:
        return ckpt_path.parent.parent.name

    # Common layouts:
    # - <model_id>/fold_0/best.pt
    # - <model_id>/fold_0/weights/best.pt
    if len(parts) >= 1 and not str(parts[0]).startswith("fold_"):
        return str(parts[0])

    # If models_dir already points at "<model_id>", we likely have:
    # - fold_0/best.pt
    return models_dir.name


def _load_seg_models(
    models_dir: Path,
    device: str,
    *,
    model_ids: list[str] | None,
    model_weights: dict[str, float],
    top_k_per_model: int,
    allow_empty: bool = False,
) -> list[SegEntry]:
    entries: list[SegEntry] = []
    pattern, ckpt_paths = _find_checkpoint_paths(models_dir)
    if pattern:
        print(f"[SEG] scanning checkpoints: {models_dir} (pattern: {pattern})")
    model_ids_set = set(model_ids) if model_ids else None

    for ckpt_path in ckpt_paths:
        state, cfg = load_checkpoint(ckpt_path)
        model_id = str(cfg.get("model_id") or _infer_model_id_from_path(models_dir, ckpt_path))
        if model_ids_set and model_id not in model_ids_set:
            continue
        score = cfg.get("score", None)
        if score is None:
            ckpt_raw = torch.load(ckpt_path, map_location="cpu")
            score = ckpt_raw.get("score", None) if isinstance(ckpt_raw, dict) else None

        m = build_segmentation_from_config(cfg)
        m.load_state_dict(state)
        m.to(device)
        m.eval()
        w = float(model_weights.get(model_id, 1.0))
        entries.append(SegEntry(model_id=model_id, ckpt_path=ckpt_path, model=m, weight=w, score=score))

    if not entries:
        tried = ", ".join(
            [
                "*/*/best.pt",
                "*/best.pt",
                "**/best.pt",
                "*/*/last.pt",
                "*/last.pt",
                "**/last.pt",
            ]
        )
        msg = f"Nenhum checkpoint encontrado em {models_dir} (patterns testados: {tried})."
        if allow_empty:
            print("[SEG]", msg)
            return []
        raise RuntimeError(msg)

    if top_k_per_model > 0:
        by_id: dict[str, list[SegEntry]] = {}
        for e in entries:
            by_id.setdefault(e.model_id, []).append(e)
        filtered: list[SegEntry] = []
        for mid, group in by_id.items():
            group_sorted = sorted(group, key=lambda x: float(x.score) if x.score is not None else -1.0, reverse=True)
            filtered.extend(group_sorted[: int(top_k_per_model)])
        entries = filtered

    print("[SEG] loaded models:", len(entries))
    for e in entries[:10]:
        print(" -", e.model_id, "|", e.ckpt_path)
    if len(entries) > 10:
        print(" ...")
    return entries


def _load_cls_models(cls_models_dir: Path, device: str) -> tuple[list[nn.Module], int, list[float]]:
    models: list[nn.Module] = []
    image_size = 0
    thresholds: list[float] = []
    for ckpt_path in sorted(cls_models_dir.glob("fold_*/best.pt")):
        state, cfg = load_checkpoint(ckpt_path)
        m, size = build_classifier_from_config(cfg)
        image_size = int(size)
        m.load_state_dict(state)
        m.to(device)
        m.eval()
        models.append(m)
        if "cls_threshold" in cfg:
            try:
                thresholds.append(float(cfg["cls_threshold"]))
            except Exception:
                pass

    if models:
        print("[CLS] loaded models:", len(models), "| image_size:", image_size)
    return models, int(image_size), thresholds


@torch.no_grad()
def predict_prob_forged(models: list[nn.Module], image: np.ndarray, device: str, image_size: int) -> float:
    import torch.nn.functional as F

    if not models:
        return 0.0
    img = normalize_image(image)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    if image_size and x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    probs: list[float] = []
    for m in models:
        logits = m(x).view(-1)
        probs.append(float(torch.sigmoid(logits)[0].item()))
    return float(np.mean(probs))


def predict_seg_ensemble_prob(
    entries: list[SegEntry],
    image: np.ndarray,
    device: str,
    *,
    tile_size: int,
    overlap: int,
    max_size: int,
    tta_modes: tuple[str, ...],
) -> np.ndarray:
    modes = tta_modes if tta_modes else ("none",)
    prob_sum: np.ndarray | None = None
    count = 0

    weights = [float(e.weight) for e in entries]
    wsum = float(sum(weights)) if weights else 1.0

    for mode in modes:
        img_t = apply_tta(image, mode)
        ens: np.ndarray | None = None
        for e in entries:
            p = predict_image(e.model, img_t, device, tile_size=tile_size, overlap=overlap, max_size=max_size)
            p = p * float(e.weight)
            ens = p if ens is None else (ens + p)
        assert ens is not None
        ens = ens / wsum
        ens = undo_tta(ens, mode)
        prob_sum = ens if prob_sum is None else (prob_sum + ens)
        count += 1

    assert prob_sum is not None
    return prob_sum / float(max(count, 1))


def _write_authentic_submission(out_path: Path, samples: Iterable) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()
        for s in samples:
            writer.writerow({"case_id": s.case_id, "annotation": AUTHENTIC_LABEL})


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble inference (segmentation + optional classifier gating) -> submission.csv")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root")
    parser.add_argument("--models-dir", default="", help="Directory with outputs/models_seg (optional)")
    parser.add_argument("--cls-models-dir", default="", help="Directory with outputs/models_cls (optional)")
    parser.add_argument("--out-csv", default="", help="Output submission.csv path")
    parser.add_argument("--config", default="", help="Optional JSON config (overrides defaults)")
    parser.add_argument("--model-ids", default="", help="Comma-separated model_id filter (default: all)")
    parser.add_argument("--top-k-per-model", type=int, default=0, help="Keep top-K checkpoints per model_id by score")
    parser.add_argument("--tta", default=",".join(DEFAULT_TTA), help="Comma-separated TTA modes")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size")
    parser.add_argument("--overlap", type=int, default=128, help="Tile overlap")
    parser.add_argument("--max-size", type=int, default=0, help="Optional resize long side")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--adaptive-threshold", action="store_true", help="Use adaptive threshold (mean + factor * std)")
    parser.add_argument("--threshold-factor", type=float, default=0.3, help="Factor for adaptive threshold")
    parser.add_argument("--min-area", type=int, default=30, help="Minimum component area")
    parser.add_argument("--min-area-percent", type=float, default=0.0, help="Minimum mask area fraction (0 disables)")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum mean prob inside mask (0 disables)")
    parser.add_argument("--closing", type=int, default=0, help="Morphological closing kernel size (0=disabled)")
    parser.add_argument("--closing-iters", type=int, default=1, help="Morphological closing iterations")
    parser.add_argument("--opening", type=int, default=0, help="Morphological opening kernel size (0=disabled)")
    parser.add_argument("--opening-iters", type=int, default=1, help="Morphological opening iterations")
    parser.add_argument("--fill-holes", action="store_true", help="Fill holes in binary mask")
    parser.add_argument("--median", type=int, default=0, help="Median smoothing kernel size (0=disabled, odd>=3)")
    parser.add_argument(
        "--cls-skip-threshold",
        default=str(DEFAULT_CLS_SKIP_THRESHOLD),
        help="Skip seg when p_forged < this (<=0 disables the gate). Use 'auto' to load from cls checkpoints.",
    )
    parser.add_argument("--device", default="", help="Device override")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of test samples (debug)")
    fb = parser.add_mutually_exclusive_group()
    fb.add_argument(
        "--fallback-authentic",
        dest="fallback_authentic",
        action="store_true",
        help="If no segmentation checkpoints are found, write an 'authentic' submission and exit 0.",
    )
    fb.add_argument(
        "--no-fallback-authentic",
        dest="fallback_authentic",
        action="store_false",
        help="Disable Kaggle default fallback behavior when checkpoints are missing.",
    )
    parser.set_defaults(fallback_authentic=None)
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with Path(args.config).open("r") as f:
            cfg = json.load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(cfg.get("data_root", args.data_root))
    samples = build_test_index(data_root)
    if args.limit:
        samples = samples[: int(args.limit)]

    out_csv = cfg.get("out_csv", args.out_csv)
    if not out_csv:
        out_csv = "/kaggle/working/submission.csv" if is_kaggle() else str(Path("submission.csv").resolve())
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_fallback = cfg.get("fallback_authentic", args.fallback_authentic)
    if cfg_fallback:
        raise RuntimeError("fallback_authentic disabled: no fallbacks allowed when checkpoints are missing.")
    fallback_authentic = False

    try:
        models_dir = _find_models_dir("models_seg", cfg.get("models_dir", args.models_dir) or None)
    except FileNotFoundError:
        if fallback_authentic:
            print("[SEG] outputs/models_seg não encontrado; gerando submission baseline 'authentic'.")
            _write_authentic_submission(out_path, samples)
            print("wrote:", out_path)
            return
        raise
    cls_models_dir = None
    cls_dir_text = cfg.get("cls_models_dir", args.cls_models_dir) or ""
    if cls_dir_text:
        cls_models_dir = Path(cls_dir_text)
    else:
        try:
            cls_models_dir = _find_models_dir("models_cls", None)
        except Exception:
            cls_models_dir = None

    model_ids = _parse_csv_list(cfg.get("model_ids", args.model_ids))

    model_weights: dict[str, float] = {}
    weights_from_config = False
    cfg_models = cfg.get("models", [])
    if cfg_models:
        weights_from_config = True
        for m in cfg_models:
            try:
                model_weights[str(m.get("model_id"))] = float(m.get("weight", 1.0))
            except Exception:
                continue
    else:
        model_weights = dict(DEFAULT_MODEL_WEIGHTS)

    if isinstance(cfg.get("tta_modes"), (list, tuple)):
        tta_modes = tuple(str(x) for x in cfg.get("tta_modes") if str(x).strip())
    else:
        tta_modes = tuple(_parse_csv_list(cfg.get("tta", args.tta)))

    if not model_ids and model_weights and weights_from_config:
        model_ids = sorted(model_weights.keys())
    tile_size = int(cfg.get("tile_size", args.tile_size))
    overlap = int(cfg.get("overlap", args.overlap))
    max_size = int(cfg.get("max_size", args.max_size))
    threshold = float(cfg.get("threshold", args.threshold))
    adaptive_threshold = bool(cfg.get("adaptive_threshold", args.adaptive_threshold))
    threshold_factor = float(cfg.get("threshold_factor", args.threshold_factor))
    min_area = int(cfg.get("min_area", args.min_area))
    min_area_percent = float(cfg.get("min_area_percent", args.min_area_percent))
    min_confidence = float(cfg.get("min_confidence", args.min_confidence))
    closing = int(cfg.get("closing", args.closing))
    closing_iters = int(cfg.get("closing_iters", args.closing_iters))
    opening = int(cfg.get("opening", args.opening))
    opening_iters = int(cfg.get("opening_iters", args.opening_iters))
    fill_holes = bool(cfg.get("fill_holes", args.fill_holes))
    median = int(cfg.get("median", args.median))
    def _parse_cls_skip_threshold(value):
        if value is None:
            return float(DEFAULT_CLS_SKIP_THRESHOLD)
        if isinstance(value, str):
            text = value.strip().lower()
            if text == "auto":
                return None
            if text == "":
                return float(DEFAULT_CLS_SKIP_THRESHOLD)
            return float(text)
        return float(value)

    cfg_cls_skip = cfg.get("cls_skip_threshold", args.cls_skip_threshold)
    cls_skip_threshold = _parse_cls_skip_threshold(cfg_cls_skip)
    top_k_per_model = int(cfg.get("top_k_per_model", args.top_k_per_model))

    seg_entries = _load_seg_models(
        models_dir,
        device,
        model_ids=model_ids or None,
        model_weights=model_weights,
        top_k_per_model=top_k_per_model,
        allow_empty=fallback_authentic,
    )
    if not seg_entries:
        raise RuntimeError("[SEG] sem checkpoints carregados; execução encerrada.")

    cls_models: list[nn.Module] = []
    cls_image_size = 0
    cls_thresholds: list[float] = []
    if cls_skip_threshold is not None and float(cls_skip_threshold) <= 0.0:
        print("[CLS] gate disabled (cls_skip_threshold<=0).")
    else:
        if cls_models_dir is None or not cls_models_dir.exists():
            raise RuntimeError("[CLS] gate enabled but no classifier checkpoints found.")
        cls_models, cls_image_size, cls_thresholds = _load_cls_models(cls_models_dir, device)
        if not cls_models:
            raise RuntimeError("[CLS] gate enabled but no classifier checkpoints found.")
        if cls_skip_threshold is None:
            if not cls_thresholds:
                raise RuntimeError("[CLS] cls_skip_threshold=auto but checkpoints lack cls_threshold.")
            cls_skip_threshold = float(np.median(cls_thresholds))
            print(f"[CLS] auto threshold from checkpoints: {cls_skip_threshold:.4f}")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()

        for s in _maybe_tqdm(samples, enabled=True, desc="infer"):
            img = load_image(s.image_path)
            if cls_models and float(cls_skip_threshold) > 0.0:
                p_forged = predict_prob_forged(cls_models, img, device, cls_image_size)
                if float(p_forged) < float(cls_skip_threshold):
                    writer.writerow({"case_id": s.case_id, "annotation": AUTHENTIC_LABEL})
                    continue

            prob = predict_seg_ensemble_prob(
                seg_entries,
                img,
                device,
                tile_size=tile_size,
                overlap=overlap,
                max_size=max_size,
                tta_modes=tta_modes,
            )
            instances = prob_to_instances(
                prob,
                threshold=threshold,
                adaptive_threshold=adaptive_threshold,
                threshold_factor=threshold_factor,
                min_area=min_area,
                min_area_percent=min_area_percent,
                min_confidence=min_confidence,
                closing_ksize=closing,
                closing_iters=closing_iters,
                opening_ksize=opening,
                opening_iters=opening_iters,
                fill_holes_enabled=fill_holes,
                median_ksize=median,
            )
            annotation = encode_instances(instances)
            writer.writerow({"case_id": s.case_id, "annotation": annotation})

    print("wrote:", out_path)


if __name__ == "__main__":
    main()
