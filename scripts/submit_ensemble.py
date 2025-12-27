#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
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
from forgeryseg.inference import normalize_image, predict_image
from forgeryseg.postprocess import prob_to_instances
from forgeryseg.rle import encode_instances


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()


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


def _apply_tta(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image
    if mode == "hflip":
        return np.ascontiguousarray(image[:, ::-1])
    if mode == "vflip":
        return np.ascontiguousarray(image[::-1, :])
    if mode == "hvflip":
        return np.ascontiguousarray(image[::-1, ::-1])
    raise ValueError(f"tta mode inválido: {mode}")


def _undo_tta(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return mask
    if mode == "hflip":
        return np.ascontiguousarray(mask[:, ::-1])
    if mode == "vflip":
        return np.ascontiguousarray(mask[::-1, :])
    if mode == "hvflip":
        return np.ascontiguousarray(mask[::-1, ::-1])
    raise ValueError(f"tta mode inválido: {mode}")


def _load_seg_models(
    models_dir: Path,
    device: str,
    *,
    model_ids: list[str] | None,
    model_weights: dict[str, float],
    top_k_per_model: int,
) -> list[SegEntry]:
    entries: list[SegEntry] = []
    for ckpt_path in sorted(models_dir.glob("*/*/best.pt")):
        try:
            state, cfg = load_checkpoint(ckpt_path)
            model_id = str(cfg.get("model_id", ckpt_path.parent.parent.name))
            if model_ids and model_id not in set(model_ids):
                continue
            score = cfg.get("score", None)
            if score is None:
                try:
                    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
                    score = ckpt_raw.get("score", None) if isinstance(ckpt_raw, dict) else None
                except Exception:
                    score = None

            m = build_segmentation_from_config(cfg)
            m.load_state_dict(state)
            m.to(device)
            m.eval()
            w = float(model_weights.get(model_id, 1.0))
            entries.append(SegEntry(model_id=model_id, ckpt_path=ckpt_path, model=m, weight=w, score=score))
        except Exception:
            print("[ERRO] falha ao carregar seg:", ckpt_path)
            traceback.print_exc()

    if not entries:
        raise RuntimeError(f"Nenhum checkpoint encontrado em {models_dir} (pattern: */*/best.pt).")

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


def _load_cls_models(cls_models_dir: Path, device: str) -> tuple[list[nn.Module], int]:
    models: list[nn.Module] = []
    image_size = 0
    for ckpt_path in sorted(cls_models_dir.glob("fold_*/best.pt")):
        try:
            state, cfg = load_checkpoint(ckpt_path)
            m, size = build_classifier_from_config(cfg)
            image_size = int(size)
            m.load_state_dict(state)
            m.to(device)
            m.eval()
            models.append(m)
        except Exception:
            print("[ERRO] falha ao carregar cls:", ckpt_path)
            traceback.print_exc()

    if models:
        print("[CLS] loaded models:", len(models), "| image_size:", image_size)
    return models, int(image_size)


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
        img_t = _apply_tta(image, mode)
        ens: np.ndarray | None = None
        for e in entries:
            p = predict_image(e.model, img_t, device, tile_size=tile_size, overlap=overlap, max_size=max_size)
            p = p * float(e.weight)
            ens = p if ens is None else (ens + p)
        assert ens is not None
        ens = ens / wsum
        ens = _undo_tta(ens, mode)
        prob_sum = ens if prob_sum is None else (prob_sum + ens)
        count += 1

    assert prob_sum is not None
    return prob_sum / float(max(count, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble inference (segmentation + optional classifier gating) -> submission.csv")
    parser.add_argument("--data-root", default="data/recodai", help="Dataset root")
    parser.add_argument("--models-dir", default="", help="Directory with outputs/models_seg (optional)")
    parser.add_argument("--cls-models-dir", default="", help="Directory with outputs/models_cls (optional)")
    parser.add_argument("--out-csv", default="", help="Output submission.csv path")
    parser.add_argument("--config", default="", help="Optional JSON config (overrides defaults)")
    parser.add_argument("--model-ids", default="", help="Comma-separated model_id filter (default: all)")
    parser.add_argument("--top-k-per-model", type=int, default=0, help="Keep top-K checkpoints per model_id by score")
    parser.add_argument("--tta", default="none,hflip", help="Comma-separated TTA modes")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size")
    parser.add_argument("--overlap", type=int, default=128, help="Tile overlap")
    parser.add_argument("--max-size", type=int, default=0, help="Optional resize long side")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--min-area", type=int, default=32, help="Minimum component area")
    parser.add_argument("--closing", type=int, default=0, help="Morphological closing kernel size (0=disabled)")
    parser.add_argument("--closing-iters", type=int, default=1, help="Morphological closing iterations")
    parser.add_argument("--fill-holes", action="store_true", help="Fill holes in binary mask")
    parser.add_argument("--median", type=int, default=0, help="Median smoothing kernel size (0=disabled, odd>=3)")
    parser.add_argument("--cls-skip-threshold", type=float, default=0.10, help="Skip seg when p_forged < this")
    parser.add_argument("--device", default="", help="Device override")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of test samples (debug)")
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

    models_dir = _find_models_dir("models_seg", cfg.get("models_dir", args.models_dir) or None)
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
    cfg_models = cfg.get("models", [])
    for m in cfg_models:
        try:
            model_weights[str(m.get("model_id"))] = float(m.get("weight", 1.0))
        except Exception:
            continue

    if isinstance(cfg.get("tta_modes"), (list, tuple)):
        tta_modes = tuple(str(x) for x in cfg.get("tta_modes") if str(x).strip())
    else:
        tta_modes = tuple(_parse_csv_list(cfg.get("tta", args.tta)))

    if not model_ids and model_weights:
        model_ids = sorted(model_weights.keys())
    tile_size = int(cfg.get("tile_size", args.tile_size))
    overlap = int(cfg.get("overlap", args.overlap))
    max_size = int(cfg.get("max_size", args.max_size))
    threshold = float(cfg.get("threshold", args.threshold))
    min_area = int(cfg.get("min_area", args.min_area))
    closing = int(cfg.get("closing", args.closing))
    closing_iters = int(cfg.get("closing_iters", args.closing_iters))
    fill_holes = bool(cfg.get("fill_holes", args.fill_holes))
    median = int(cfg.get("median", args.median))
    cls_skip_threshold = float(cfg.get("cls_skip_threshold", args.cls_skip_threshold))
    top_k_per_model = int(cfg.get("top_k_per_model", args.top_k_per_model))

    seg_entries = _load_seg_models(
        models_dir,
        device,
        model_ids=model_ids or None,
        model_weights=model_weights,
        top_k_per_model=top_k_per_model,
    )

    cls_models: list[nn.Module] = []
    cls_image_size = 0
    if cls_models_dir is not None and cls_models_dir.exists():
        cls_models, cls_image_size = _load_cls_models(cls_models_dir, device)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()

        for s in _maybe_tqdm(samples, enabled=True, desc="infer"):
            img = load_image(s.image_path)
            if cls_models:
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
                min_area=min_area,
                closing_ksize=closing,
                closing_iters=closing_iters,
                fill_holes_enabled=fill_holes,
                median_ksize=median,
            )
            annotation = encode_instances(instances)
            writer.writerow({"case_id": s.case_id, "annotation": annotation})

    print("wrote:", out_path)


if __name__ == "__main__":
    main()
