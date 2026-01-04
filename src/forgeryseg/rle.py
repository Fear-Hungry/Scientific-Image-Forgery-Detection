from __future__ import annotations

import json
from typing import Sequence

import numpy as np


def _encode_single(mask: np.ndarray) -> list[int]:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")
    mask = (mask > 0).astype(np.uint8)
    dots = np.where(mask.flatten(order="F") == 1)[0]
    run_lengths: list[int] = []
    prev = -2
    for b in dots:
        b = int(b)
        if b > prev + 1:
            run_lengths.extend((int(b + 1), 0))  # 1-based starts
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def masks_to_annotation(masks: Sequence[np.ndarray]) -> str:
    cleaned = [(m > 0).astype(np.uint8) for m in masks if m is not None and np.any(m)]
    if not cleaned:
        return "authentic"
    return ";".join(json.dumps(_encode_single(m)) for m in cleaned)


def _decode_single(rle: list[int], shape: tuple[int, int]) -> np.ndarray:
    if len(rle) % 2 != 0:
        raise ValueError(f"RLE length must be even, got {len(rle)}")
    h, w = shape
    flat = np.zeros(h * w, dtype=np.uint8)
    for start, length in zip(rle[0::2], rle[1::2], strict=True):
        if length <= 0:
            continue
        start0 = int(start) - 1
        end = start0 + int(length)
        flat[start0:end] = 1
    return flat.reshape((h, w), order="F")


def annotation_to_masks(annotation: str | float | None, shape: tuple[int, int]) -> list[np.ndarray]:
    if annotation is None:
        return []
    if isinstance(annotation, float) and np.isnan(annotation):
        return []
    ann = str(annotation).strip()
    if not ann or ann.lower() == "authentic":
        return []

    parts = [p.strip() for p in ann.split(";") if p.strip()]
    masks: list[np.ndarray] = []
    for part in parts:
        rle = json.loads(part)
        masks.append(_decode_single(rle, shape))
    return masks


def annotation_to_union_mask(annotation: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    masks = annotation_to_masks(annotation, shape)
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    stacked = np.stack(masks, axis=0).astype(bool)
    return np.any(stacked, axis=0).astype(np.uint8)
