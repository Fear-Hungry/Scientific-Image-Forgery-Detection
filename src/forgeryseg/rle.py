from __future__ import annotations

import json
from typing import Iterable, List, Sequence

import numpy as np

from .constants import AUTHENTIC_LABEL


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
    return (arr > 0).astype(np.uint8)


def rle_encode(mask: np.ndarray) -> List[int]:
    """
    Encode a single binary mask using Kaggle-style RLE.

    - Column-major order (Fortran / mask.T.flatten())
    - 1-indexed start positions
    """
    mask = _normalize_mask(mask)
    if mask.max() == 0:
        return []

    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return changes.tolist()


def rle_decode(rle: Sequence[int] | str | None, shape: tuple[int, int]) -> np.ndarray:
    """Decode a single RLE list or string into a binary mask."""
    if rle is None:
        return np.zeros(shape, dtype=np.uint8)

    if isinstance(rle, str):
        text = rle.strip()
        if text == "" or text.lower() == AUTHENTIC_LABEL:
            return np.zeros(shape, dtype=np.uint8)
        if text.startswith("["):
            rle = json.loads(text)
        else:
            rle = [int(x) for x in text.split()]

    rle = list(rle)
    if len(rle) % 2 != 0:
        raise ValueError("RLE length must be even (start, length pairs)")

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(rle[0::2], rle[1::2]):
        if length <= 0:
            continue
        start_index = int(start) - 1
        end_index = start_index + int(length)
        mask[start_index:end_index] = 1

    return mask.reshape(shape, order="F")


def _normalize_instances(masks: Iterable[np.ndarray] | np.ndarray | None) -> List[np.ndarray]:
    if masks is None:
        return []
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            return [_normalize_mask(masks)]
        if masks.ndim == 3:
            return [_normalize_mask(m) for m in masks]
    if isinstance(masks, (list, tuple)):
        return [_normalize_mask(m) for m in masks]
    raise ValueError("Unsupported mask container for RLE encoding")


def encode_instances(masks: Iterable[np.ndarray] | np.ndarray | None) -> str:
    """
    Encode a list/array of instance masks into the competition annotation string.
    Returns "authentic" if no positive pixels are found.
    """
    instances = _normalize_instances(masks)
    parts: List[str] = []
    for mask in instances:
        runs = rle_encode(mask)
        if runs:
            parts.append(json.dumps(runs))

    if not parts:
        return AUTHENTIC_LABEL

    return ";".join(parts)


def decode_annotation(annotation: str | None, shape: tuple[int, int]) -> List[np.ndarray]:
    """Decode an annotation string into a list of instance masks."""
    if annotation is None:
        return []

    text = annotation.strip()
    if text == "" or text.lower() == AUTHENTIC_LABEL:
        return []

    masks = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        masks.append(rle_decode(part, shape))
    return masks
