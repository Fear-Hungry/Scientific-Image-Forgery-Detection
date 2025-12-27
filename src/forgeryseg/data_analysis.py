from __future__ import annotations

from collections import Counter
from pathlib import Path
import traceback
from typing import Any, Sequence

import numpy as np


def quick_dataset_stats(
    samples: Sequence[Any],
    *,
    max_items: int = 200,
    seed: int = 42,
    name: str = "train",
    strict: bool = False,
) -> dict[str, object]:
    """Print lightweight dataset stats (sizes/modes/masks) from a sample of items.

    Expected sample fields (like `forgeryseg.dataset.Sample`):
    - `image_path`: filesystem path to an image
    - `mask_path`: optional filesystem path to a `.npy` mask (2D union mask or 3D instances)
    """
    if not samples:
        print(f"[{name}] vazio.")
        return {"total": 0, "sampled": 0}

    rng = np.random.default_rng(int(seed))
    idxs = np.arange(len(samples), dtype=np.int64)
    if len(idxs) > int(max_items):
        idxs = rng.choice(idxs, size=int(max_items), replace=False)

    modes: Counter[str] = Counter()
    heights: list[int] = []
    widths: list[int] = []
    mask_present = 0
    mask_shape_mismatch = 0
    mask_positive_frac: list[float] = []

    from PIL import Image

    for i in idxs.tolist():
        s = samples[int(i)]
        try:
            image_path = Path(getattr(s, "image_path"))
            with Image.open(image_path) as img:
                modes[str(img.mode)] += 1
                w, h = img.size
            heights.append(int(h))
            widths.append(int(w))

            mask_path = getattr(s, "mask_path", None)
            if mask_path is not None and Path(mask_path).exists():
                mask_present += 1
                masks = np.load(mask_path, mmap_mode="r")
                union = masks if masks.ndim == 2 else masks.max(axis=0)
                if union.shape != (h, w):
                    mask_shape_mismatch += 1
                pos = int((union > 0).sum())
                mask_positive_frac.append(pos / float(union.shape[0] * union.shape[1]))
        except Exception:
            print(f"[{name}] erro ao ler:", getattr(s, "image_path", s))
            traceback.print_exc()
            if strict:
                raise

    n = len(idxs)
    h_min, h_max = (min(heights), max(heights)) if heights else (None, None)
    w_min, w_max = (min(widths), max(widths)) if widths else (None, None)
    print(f"[{name}] amostra: {n}/{len(samples)}")
    print(f"[{name}] modos (originais):", dict(modes))
    print(f"[{name}] H: min={h_min} max={h_max} | W: min={w_min} max={w_max}")
    if mask_present:
        frac_mean = float(np.mean(mask_positive_frac)) if mask_positive_frac else 0.0
        frac_p95 = float(np.percentile(mask_positive_frac, 95)) if len(mask_positive_frac) >= 2 else frac_mean
        print(f"[{name}] máscaras presentes (na amostra): {mask_present} | shape mismatch: {mask_shape_mismatch}")
        print(f"[{name}] fração positiva (union): mean={frac_mean:.6f} p95={frac_p95:.6f}")

    return {
        "total": int(len(samples)),
        "sampled": int(n),
        "modes": dict(modes),
        "height_min": h_min,
        "height_max": h_max,
        "width_min": w_min,
        "width_max": w_max,
        "mask_present": int(mask_present),
        "mask_shape_mismatch": int(mask_shape_mismatch),
        "mask_positive_frac_mean": float(np.mean(mask_positive_frac)) if mask_positive_frac else 0.0,
    }

