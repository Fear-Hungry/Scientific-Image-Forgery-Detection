from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .typing import Pathish


def resolve_existing_path(
    path: Pathish,
    *,
    roots: Iterable[Path] | None = None,
    search_kaggle_input: bool = True,
) -> Path:
    """
    Resolve a (possibly relative) path by searching multiple roots and (optionally) /kaggle/input.

    Returns the first existing candidate. If nothing exists, returns the original path (as Path).
    """
    p = Path(path)
    if p.exists():
        return p
    if p.is_absolute():
        return p

    if roots is not None:
        for root in roots:
            cand = Path(root) / p
            if cand.exists():
                return cand

    if search_kaggle_input:
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for d in kaggle_input.iterdir():
                if not d.is_dir():
                    continue
                cand = d / p
                if cand.exists():
                    return cand
                # common: dataset root contains a single folder with the repo inside
                try:
                    for child in d.iterdir():
                        if not child.is_dir():
                            continue
                        cand2 = child / p
                        if cand2.exists():
                            return cand2
                except PermissionError:
                    continue

    return p

