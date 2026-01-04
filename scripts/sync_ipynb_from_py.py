from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path


def _parse_percent_cells(py_path: Path) -> list[dict]:
    lines = py_path.read_text(encoding="utf-8").splitlines()
    markers = [i for i, line in enumerate(lines) if line.startswith("# %%")]
    if not markers:
        raise ValueError(f"No '# %%' cell markers found in {py_path}")

    cells: list[dict] = []
    for mi, start in enumerate(markers):
        header = lines[start]
        end = markers[mi + 1] if mi + 1 < len(markers) else len(lines)
        content = lines[start + 1 : end]

        if header.startswith("# %% [markdown]"):
            md_lines: list[str] = []
            for l in content:
                if l.startswith("# "):
                    md_lines.append(l[2:])
                elif l.startswith("#"):
                    md_lines.append(l[1:])
                else:
                    md_lines.append(l)
            source = [s + "\n" for s in md_lines]
            while source and source[-1] == "\n":
                source.pop()
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            source = [s + "\n" for s in content]
            while source and source[-1] == "\n":
                source.pop()
            cells.append(
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}
            )

    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync a percent-format .py notebook to a .ipynb file.")
    ap.add_argument("--py", type=Path, required=True, help="Input .py path (must contain '# %%' markers).")
    ap.add_argument(
        "--ipynb",
        type=Path,
        default=None,
        help="Output .ipynb path (default: same stem as --py).",
    )
    args = ap.parse_args()

    py_path = Path(args.py)
    ipynb_path = Path(args.ipynb) if args.ipynb is not None else py_path.with_suffix(".ipynb")

    cells = _parse_percent_cells(py_path)

    if ipynb_path.exists():
        old = json.loads(ipynb_path.read_text(encoding="utf-8"))
        metadata = old.get("metadata", {})
        nbformat = old.get("nbformat", 4)
        nbformat_minor = old.get("nbformat_minor", 5)
    else:
        metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
        nbformat = 4
        nbformat_minor = 5

    out = {"cells": cells, "metadata": metadata, "nbformat": nbformat, "nbformat_minor": nbformat_minor}
    ipynb_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {ipynb_path} (cells={len(cells)})")


if __name__ == "__main__":
    main()

