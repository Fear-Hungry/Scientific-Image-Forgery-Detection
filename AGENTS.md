# Repository Guidelines

## Project Structure & Module Organization

- `src/forgeryseg/` holds the core library (dataset loading, RLE, metrics, training, postprocess).
- `src/forgeryseg/models/` contains model definitions used by training scripts.
- `scripts/` provides CLI entry points for training, prediction, and submission generation.
- `configs/` stores JSON configs used by training and inference.
- `data/` contains the Kaggle snapshot (expected root: `data/recodai`).
- `tests/` hosts pytest-based unit tests; `notebooks/` is for experiments and demos.

## Build, Test, and Development Commands

- `pip install -r requirements.txt` installs runtime and dev dependencies.
- `python scripts/sanity_submissions.py --data-root data/recodai --out-dir outputs/sanity --split train` validates RLE/metric outputs on the train set.
- `python scripts/train_baseline.py --config configs/baseline_fpn_convnext.json --data-root data/recodai --output-dir outputs/models --folds 5` runs the baseline CV training.
- `python scripts/predict_oof.py --data-root data/recodai --output-dir outputs/preds --folds 5 --threshold 0.5 --min-area 32 --tile-size 1024 --overlap 128` generates OOF predictions.
- `pytest -q` runs the unit tests.

## Coding Style & Naming Conventions

- Python, 4-space indentation, and PEP 8-style naming.
- Use `snake_case` for functions/variables and `PascalCase` for classes (see `src/forgeryseg/dataset.py`).
- Prefer `pathlib.Path` for filesystem paths; keep `data_root` arguments aligned with `data/recodai`.
- New features should land in `src/forgeryseg/` with CLI wrappers in `scripts/`.
- Regra de notebook: implemente primeiro em `.py` e somente depois sincronize a mesma lógica no notebook `.ipynb` correspondente.

## Testing Guidelines

- Pytest is the test runner; files follow `tests/test_*.py`.
- `tests/test_metric.py` covers scoring behavior; `scipy` is optional and skipped if missing.
- Add tests for metric, RLE, or post-processing changes, and keep them minimal and deterministic.

## Commit & Pull Request Guidelines

- Git history is not available in this snapshot; use concise, imperative commit subjects with an optional scope (e.g., `scripts: add threshold tuning`).
- PRs should include a short summary, reproduction commands, and any metric deltas.
- If adding data or large artifacts, document provenance and keep generated outputs in `outputs/`.

## Data & Configuration Notes

- The project expects the Kaggle data layout under `data/recodai` (see `README.md`).
- Config files in `configs/` are the source of truth for baseline training runs.
