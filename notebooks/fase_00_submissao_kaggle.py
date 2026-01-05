from __future__ import annotations

# %% [markdown]
# # Recod.ai/LUC — Submission (Kaggle)
#
# Gera `submission.csv` no formato da competição a partir de 1+ configs em `configs/*.json`,
# com suporte a:
#
# - Segmentação (DINOv2) + pós-processamento + TTA (via config)
# - `fft_gate` (opcional) para revisar casos `authentic`
# - `dinov2_freq_fusion` (opcional) via `model_type`
# - Ensemble de múltiplas submissões (opcional)
#
# **No Kaggle**:
# 1. Anexe o dataset da competição.
# 2. (Opcional) Anexe um dataset com seus checkpoints em `outputs/models/*.pth`.
# 3. Rode todas as células; o arquivo final fica em `/kaggle/working/submission.csv`.
#
# Observação: por regra do repo, a lógica nasce aqui (`.py`) e é espelhada no `.ipynb`.
# %%
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

print(f"python={sys.version.split()[0]} platform={platform.platform()}")
print(f"torch={torch.__version__}")
try:
    import torchvision  # type: ignore

    print(f"torchvision={torchvision.__version__}")
except Exception as e:
    print(f"[warn] torchvision not available ({type(e).__name__}: {e})")

try:
    import timm  # type: ignore

    print(f"timm={timm.__version__}")
except Exception as e:
    print(f"[warn] timm not available ({type(e).__name__}: {e})")

try:
    import cv2  # type: ignore

    print(f"opencv={cv2.__version__}")
except Exception as e:
    print(f"[warn] opencv not available ({type(e).__name__}: {e})")

print(f"numpy={np.__version__} pandas={pd.__version__}")


def _find_code_root() -> Path:
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "src" / "forgeryseg").exists():
            return p

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if not d.is_dir():
                continue
            if (d / "src" / "forgeryseg").exists():
                return d
            # common: dataset root contains a single folder with the repo inside
            try:
                for child in d.iterdir():
                    if child.is_dir() and (child / "src" / "forgeryseg").exists():
                        return child
            except PermissionError:
                continue

    raise FileNotFoundError(
        "Não encontrei o código (src/forgeryseg). "
        "No Kaggle: anexe um Dataset contendo este repo (com pastas src/ e configs/)."
    )


CODE_ROOT = _find_code_root()
SRC = CODE_ROOT / "src"
CONFIG_ROOT = CODE_ROOT / "configs"
print(f"code_root={CODE_ROOT}")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forgeryseg.ensemble_io import ensemble_submissions_from_csvs
from forgeryseg.eval import score_submission_csv, validate_submission_format
from forgeryseg.submission import write_submission_csv

# %%
# -------------------------
# Config (edite aqui)
# -------------------------

DATA_ROOT: Path | None = None  # None => auto-detect (Kaggle -> local)
SPLIT = "test"  # "test" no Kaggle (train/supplemental só para debug)
LIMIT = 0  # 0 = sem limite
SKIP_MISSING_CONFIGS = True  # se faltar config/ckpt, pula ao invés de quebrar

# 1+ configs para gerar submissões individuais
BASE_CONFIG_PATHS = [
    # Base + TTA forte (bom custo/benefício)
    CONFIG_ROOT / "dino_v3_518_r69_fft_gate_tta_plus.json",
    # Multi-escala (melhor para regiões pequenas + grandes)
    CONFIG_ROOT / "dino_v4_518_r69_multiscale_fft_gate_tta_plus.json",
    # Fusão espacial+frequência (diversidade)
    CONFIG_ROOT / "dino_v3_518_r69_freq_fusion_fft_gate_tta_plus.json",
]

# Se existir `configs/tuned_<stem>_optuna_<objective>.json` (vindo do notebook de treino),
# troca automaticamente pelo tuned (melhor score no val subset).
AUTO_USE_TUNED = True
TUNED_OBJECTIVE = "combo"  # mean_score | mean_forged | combo

# ensemble (opcional) se CONFIG_PATHS tiver 2+
DO_ENSEMBLE = len(BASE_CONFIG_PATHS) > 1
ENSEMBLE_METHOD = "weighted"  # weighted | majority | union | intersection
ENSEMBLE_THRESHOLD = 0.5  # só para method="weighted"

# Se quiser pesos fixos, preencha WEIGHTS (mesmo tamanho de CONFIG_PATHS).
# Caso contrário, se SCORES for fornecido, os pesos são derivados automaticamente (melhor score => maior peso).
WEIGHTS: list[float] | None = None
SCORES: list[float] | None = None
AUTO_SCORES_FROM_CKPT = True  # usa `val_of1` do checkpoint para derivar pesos (se disponível)

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
FINAL_OUT = OUT_DIR / "submission.csv"

# %%


def _find_recodai_root() -> Path:
    if DATA_ROOT is not None:
        return Path(DATA_ROOT)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if not d.is_dir():
                continue
            if (d / "recodai" / "sample_submission.csv").exists():
                return d / "recodai"
            if (d / "sample_submission.csv").exists() and (d / "test_images").exists():
                return d

    local = Path("data/recodai")
    if local.exists():
        return local
    local2 = CODE_ROOT / "data" / "recodai"
    if local2.exists():
        return local2

    raise FileNotFoundError(
        "Não encontrei o data root. Defina DATA_ROOT manualmente "
        "(ex.: /kaggle/input/<dataset>/recodai ou data/recodai)."
    )


# %%
# -------------------------
# Run
# -------------------------

data_root = _find_recodai_root()
print(f"data_root={data_root}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True

from forgeryseg.paths import resolve_existing_path


def _resolve_config_path(p: Path) -> Path:
    if p.exists():
        return p
    rel = Path("configs") / p.name
    return resolve_existing_path(rel, roots=[CODE_ROOT], search_kaggle_input=True)


def _maybe_use_tuned(base: Path) -> Path:
    base = _resolve_config_path(base)
    if not bool(AUTO_USE_TUNED):
        return base
    tuned_name = f"tuned_{base.stem}_optuna_{TUNED_OBJECTIVE}.json"
    tuned_rel = Path("configs") / tuned_name
    tuned = resolve_existing_path(tuned_rel, roots=[CODE_ROOT], search_kaggle_input=True)
    return tuned if tuned.exists() else base


CONFIG_PATHS = [_maybe_use_tuned(p) for p in BASE_CONFIG_PATHS]
print(f"configs={[p.as_posix() for p in CONFIG_PATHS]}")

sub_paths: list[Path] = []
sub_scores: list[float] = []
for cfg_path in CONFIG_PATHS:
    if not cfg_path.exists():
        msg = f"[warn] Config não encontrado: {cfg_path}"
        if SKIP_MISSING_CONFIGS:
            print(msg)
            continue
        raise FileNotFoundError(msg)

    cfg = json.loads(cfg_path.read_text())
    name = str(cfg.get("name", cfg_path.stem))
    out_path = OUT_DIR / f"submission_{name}.csv"
    try:
        write_submission_csv(
            config_path=cfg_path,
            data_root=data_root,
            split=SPLIT,  # type: ignore[arg-type]
            out_path=out_path,
            device=device,
            limit=LIMIT,
            path_roots=[OUT_DIR, Path.cwd(), CODE_ROOT, CONFIG_ROOT],
            amp=True,
        )
        sub_paths.append(out_path)

        ckpt = None
        try:
            ckpt_rel = cfg.get("model", {}).get("checkpoint")
            if isinstance(ckpt_rel, str) and ckpt_rel.strip():
                ckpt = resolve_existing_path(ckpt_rel, roots=[OUT_DIR, CODE_ROOT], search_kaggle_input=True)
        except Exception:
            ckpt = None

        if ckpt is not None and ckpt.exists():
            try:
                d = torch.load(ckpt, map_location="cpu")
                v = d.get("val_of1", None)
                if isinstance(v, (int, float)):
                    sub_scores.append(float(v))
                else:
                    sub_scores.append(float("nan"))
            except Exception:
                sub_scores.append(float("nan"))
        else:
            sub_scores.append(float("nan"))
    except FileNotFoundError as e:
        if SKIP_MISSING_CONFIGS:
            print(f"[warn] {e} (pulando {cfg_path.name})")
            continue
        raise

if not sub_paths:
    raise RuntimeError("Nenhuma submissão foi gerada (verifique configs/checkpoints).")

if DO_ENSEMBLE and len(sub_paths) > 1:
    if (
        SCORES is None
        and WEIGHTS is None
        and AUTO_SCORES_FROM_CKPT
        and len(sub_scores) == len(sub_paths)
        and all(not (s != s) for s in sub_scores)  # not NaN
    ):
        SCORES = list(sub_scores)
        print(f"[info] Using SCORES from checkpoints: {SCORES}")

    ensemble_submissions_from_csvs(
        sub_paths=sub_paths,
        data_root=data_root,
        split=SPLIT,
        out_path=FINAL_OUT,
        method=str(ENSEMBLE_METHOD),
        weights=WEIGHTS,
        scores=SCORES,
        threshold=float(ENSEMBLE_THRESHOLD),
    )
else:
    # 1 config => apenas renomeia como submission.csv
    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    if sub_paths:
        Path(sub_paths[0]).replace(FINAL_OUT)
    print(f"Wrote {FINAL_OUT}")

# %%
# -------------------------
# Score / Sanity-check do submission.csv
# -------------------------
import json
from pathlib import Path

# >>> AJUSTE AQUI <<<
EVAL_CSV = FINAL_OUT  # por padrão usa o submission final gerado
EVAL_SPLIT = "test"  # "train" ou "supplemental" para calcular score; "test" só valida formato
# --------------------

eval_csv = Path(EVAL_CSV)
print("data_root =", data_root)
print("EVAL_CSV  =", eval_csv)
print("EVAL_SPLIT=", EVAL_SPLIT)

fmt = validate_submission_format(eval_csv, data_root=data_root, split=EVAL_SPLIT)  # type: ignore[arg-type]
print("\n[Format check]")
print(json.dumps(fmt, indent=2, ensure_ascii=False))

if EVAL_SPLIT in ("train", "supplemental"):
    print("\n[Local score]")
    try:
        res = score_submission_csv(eval_csv, data_root=data_root, split=EVAL_SPLIT)  # type: ignore[arg-type]
        print(json.dumps(res.as_dict(csv_path=eval_csv, split=EVAL_SPLIT), indent=2, ensure_ascii=False))
    except ImportError as e:
        print("\n[ERRO] Para calcular o oF1, precisa de SciPy (Hungarian matching).")
        print("Detalhe:", e)
else:
    print("\nSplit=test => não há ground truth; não dá para calcular score real aqui.")
    print("Use train/supplemental para score local ou apenas confie no Format check acima.")
