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
import dataclasses
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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

from forgeryseg.ensemble import ensemble_annotations, rank_weights_by_score
from forgeryseg.submission import list_ordered_cases, write_submission_csv

# %%
# -------------------------
# Config (edite aqui)
# -------------------------

DATA_ROOT: Path | None = None  # None => auto-detect (Kaggle -> local)
SPLIT = "test"  # "test" no Kaggle (train/supplemental só para debug)
LIMIT = 0  # 0 = sem limite
SKIP_MISSING_CONFIGS = True  # se faltar config/ckpt, pula ao invés de quebrar

# 1+ configs para gerar submissões individuais
CONFIG_PATHS = [
    CONFIG_ROOT / "dino_v3_518_r69_fft_gate.json",
    # CONFIG_ROOT / "dino_v2_518_basev1.json",
    # CONFIG_ROOT / "dino_v1_718_u52.json",
    # CONFIG_ROOT / "dino_v3_518_r69_freq_fusion.json",
]

# ensemble (opcional) se CONFIG_PATHS tiver 2+
DO_ENSEMBLE = len(CONFIG_PATHS) > 1
ENSEMBLE_METHOD = "weighted"  # weighted | majority | union | intersection
ENSEMBLE_THRESHOLD = 0.5  # só para method="weighted"

# Se quiser pesos fixos, preencha WEIGHTS (mesmo tamanho de CONFIG_PATHS).
# Caso contrário, se SCORES for fornecido, os pesos vêm de rank_weights_by_score(scores).
WEIGHTS: list[float] | None = None
SCORES: list[float] | None = None

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


def ensemble_submissions_from_csvs(
    *,
    sub_paths: list[Path],
    data_root: Path,
    split: str,
    out_path: Path,
    method: str = "weighted",
    weights: list[float] | None = None,
    scores: list[float] | None = None,
    threshold: float = 0.5,
) -> None:
    tables: list[dict[str, str]] = []
    for p in sub_paths:
        df = pd.read_csv(p)
        if "case_id" not in df.columns or "annotation" not in df.columns:
            raise ValueError(f"{p} precisa ter colunas case_id,annotation")
        tables.append(dict(zip(df["case_id"].astype(str), df["annotation"], strict=True)))

    if method == "weighted":
        if weights is None:
            if scores is None:
                weights = [1.0 / len(sub_paths)] * len(sub_paths)
            else:
                weights = rank_weights_by_score(scores)
        if len(weights) != len(sub_paths):
            raise ValueError("weights precisa ter o mesmo tamanho de sub_paths")
        print(f"ensemble weights={weights}")

    cases = list_ordered_cases(data_root, split)  # type: ignore[arg-type]

    import cv2

    rows: list[dict[str, str]] = []
    for case in tqdm(cases, desc="Ensemble"):
        h, w = cv2.imread(str(case.image_path), cv2.IMREAD_UNCHANGED).shape[:2]
        anns = [t.get(case.case_id, "authentic") for t in tables]
        ann_out = ensemble_annotations(
            anns,
            shape=(h, w),
            method=method,  # type: ignore[arg-type]
            weights=weights if method == "weighted" else None,
            threshold=float(threshold),
        )
        rows.append({"case_id": case.case_id, "annotation": ann_out})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    n_auth = int((out_df["annotation"] == "authentic").sum())
    print(f"Wrote {out_path} ({n_auth}/{len(out_df)} authentic)")


# %%
# -------------------------
# Run
# -------------------------

data_root = _find_recodai_root()
print(f"data_root={data_root}")
print(f"configs={[p.as_posix() for p in CONFIG_PATHS]}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True

sub_paths: list[Path] = []
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
            path_roots=[Path.cwd(), CODE_ROOT],
        )
        sub_paths.append(out_path)
    except FileNotFoundError as e:
        if SKIP_MISSING_CONFIGS:
            print(f"[warn] {e} (pulando {cfg_path.name})")
            continue
        raise

if not sub_paths:
    raise RuntimeError("Nenhuma submissão foi gerada (verifique configs/checkpoints).")

if DO_ENSEMBLE and len(sub_paths) > 1:
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

import numpy as np
import pandas as pd
from tqdm import tqdm

from forgeryseg.dataset import list_cases, load_mask_instances
from forgeryseg.metric import of1_score
from forgeryseg.rle import annotation_to_masks

try:
    from PIL import Image
except Exception:
    Image = None


# >>> AJUSTE AQUI <<<
EVAL_CSV = FINAL_OUT  # por padrão usa o submission final gerado
EVAL_SPLIT = "test"  # "train" ou "supplemental" para calcular score; "test" só valida formato
# --------------------


def _is_authentic(ann) -> bool:
    if ann is None:
        return True
    if isinstance(ann, float) and np.isnan(ann):
        return True
    s = str(ann).strip().lower()
    return (s == "") or (s == "authentic")


def validate_submission_format(csv_path: Path, *, data_root: Path, split: str) -> dict:
    """
    Valida:
      - CSV tem colunas case_id, annotation
      - case_id existe no split
      - annotations decodificam sem erro (usando shape da imagem)
    """
    df = pd.read_csv(csv_path)
    if "case_id" not in df.columns or "annotation" not in df.columns:
        raise ValueError(f"{csv_path} precisa ter colunas: case_id, annotation")

    df["case_id"] = df["case_id"].astype(str)
    if df["case_id"].duplicated().any():
        dup = df.loc[df["case_id"].duplicated(), "case_id"].iloc[:5].tolist()
        raise ValueError(f"{csv_path} tem case_id duplicado (ex.: {dup})")

    pred = dict(zip(df["case_id"], df["annotation"], strict=True))

    cases = list_cases(data_root, split, include_authentic=True, include_forged=True)
    case_by_id = {c.case_id: c for c in cases}

    if split == "test":
        expected_ids = pd.read_csv(data_root / "sample_submission.csv")["case_id"].astype(str).tolist()
    else:
        expected_ids = list(case_by_id.keys())

    missing_in_csv = [cid for cid in expected_ids if cid not in pred]
    extra_in_csv = [cid for cid in pred.keys() if cid not in case_by_id]

    decode_errors = []
    decoded_non_empty = 0

    # validar decodificação usando H/W da imagem
    if Image is None:
        print("[warn] PIL não disponível; pulando validação de decode por shape da imagem.")
    else:
        for cid, ann in tqdm(pred.items(), desc="Validating RLE"):
            if cid not in case_by_id:
                continue
            if _is_authentic(ann):
                continue

            case = case_by_id[cid]
            try:
                with Image.open(case.image_path) as im:
                    w, h = im.size
                masks = annotation_to_masks(ann, (h, w))
                if len(masks) > 0 and any(np.any(m) for m in masks):
                    decoded_non_empty += 1
            except Exception as e:
                decode_errors.append((cid, str(e)))

    return {
        "csv_path": str(csv_path),
        "split": split,
        "n_cases_in_split": len(cases),
        "n_rows_in_csv": len(df),
        "missing_case_ids_in_csv": len(missing_in_csv),
        "extra_case_ids_in_csv": len(extra_in_csv),
        "n_decode_errors": len(decode_errors),
        "n_non_empty_decoded": decoded_non_empty,
        "sample_missing_ids": missing_in_csv[:5],
        "sample_extra_ids": extra_in_csv[:5],
        "sample_decode_errors": decode_errors[:5],
    }


def score_submission(csv_path: Path, *, data_root: Path, split: str) -> dict:
    """
    Score local no estilo do sanity_submissions.py:
      - autêntica: 1 se pred "authentic", senão 0
      - forjada: 0 se pred "authentic", senão oF1(pred_masks, gt_masks)
    """
    df = pd.read_csv(csv_path)
    if "case_id" not in df.columns or "annotation" not in df.columns:
        raise ValueError(f"{csv_path} precisa ter colunas: case_id, annotation")
    df["case_id"] = df["case_id"].astype(str)

    pred = dict(zip(df["case_id"], df["annotation"], strict=True))

    cases = list_cases(data_root, split, include_authentic=True, include_forged=True)

    scores_all = []
    scores_auth = []
    scores_forg = []
    n_auth_pred_as_forged = 0
    n_forg_pred_as_auth = 0
    decode_errors = 0

    for case in tqdm(cases, desc=f"Scoring {split}"):
        ann = pred.get(case.case_id, "authentic")

        # Caso autêntico (sem máscara GT)
        if case.mask_path is None:
            s = 1.0 if _is_authentic(ann) else 0.0
            if s == 0.0:
                n_auth_pred_as_forged += 1
            scores_all.append(s)
            scores_auth.append(s)
            continue

        # Caso forjado (com GT)
        if _is_authentic(ann):
            n_forg_pred_as_auth += 1
            s = 0.0
            scores_all.append(s)
            scores_forg.append(s)
            continue

        gt_masks = load_mask_instances(case.mask_path)
        h, w = gt_masks[0].shape

        try:
            pred_masks = annotation_to_masks(ann, (h, w))
            s = of1_score(pred_masks, gt_masks)
        except ImportError:
            raise
        except Exception:
            decode_errors += 1
            s = 0.0

        scores_all.append(float(s))
        scores_forg.append(float(s))

    def _mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    return {
        "csv_path": str(csv_path),
        "split": split,
        "mean_score": _mean(scores_all),
        "mean_authentic": _mean(scores_auth),
        "mean_forged": _mean(scores_forg),
        "n_cases": len(scores_all),
        "n_authentic": len(scores_auth),
        "n_forged": len(scores_forg),
        "auth_pred_as_forged": int(n_auth_pred_as_forged),
        "forg_pred_as_auth": int(n_forg_pred_as_auth),
        "decode_errors_scoring": int(decode_errors),
    }


# --------- RUN ---------
EVAL_CSV = Path(EVAL_CSV)
print("data_root =", data_root)
print("EVAL_CSV  =", EVAL_CSV)
print("EVAL_SPLIT=", EVAL_SPLIT)

fmt = validate_submission_format(EVAL_CSV, data_root=data_root, split=EVAL_SPLIT)
print("\n[Format check]")
print(json.dumps(fmt, indent=2, ensure_ascii=False))

if EVAL_SPLIT in ("train", "supplemental"):
    print("\n[Local score]")
    try:
        res = score_submission(EVAL_CSV, data_root=data_root, split=EVAL_SPLIT)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except ImportError as e:
        print("\n[ERRO] Para calcular o oF1, precisa de SciPy (Hungarian matching).")
        print("Detalhe:", e)
else:
    print("\nSplit=test => não há ground truth; não dá para calcular score real aqui.")
    print("Use train/supplemental para score local ou apenas confie no Format check acima.")
