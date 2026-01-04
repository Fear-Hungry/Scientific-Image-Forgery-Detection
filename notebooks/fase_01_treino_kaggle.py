from __future__ import annotations

# %% [markdown]
# # Recod.ai/LUC — Training (Kaggle, internet ON)
#
# Este notebook gera os **pesos (`*.pth`)** necessários para rodar a submissão offline:
#
# - Segmentação (DINOv2 + decoder) → `outputs/models/r69.pth`
# - (Opcional) Classificador FFT → `outputs/models/fft_cls.pth`
#
# Fluxo recomendado:
#
# 1. Kaggle Notebook **com internet ON** + **GPU**.
# 2. Anexe o dataset da competição.
# 3. Anexe um dataset com **este repo** (ou clone).
# 4. Rode as células para treinar e salvar os checkpoints em `/kaggle/working/outputs/models/`.
# 5. Empacote um folder `kaggle_bundle/` para criar um Kaggle Dataset com código + pesos.
#
# Observação: por regra do repo, a lógica nasce aqui (`.py`) e é espelhada no `.ipynb`.
#
# %%
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

print(f"python={sys.version.split()[0]} platform={platform.platform()}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")


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

# %%
# -------------------------
# (Opcional) Instalar deps
# -------------------------
#
# No Kaggle, normalmente já existe torch/torchvision. Se faltar timm/albumentations/etc,
# use INSTALL_DEPS=True com internet ON.
INSTALL_DEPS = False

if INSTALL_DEPS:
    req = CODE_ROOT / "requirements-kaggle.txt"
    print(f"Installing: {req}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])

# %%
from forgeryseg.kaggle import package_kaggle_dataset
from forgeryseg.submission import write_submission_csv
from forgeryseg.training.dino_decoder import train_dino_decoder
from forgeryseg.training.fft_classifier import train_fft_classifier

# %%
# -------------------------
# Config (edite aqui)
# -------------------------

DATA_ROOT: Path | None = None  # None => auto-detect (Kaggle -> local)

SEG_TRAIN_CONFIG = CONFIG_ROOT / "dino_v3_518_r69.json"
FFT_TRAIN_CONFIG = CONFIG_ROOT / "fft_classifier_logmag_256.json"

TRAIN_SEG = True
TRAIN_FFT = True

SEG_FOLDS = 1  # use 1 para gerar r69.pth diretamente; >1 cria r69_fold{i}.pth
FFT_FOLDS = 1  # use 1 para gerar fft_cls.pth diretamente; >1 cria fft_cls_fold{i}.pth

SEG_EPOCHS = 5
SEG_BATCH = 4
SEG_LR = 1e-3
SEG_WD = 1e-4
SEG_NUM_WORKERS = 2
SEG_AUG = "robust"  # none | basic | robust
SEG_SCHEDULER = "cosine"  # none | cosine | onecycle
SEG_PATIENCE = 3  # early stopping em val_of1 (0 desliga)

FFT_EPOCHS = 5
FFT_BATCH = 32
FFT_LR = 1e-3
FFT_WD = 1e-4
FFT_NUM_WORKERS = 2
FFT_SCHEDULER = "cosine"  # none | cosine | onecycle

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT_MODELS = OUT_DIR / "outputs" / "models"

SEG_OUT = OUT_MODELS / "r69.pth"
FFT_OUT = OUT_MODELS / "fft_cls.pth"

# (Opcional) checar score local rapidamente após treinar:
EVAL_AFTER_TRAIN = True
EVAL_SPLIT = "train"  # train | supplemental
EVAL_LIMIT = 0  # 0 = sem limite (usa tudo)

# Empacotar um folder pronto para upload como Kaggle Dataset (offline):
DO_PACKAGE = True
PKG_OUT = OUT_DIR / "kaggle_bundle"

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
            if (d / "sample_submission.csv").exists() and (
                (d / "train_images").exists() or (d / "test_images").exists()
            ):
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

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print(f"device={device} (Dica: ative GPU em Settings -> Accelerator)")

OUT_MODELS.mkdir(parents=True, exist_ok=True)

# %%
# -----
# Train (Segmentation)
# -----

seg_result = None
if TRAIN_SEG:
    seg_result = train_dino_decoder(
        config_path=SEG_TRAIN_CONFIG,
        data_root=data_root,
        out_path=SEG_OUT,
        device=device_str,
        split="train",
        epochs=int(SEG_EPOCHS),
        batch_size=int(SEG_BATCH),
        lr=float(SEG_LR),
        weight_decay=float(SEG_WD),
        num_workers=int(SEG_NUM_WORKERS),
        folds=int(SEG_FOLDS),
        fold=None,
        aug=SEG_AUG,  # type: ignore[arg-type]
        scheduler=SEG_SCHEDULER,  # type: ignore[arg-type]
        patience=int(SEG_PATIENCE),
    )

    # Se treinou k-fold, copia o melhor fold para o path "base" (r69.pth),
    # para facilitar o uso em configs que apontam para outputs/models/r69.pth.
    if seg_result is not None and int(SEG_FOLDS) > 1:
        best = max(seg_result.fold_results, key=lambda fr: fr.best_val_of1)
        if best.checkpoint_path != SEG_OUT:
            SEG_OUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best.checkpoint_path, SEG_OUT)
            print(f"Copied best fold checkpoint -> {SEG_OUT} (from {best.checkpoint_path})")

# %%
# -----
# Train (FFT classifier)
# -----

fft_saved = None
if TRAIN_FFT:
    fft_saved = train_fft_classifier(
        config_path=FFT_TRAIN_CONFIG,
        data_root=data_root,
        out_path=FFT_OUT,
        device=device,
        epochs=int(FFT_EPOCHS),
        batch_size=int(FFT_BATCH),
        lr=float(FFT_LR),
        weight_decay=float(FFT_WD),
        num_workers=int(FFT_NUM_WORKERS),
        folds=int(FFT_FOLDS),
        scheduler=FFT_SCHEDULER,  # type: ignore[arg-type]
    )

    if fft_saved and int(FFT_FOLDS) > 1:
        # escolhe melhor fold por menor val_loss no checkpoint
        best_path = min(
            fft_saved,
            key=lambda p: float(torch.load(p, map_location="cpu").get("val_loss", float("inf"))),
        )
        if best_path != FFT_OUT:
            FFT_OUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, FFT_OUT)
            print(f"Copied best FFT fold -> {FFT_OUT} (from {best_path})")

# %%
# -------------------------
# Quick evaluation (local)
# -------------------------
#
# Gera um submission no split train/supplemental e calcula oF1 local.

if EVAL_AFTER_TRAIN:
    from forgeryseg.eval import score_submission_csv, validate_submission_format

    eval_cfg = CONFIG_ROOT / "dino_v3_518_r69_fft_gate.json"
    eval_csv = OUT_DIR / f"submission_{EVAL_SPLIT}.csv"

    stats = write_submission_csv(
        config_path=eval_cfg,
        data_root=data_root,
        split=EVAL_SPLIT,  # type: ignore[arg-type]
        out_path=eval_csv,
        device=device,
        limit=int(EVAL_LIMIT),
        path_roots=[OUT_DIR, CODE_ROOT, CONFIG_ROOT],
    )
    print(stats)

    fmt = validate_submission_format(eval_csv, data_root=data_root, split=EVAL_SPLIT)  # type: ignore[arg-type]
    print("\n[Format check]")
    print(json.dumps(fmt, indent=2, ensure_ascii=False))

    score = score_submission_csv(eval_csv, data_root=data_root, split=EVAL_SPLIT)  # type: ignore[arg-type]
    print("\n[Local score]")
    print(json.dumps(score.as_dict(csv_path=eval_csv, split=EVAL_SPLIT), indent=2, ensure_ascii=False))

# %%
# -------------------------
# Package Kaggle Dataset folder
# -------------------------
#
# Cria um folder pronto para upload como Kaggle Dataset:
# - código (src/scripts/configs/notebooks/docs)
# - + `outputs/models/*.pth` (opcional)
#
# Depois, anexe esse dataset no notebook de submissão (internet OFF).

if DO_PACKAGE:
    out_root = package_kaggle_dataset(
        out_dir=PKG_OUT,
        include_models=True,
        models_dir=OUT_MODELS,
        repo_root=CODE_ROOT,
    )
    print(f"Wrote Kaggle bundle at: {out_root.resolve()}")
    print("Crie um Kaggle Dataset a partir desse folder e anexe no notebook de submissão offline.")
