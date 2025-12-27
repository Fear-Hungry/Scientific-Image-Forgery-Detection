# %% [markdown]
# # Fase 3 — Pipeline completo (treino + inferência + submissão)
#
# Este notebook é um "guia executável" para:
# 1) Treinar um baseline de segmentação (opcional),
# 2) Validar com oF1 (opcional, depende de `scipy`),
# 3) Rodar inferência no `test_images/` e gerar `submission.csv`.
#
# **Modo Kaggle (Code Competition)**
# - Internet: OFF no momento da submissão.
# - Tempo típico: até 4h.
# - Saída esperada: `submission.csv` (ou `submission.parquet`).
#
# **Importante**
# - Por padrão, este notebook usa o código do projeto em `src/forgeryseg/`.
# - No Kaggle, a forma mais prática é adicionar este repositório (código + pesos)
#   como um *Dataset* e o notebook automaticamente encontra e adiciona no `sys.path`.
#
# ---

# %%
# Célula 1 — Regras do Kaggle (sanidade)
print("Kaggle submission constraints (lembrete):")
print("- Submissions via Notebook")
print("- Runtime <= 4h (CPU/GPU)")
print("- Internet: OFF no submit")
print("- Output: submission.csv ou submission.parquet")

# %%
# Célula 2 — Imports + ambiente
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

print("python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SEED = 42
set_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Célula 3 — Localizar o código do repo (Kaggle/local) e habilitar imports


def find_project_root() -> Path:
    """
    Encontra um diretório que contenha `src/forgeryseg/`.
    - Local: procura no CWD e ancestrais.
    - Kaggle: procura também em `/kaggle/input/*`.
    """
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "src" / "forgeryseg").exists():
            return p

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for ds in sorted(kaggle_input.glob("*")):
            if (ds / "src" / "forgeryseg").exists():
                return ds
            for sub in sorted(ds.glob("*")):
                if (sub / "src" / "forgeryseg").exists():
                    return sub

    raise FileNotFoundError("Não achei `src/forgeryseg/` (adicione o repo como Dataset no Kaggle ou rode no repo local).")


PROJECT_ROOT = find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_ROOT:", SRC_ROOT)

from forgeryseg.augment import get_train_augment, get_val_augment
from forgeryseg.dataset import PatchDataset, build_test_index, build_train_index, load_image, load_mask_instances
from forgeryseg.inference import predict_image
from forgeryseg.losses import BCEDiceLoss, BCETverskyLoss
from forgeryseg.metric import score_image
from forgeryseg.postprocess import binarize
from forgeryseg.rle import encode_instances
from forgeryseg.train import train_one_epoch, validate

# %%
# Célula 4 — Paths do dataset (Kaggle/local) + config

KAGGLE_COMP_DATASET = Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection")
if (KAGGLE_COMP_DATASET / "train_images").exists():
    DATA_ROOT = KAGGLE_COMP_DATASET
else:
    DATA_ROOT = PROJECT_ROOT / "data" / "recodai"

OUTPUT_ROOT = Path("/kaggle/working/outputs") if is_kaggle() else (PROJECT_ROOT / "outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline_fpn_convnext.json"
cfg: dict = {}
if CONFIG_PATH.exists():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

print("DATA_ROOT:", DATA_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("Config loaded:", bool(cfg))

# %%
# Célula 5 — Index (train/test) + contagens

train_samples = build_train_index(DATA_ROOT)
test_samples = build_test_index(DATA_ROOT)

print("Train samples:", len(train_samples))
print("Test samples:", len(test_samples))
print("Train authentic:", sum(1 for s in train_samples if s.is_authentic))
print("Train forged:", sum(1 for s in train_samples if s.is_authentic is False))

# %% [markdown]
# ## Análise dos Dados e Pré-processamento
#
# Antes de treinar modelos, é útil realizar uma exploração dos dados. Cada imagem possui um identificador `case_id`.
# No dataset de treino, existe um subconjunto de imagens **autênticas** (sem manipulação) e imagens **forjadas**
# (copy-move). Para as imagens forjadas, há uma máscara de segmentação indicando os pixels duplicados; para imagens
# autênticas, não há máscara (equivalente a "nenhum pixel forjado").
#
# **Observação importante (treino):** no snapshot do Kaggle, o mesmo `case_id` pode aparecer em **`authentic/` e
# `forged/`** (duas imagens diferentes). Para indexar sem colisões, use o caminho relativo (`rel_path`) ou uma chave
# composta (ex.: `f\"{label}/{case_id}\"`).
#
# O que precisamos construir:
#
# - **Segmentação:** pares `(imagem, máscara)` (aqui usamos a **união** das instâncias como máscara binária, e depois
#   recuperamos instâncias via componentes conexos na hora do `submission`).
# - **Classificação (opcional):** rótulo binário `y_cls` para decidir se é `authentic` (0) ou `forged` (1).
#
# Pré-processamento (baseline deste repo):
#
# - Leitura com PIL e conversão para **RGB**.
# - Conversão para `float32`, escala para `[0, 1]` e **normalização ImageNet**.
# - Treino *patch-based* (`PatchDataset`): amostra crops de tamanho `patch_size`, com *oversampling* de regiões positivas
#   em imagens forjadas (controlado por `positive_prob`/`min_pos_pixels`).
# - Inferência em imagem inteira via **tiling** (`tile_size`/`overlap`) para lidar com imagens grandes.
#
# ### Dimensionamento e formato (decisão do baseline)
#
# - **Canais:** padronizamos todas as imagens para **3 canais (RGB)**. Se a imagem for originalmente em escala de cinza,
#   duplicamos o canal (via `PIL.Image.convert("RGB")`), o que funciona bem para *backbones* pré-treinados em ImageNet.
# - **Tamanho no treino:** ao invés de fazer *downscale* agressivo da figura inteira (que pode apagar falsificações
#   pequenas), treinamos com **patches 512×512** (crop) e fazemos **padding** quando a imagem é menor. Isso fixa o shape
#   de entrada e mantém detalhes locais.
# - **Tamanho na inferência:** rodamos em **tiles** (ex.: `tile_size=1024`, `overlap=128`) para preservar resolução em
#   imagens grandes. Se o runtime ficar inviável, use `MAX_SIZE` para limitar o lado maior (trade-off controlado).
#
# ### Normalização (decisão do baseline)
#
# - **Escala:** convertemos para `float32` e, quando a imagem vem em `uint8`, reescalamos para **[0, 1]**.
# - **Padronização por canal:** aplicamos **média/desvio do ImageNet** (o padrão esperado por encoders pré-treinados).
# - **Sem equalização fixa:** não aplicamos equalização/histogram matching como pré-processamento determinístico para não
#   correr o risco de mascarar/alterar evidências sutis de copy-move. Em vez disso, lidamos com variações de contraste
#   via **data augmentation** (ex.: `RandomBrightnessContrast`, `RandomGamma`, `CLAHE`) e pela robustez do modelo.
#
# ### Divisão de dados (decisão do baseline: 5-fold CV + ensemble)
#
# Em *code competitions*, o conjunto de teste real é **oculto** e não existe um "val set oficial" fixo. Para
# desenvolvimento local, precisamos criar uma validação a partir do treino:
#
# - **Opção simples:** holdout (ex.: 80/20 estratificado).
# - **Opção de performance:** **K-fold cross-validation** (ex.: 5-fold), treinando 5 modelos e fazendo **ensemble** na
#   inferência. Isso melhora o uso do treino e tende a reduzir overfitting, mas custa ~5× mais tempo de treino.
#
# **Escolha aqui:** usamos **5 folds** e fazemos **ensemble** dos modelos finais.
# Para evitar vazamento, fazemos o split **agrupando por `case_id`** (quando existe par `authentic/` e `forged/` com o
# mesmo id, eles caem no mesmo fold).
#

# %%
# Célula 5b — Exemplo: carregando (imagem, máscara) e label binário
sample0 = train_samples[0]
image0 = load_image(sample0.image_path)
gt_instances0 = load_mask_instances(sample0.mask_path) if sample0.mask_path else []

if gt_instances0:
    union_mask0 = np.max(np.stack(gt_instances0, axis=0), axis=0).astype(np.uint8)
else:
    union_mask0 = np.zeros(image0.shape[:2], dtype=np.uint8)

y_cls0 = 0 if sample0.is_authentic else 1

print("case_id:", sample0.case_id)
print("label:", sample0.label, "| y_cls:", y_cls0)
print("image shape:", image0.shape, "dtype:", image0.dtype)
print("instances:", len(gt_instances0), "| union mask sum:", int(union_mask0.sum()))

# %%
# Célula 5c — EDA rápida (opcional): tamanhos e áreas de máscara
RUN_EDA = False  # deixe False no submit; True para explorar interativamente

if RUN_EDA:
    import pandas as pd
    from IPython.display import display
    from PIL import Image

    rows = []
    for s in train_samples:
        width = height = None
        mode = None
        with Image.open(s.image_path) as img:
            width, height = img.size
            mode = img.mode

        mask_instances = 0
        mask_area = 0
        mask_area_frac = 0.0
        if s.mask_path is not None:
            masks = np.load(s.mask_path)
            if masks.ndim == 2:
                masks = masks[None, ...]
            mask_instances = int(masks.shape[0])
            union = masks.max(axis=0)
            mask_area = int((union > 0).sum())
            mask_area_frac = mask_area / float(width * height)

        rows.append(
            {
                "case_id": s.case_id,
                "label": s.label,
                "rel_path": str(s.rel_path),
                "width": width,
                "height": height,
                "mode": mode,
                "mask_instances": mask_instances,
                "mask_area": mask_area,
                "mask_area_frac": mask_area_frac,
            }
        )

    df_train = pd.DataFrame(rows)
    display(df_train.head())
    display(df_train["label"].value_counts())
    display(df_train["mode"].value_counts())
    print("unique case_id:", int(df_train["case_id"].nunique()))
    print("duplicated case_id (train):", int(df_train.duplicated("case_id").sum()))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.scatter(df_train["width"], df_train["height"], s=3, alpha=0.25)
        plt.title("Train image sizes (width x height)")
        plt.xlabel("width")
        plt.ylabel("height")
        plt.show()

        plt.figure(figsize=(6, 4))
        df_train[df_train["label"] == "forged"]["mask_area_frac"].hist(bins=40)
        plt.title("Mask area fraction (forged)")
        plt.xlabel("mask_area_frac")
        plt.show()
    except Exception as exc:
        print("plots indisponíveis:", repr(exc))

# %%
# Célula 6 — Split em folds (5-fold) com agrupamento por case_id


def _case_id_groups(samples) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, s in enumerate(samples):
        groups.setdefault(str(s.case_id), []).append(int(idx))
    return groups


def iter_case_id_folds(samples, n_splits: int, seed: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Gera folds garantindo que o mesmo case_id não apareça em treino e validação.
    A estratificação é feita no nível do case_id por "tipo de grupo" (par vs solo),
    o que ajuda a manter a proporção authentic/forged por fold no snapshot deste dataset.
    """
    groups = _case_id_groups(samples)
    case_ids = sorted(groups.keys())
    # 0 = par (authentic+forged), 1 = solo (apenas forged)
    y_group = np.array([0 if len(groups[cid]) >= 2 else 1 for cid in case_ids], dtype=int)

    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_g, val_g in splitter.split(np.zeros(len(case_ids)), y_group):
            train_idx: list[int] = []
            val_idx: list[int] = []
            for gi in train_g:
                train_idx.extend(groups[case_ids[int(gi)]])
            for gi in val_g:
                val_idx.extend(groups[case_ids[int(gi)]])
            yield np.array(sorted(train_idx), dtype=int), np.array(sorted(val_idx), dtype=int)
        return
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    indices = np.arange(len(case_ids))
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(y_group):
        label_indices = indices[y_group == label]
        rng.shuffle(label_indices)
        for i, idx in enumerate(label_indices):
            folds[i % n_splits].append(int(idx))

    for fold_idx in range(n_splits):
        val_g = np.array(sorted(folds[fold_idx]), dtype=int)
        val_g_set = set(val_g.tolist())
        train_g = np.array([int(i) for i in indices if int(i) not in val_g_set], dtype=int)

        train_idx: list[int] = []
        val_idx: list[int] = []
        for gi in train_g:
            train_idx.extend(groups[case_ids[int(gi)]])
        for gi in val_g:
            val_idx.extend(groups[case_ids[int(gi)]])

        yield np.array(sorted(train_idx), dtype=int), np.array(sorted(val_idx), dtype=int)


N_FOLDS = int(cfg.get("folds", 5)) if cfg else 5
FOLD = 0

folds = list(iter_case_id_folds(train_samples, n_splits=N_FOLDS, seed=SEED))
train_idx, val_idx = folds[FOLD]

train_fold_samples = [train_samples[int(i)] for i in train_idx]
val_fold_samples = [train_samples[int(i)] for i in val_idx]

print(f"fold {FOLD}/{N_FOLDS}: train={len(train_fold_samples)} val={len(val_fold_samples)}")
print("val forged:", sum(1 for s in val_fold_samples if s.is_authentic is False))

# %%
# Célula 7 — Config de treino (patch-based)
from torch.utils.data import DataLoader, WeightedRandomSampler

PATCH_SIZE = int(cfg.get("patch_size", 512)) if cfg else 512
BATCH_SIZE = int(cfg.get("batch_size", 8)) if cfg else 8
NUM_WORKERS = int(cfg.get("num_workers", 2)) if cfg else 2

POSITIVE_PROB = float(cfg.get("positive_prob", 0.7)) if cfg else 0.7
MIN_POS_PIXELS = int(cfg.get("min_pos_pixels", 32)) if cfg else 32
MAX_TRIES = int(cfg.get("max_tries", 10)) if cfg else 10
POS_SAMPLE_WEIGHT = float(cfg.get("pos_sample_weight", 2.0)) if cfg else 2.0

# %% [markdown]
# ## Data Augmentation (Aumento de Dados)
#
# Este baseline usa **Albumentations** para aplicar aumentos **coerentes** entre `image` e `mask` (para geometria),
# e aumentos **apenas na imagem** (para ruído/cor/blur).
#
# Geometria (image + mask):
#
# - **Flips** horizontal/vertical
# - **Rotação 90°** aleatória e **pequenas rotações** (Affine)
# - **Escala/zoom e translação** (Affine + RandomResizedCrop)
#
# Robustez fotométrica (apenas image):
#
# - **Brilho/contraste**, **gamma** e **CLAHE** (leve)
# - **Ruído gaussiano** e **blur** (gauss/motion)
# - **Compressão** (artefatos tipo JPEG) e **cutout**
#
# Copy-move sintético (image + mask):
#
# - Para patches com máscara vazia (amostra autêntica), aplicamos um **copy-move on-the-fly**:
#   copiamos uma região e colamos em outra posição no mesmo patch, marcando **origem e destino** na máscara.
#   Opcionalmente aplicamos pequena rotação/escala no patch colado.
#

# %%
# Célula 7b — Config do augmentation (inclui copy-move sintético)
COPY_MOVE_PROB = float(cfg.get("copy_move_prob", 0.25)) if cfg else 0.25
COPY_MOVE_MIN_AREA_FRAC = float(cfg.get("copy_move_min_area_frac", 0.05)) if cfg else 0.05
COPY_MOVE_MAX_AREA_FRAC = float(cfg.get("copy_move_max_area_frac", 0.20)) if cfg else 0.20
COPY_MOVE_ROTATION_LIMIT = float(cfg.get("copy_move_rotation_limit", 15.0)) if cfg else 15.0

scale_range = cfg.get("copy_move_scale_range", [0.9, 1.1]) if cfg else [0.9, 1.1]
if isinstance(scale_range, (list, tuple)) and len(scale_range) == 2:
    COPY_MOVE_SCALE_RANGE = (float(scale_range[0]), float(scale_range[1]))
else:
    COPY_MOVE_SCALE_RANGE = (0.9, 1.1)

try:
    train_aug = get_train_augment(
        patch_size=PATCH_SIZE,
        copy_move_prob=COPY_MOVE_PROB,
        copy_move_min_area_frac=COPY_MOVE_MIN_AREA_FRAC,
        copy_move_max_area_frac=COPY_MOVE_MAX_AREA_FRAC,
        copy_move_rotation_limit=COPY_MOVE_ROTATION_LIMIT,
        copy_move_scale_range=COPY_MOVE_SCALE_RANGE,
    )
    val_aug = get_val_augment()
except ImportError as exc:
    print("albumentations indisponível; desativando augmentations de treino. Motivo:", repr(exc))
    train_aug = None
    val_aug = None

def make_loaders(train_samples_fold, val_samples_fold, *, train_aug, val_aug):
    train_ds = PatchDataset(
        train_samples_fold,
        patch_size=PATCH_SIZE,
        train=True,
        augment=train_aug,
        positive_prob=POSITIVE_PROB,
        min_pos_pixels=MIN_POS_PIXELS,
        max_tries=MAX_TRIES,
        seed=SEED,
    )
    val_ds = PatchDataset(
        val_samples_fold,
        patch_size=PATCH_SIZE,
        train=False,
        augment=val_aug,
        seed=SEED,
    )

    weights = [POS_SAMPLE_WEIGHT if s.is_authentic is False else 1.0 for s in train_samples_fold]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_samples_fold), replacement=True)

    train_loader_fold = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
    )
    val_loader_fold = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
    )
    return train_loader_fold, val_loader_fold

# %%
# Célula 8 — Modelo (SMP FPN+ConvNeXt se disponível; fallback: torchvision DeepLabV3)
import torch.nn as nn


def build_fallback_deeplab(pretrained: bool = True) -> nn.Module:
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    base = deeplabv3_resnet50(weights=weights)
    in_ch = int(base.classifier[-1].in_channels)
    base.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)
    base.aux_classifier = None

    class Wrapper(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.model(x)
            if isinstance(out, dict):
                return out["out"]
            return out

    return Wrapper(base)


def build_model_auto() -> nn.Module:
    encoder_name = cfg.get("encoder_name", "convnext_tiny") if cfg else "convnext_tiny"
    encoder_weights = cfg.get("encoder_weights", "imagenet") if cfg else "imagenet"
    if encoder_weights == "":
        encoder_weights = None
    try:
        from forgeryseg.models.fpn_convnext import build_model as build_fpn

        print("backend: segmentation_models_pytorch (FPN)")
        try:
            return build_fpn(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
        except Exception as exc:
            # Em runtime sem internet (ex.: Kaggle submit), baixar pesos pode falhar.
            print("FPN com pesos falhou; tentando sem pesos. Motivo:", repr(exc))
            return build_fpn(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
    except Exception as exc:
        print("backend fallback: torchvision (DeepLabV3) — motivo:", repr(exc))
        try:
            return build_fallback_deeplab(pretrained=True)
        except Exception as exc2:
            print("DeepLabV3 com pesos falhou; tentando sem pesos. Motivo:", repr(exc2))
            return build_fallback_deeplab(pretrained=False)


print("model builder ready")

# %%
# Célula 9 — Loss, otimizador e loop de treino (opcional)

LOSS_NAME = (cfg.get("loss", "bce_dice") if cfg else "bce_dice").lower()
LR = float(cfg.get("learning_rate", 1e-4)) if cfg else 1e-4
WEIGHT_DECAY = float(cfg.get("weight_decay", 1e-4)) if cfg else 1e-4
EPOCHS = int(cfg.get("epochs", 10)) if cfg else 10
USE_AMP = bool(cfg.get("use_amp", True)) if cfg else True
USE_AMP = USE_AMP and DEVICE.startswith("cuda")

fold_dir = OUTPUT_ROOT / "models" / f"fold_{FOLD}"
fold_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = fold_dir / "best.pt"

RUN_TRAIN = False  # mude para True para treinar aqui
TRAIN_FOLDS = [FOLD]  # para CV, use: list(range(N_FOLDS))

if RUN_TRAIN:
    for fold_id in TRAIN_FOLDS:
        train_idx, val_idx = folds[int(fold_id)]
        train_samples_fold = [train_samples[int(i)] for i in train_idx]
        val_samples_fold = [train_samples[int(i)] for i in val_idx]
        train_loader_fold, val_loader_fold = make_loaders(
            train_samples_fold,
            val_samples_fold,
            train_aug=train_aug,
            val_aug=val_aug,
        )

        fold_dir = OUTPUT_ROOT / "models" / f"fold_{int(fold_id)}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = fold_dir / "best.pt"

        model = build_model_auto().to(DEVICE)
        if LOSS_NAME == "bce_tversky":
            criterion = BCETverskyLoss(
                alpha=float(cfg.get("tversky_alpha", 0.7)) if cfg else 0.7,
                beta=float(cfg.get("tversky_beta", 0.3)) if cfg else 0.3,
                tversky_weight=float(cfg.get("tversky_weight", 1.0)) if cfg else 1.0,
            )
        else:
            criterion = BCEDiceLoss(dice_weight=float(cfg.get("dice_weight", 1.0)) if cfg else 1.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_loss = float("inf")
        for epoch in range(1, EPOCHS + 1):
            train_stats = train_one_epoch(model, train_loader_fold, criterion, optimizer, DEVICE, use_amp=USE_AMP)
            val_stats, val_dice = validate(model, val_loader_fold, criterion, DEVICE)
            print(
                f"fold {int(fold_id)} epoch {epoch:02d}/{EPOCHS} "
                f"train_loss={train_stats.loss:.4f} val_loss={val_stats.loss:.4f} val_dice={val_dice:.4f}"
            )

            if val_stats.loss < best_loss:
                best_loss = val_stats.loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_stats.loss,
                        "config": cfg,
                    },
                    ckpt_path,
                )
        print("saved:", ckpt_path)

        del model, optimizer
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

print("checkpoint exists:", ckpt_path.exists(), str(ckpt_path))

# %%
# Célula 10 — Carregar checkpoint (necessário para inferência/submissão)

def _load_checkpoint(path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("config", {})
    return ckpt, {}


def _ckpt_path_for_fold(fold: int) -> Path:
    return (OUTPUT_ROOT / "models" / f"fold_{fold}") / "best.pt"


USE_ENSEMBLE = True
INFER_FOLDS = list(range(N_FOLDS)) if USE_ENSEMBLE else [FOLD]

models: list[nn.Module] = []
loaded_folds: list[int] = []
missing_folds: list[int] = []

for fold_id in INFER_FOLDS:
    fold_ckpt = _ckpt_path_for_fold(fold_id)
    if not fold_ckpt.exists():
        missing_folds.append(int(fold_id))
        continue

    m = build_model_auto()
    state, _ = _load_checkpoint(fold_ckpt)
    m.load_state_dict(state)
    m.to(DEVICE)
    m.eval()
    models.append(m)
    loaded_folds.append(int(fold_id))

print("loaded folds:", loaded_folds)
if missing_folds:
    print("missing folds:", missing_folds)

# %%
# Célula 11 — Pós-processamento (componentes conexos) com fallback se faltar SciPy
try:
    from forgeryseg.postprocess import extract_components as extract_components_impl

    _cc_backend = "scipy"
except Exception:
    extract_components_impl = None
    _cc_backend = "none"


def extract_components_safe(mask: np.ndarray, min_area: int = 0) -> list[np.ndarray]:
    if extract_components_impl is not None:
        return extract_components_impl(mask, min_area=min_area)

    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.max() == 0:
        return []
    num_labels, labels = cv2.connectedComponents(m, connectivity=4)
    instances: list[np.ndarray] = []
    for idx in range(1, int(num_labels)):
        comp = (labels == idx)
        if min_area and int(comp.sum()) < int(min_area):
            continue
        instances.append(comp.astype(np.uint8))
    return instances


print("connected components backend:", _cc_backend, "(fallback=opencv)")

# %%
# Célula 12 — Validação em imagem inteira (oF1) + tuning simples de threshold (opcional)

RUN_VAL_FULL = False  # mude para True para validar com oF1 em imagem inteira

TILE_SIZE = 1024
OVERLAP = 128
MAX_SIZE = 0  # se quiser, defina ex.: 2048 para reduzir custo

THRESHOLD = 0.5
MIN_AREA = 32
VAL_LIMIT = 200  # limite para não ficar gigante


def predict_prob_ensemble(image: np.ndarray) -> np.ndarray:
    if not models:
        raise RuntimeError("Sem modelos carregados (nenhum checkpoint encontrado em outputs/models/fold_*/best.pt).")

    prob_sum: np.ndarray | None = None
    for m in models:
        prob = predict_image(m, image, DEVICE, tile_size=TILE_SIZE, overlap=OVERLAP, max_size=MAX_SIZE)
        if prob_sum is None:
            prob_sum = prob.astype(np.float32, copy=False)
        else:
            prob_sum += prob.astype(np.float32, copy=False)
    return prob_sum / float(len(models))


def predict_instances_for_sample(sample, threshold: float, min_area: int) -> list[np.ndarray]:
    image = load_image(sample.image_path)
    prob = predict_prob_ensemble(image)
    bin_mask = binarize(prob, threshold=threshold)
    return extract_components_safe(bin_mask, min_area=min_area)


def mean_of1(samples, threshold: float, min_area: int, limit: int = 0) -> float | None:
    scores: list[float] = []
    for i, sample in enumerate(samples):
        if limit and i >= limit:
            break
        gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
        pred_instances = predict_instances_for_sample(sample, threshold=threshold, min_area=min_area)
        try:
            scores.append(float(score_image(gt_instances, pred_instances)))
        except Exception as exc:
            print("metric indisponível (provável falta de scipy):", repr(exc))
            return None
    if not scores:
        return None
    return float(np.mean(scores))


if RUN_VAL_FULL and models:
    score = mean_of1(val_fold_samples, threshold=THRESHOLD, min_area=MIN_AREA, limit=VAL_LIMIT)
    print("val mean oF1:", score)

    # grid simples (faixa curta; ajuste conforme custo)
    thresholds = [0.3, 0.4, 0.5, 0.6]
    min_areas = [0, 16, 32, 64, 128]
    best = (None, None, -1.0)
    for t in thresholds:
        for a in min_areas:
            s = mean_of1(val_fold_samples, threshold=t, min_area=a, limit=VAL_LIMIT)
            if s is None:
                break
            print(f"t={t:.2f} a={a:4d} -> {s:.5f}")
            if s > best[2]:
                best = (t, a, s)
    print("best:", best)

# %%
# Célula 13 — Gerar submission.csv (test)

SUBMISSION_PATH = Path("/kaggle/working/submission.csv") if is_kaggle() else (OUTPUT_ROOT / "submission.csv")

RUN_SUBMISSION = True  # deixe True no Kaggle submit

if RUN_SUBMISSION:
    if not models:
        print("Sem checkpoint(s) para inferência. Treine e salve em outputs/models/fold_*/best.pt (ou use scripts/).")
    else:
        SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SUBMISSION_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
            writer.writeheader()
            for sample in test_samples:
                instances = predict_instances_for_sample(sample, threshold=THRESHOLD, min_area=MIN_AREA)
                annotation = encode_instances(instances)
                writer.writerow({"case_id": sample.case_id, "annotation": annotation})

        print("wrote:", SUBMISSION_PATH)

# %%
# Célula 14 — Preview do CSV
try:
    import pandas as pd

    from IPython.display import display

    if not SUBMISSION_PATH.exists():
        print("submission ainda não foi gerada:", SUBMISSION_PATH)
    else:
        df_sub = pd.read_csv(SUBMISSION_PATH)
        display(df_sub.head())
        print("rows:", len(df_sub))
except Exception as exc:
    print("preview indisponível:", repr(exc))
