# %% [markdown]
# Fase 2 — Baseline de Segmentação (Kaggle / Notebook-only)
#
# Objetivo desta etapa:
# - Preparar o dataset (carregar imagens + máscaras)
# - Padronizar entrada (resize + pad) e binarizar máscaras
# - Definir augmentations consistentes (imagem + máscara + área válida)
# - Criar K folds (K=5) para validação

# %%
# Celula 1 — Regras do Kaggle (sanidade)
print("Kaggle submission constraints (lembrete):")
print("- Submissions via Notebook")
print("- Runtime <= 4h (CPU/GPU)")
print("- Internet: OFF")
print("- Output: submission.csv ou submission.parquet")

# %%
# Celula 2 — Imports + ambiente
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from PIL import Image
from tqdm.auto import tqdm

print("Python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("torch:", torch.__version__)
print("opencv:", cv2.__version__)
print("albumentations:", A.__version__)

# %%
# Celula 3 — Paths + seed
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()


def find_dataset_root() -> Path:
    # Kaggle: padrão da competição
    if is_kaggle():
        base = Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection")
        if base.exists():
            return base

        # fallback: procura qualquer dataset anexado que tenha a estrutura esperada
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for ds in sorted(kaggle_input.glob("*")):
                if (ds / "train_images").exists() and (ds / "test_images").exists():
                    return ds

    # Local (repo): `data/`
    base = Path("data").resolve()
    if (base / "train_images").exists() and (base / "test_images").exists():
        return base

    raise FileNotFoundError(
        "Dataset não encontrado.\n"
        "- No Kaggle: anexe o dataset da competição (Add data) e garanta que existe `train_images/`.\n"
        "- Local: espere `data/train_images` e `data/train_masks`."
    )


DATA_ROOT = find_dataset_root()
TRAIN_IMAGES = DATA_ROOT / "train_images"
TRAIN_MASKS = DATA_ROOT / "train_masks"

print("DATA_ROOT:", DATA_ROOT)
print("Train images dir:", TRAIN_IMAGES)
print("Train masks dir:", TRAIN_MASKS)
num_auth = len(list((TRAIN_IMAGES / "authentic").glob("*.png")))
num_forged = len(list((TRAIN_IMAGES / "forged").glob("*.png")))
num_masks = len(list(TRAIN_MASKS.glob("*.npy")))
print("Train/authentic:", num_auth)
print("Train/forged:", num_forged)
print("Masks:", num_masks)
if num_auth == 0 and num_forged == 0:
    raise FileNotFoundError(
        "Nenhuma imagem encontrada em `train_images/authentic` e `train_images/forged`.\n"
        f"DATA_ROOT={DATA_ROOT}\n"
        "No Kaggle, isso normalmente significa que o dataset da competição não foi anexado ao notebook."
    )

# %%
# Celula 4 — Index do dataset (DataFrame principal)
@dataclass(frozen=True)
class Sample:
    case_id: str
    label: str  # authentic/forged
    img_path: Path
    mask_path: Path | None


def build_train_index(train_images_dir: Path, train_masks_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for label in ["authentic", "forged"]:
        for img_path in sorted((train_images_dir / label).glob("*.png")):
            case_id = img_path.stem
            mask_path = train_masks_dir / f"{case_id}.npy" if label == "forged" else None
            file_size = img_path.stat().st_size

            width = height = None
            mode = None
            read_error = None
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
            except Exception as exc:
                read_error = f"{type(exc).__name__}: {exc}"

            mask_instances = 0
            mask_area = 0
            mask_area_frac = 0.0
            if mask_path is not None:
                mask = np.load(mask_path)
                if mask.ndim == 2:
                    mask = mask[None, ...]
                mask_instances = int(mask.shape[0])
                union = mask.max(axis=0)
                mask_area = int((union > 0).sum())
                if width is not None and height is not None:
                    mask_area_frac = mask_area / float(width * height)
                else:
                    mask_area_frac = np.nan

            rows.append(
                {
                    "case_id": case_id,
                    "split": "train",
                    "label": label,
                    "img_path": str(img_path),
                    "mask_path": None if mask_path is None else str(mask_path),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "file_size": file_size,
                    "mask_instances": mask_instances,
                    "mask_area": mask_area,
                    "mask_area_frac": mask_area_frac,
                    "read_error": read_error,
                }
            )
    if not rows:
        raise FileNotFoundError(
            "Index vazio: não encontrei arquivos `*.png`.\n"
            f"train_images_dir={train_images_dir}\n"
            "Verifique se o dataset correto foi anexado no Kaggle (Add data)."
        )
    return pd.DataFrame(rows)


df = build_train_index(TRAIN_IMAGES, TRAIN_MASKS)
display(df.head())
print("Rows:", len(df))
if "label" not in df.columns:
    raise KeyError(f"df não tem coluna 'label'. colunas={df.columns.tolist()}")
display(df["label"].value_counts())
print("Read errors:", int(df["read_error"].notna().sum()))

# %%
# Celula 5 — Estatísticas rápidas (tamanho / área de máscara)
plt.figure(figsize=(6, 4))
plt.scatter(df["width"], df["height"], s=4, alpha=0.3)
plt.title("Image sizes (width x height)")
plt.xlabel("width")
plt.ylabel("height")
plt.show()

plt.figure(figsize=(6, 4))
df[df["label"] == "forged"]["mask_area_frac"].hist(bins=40)
plt.title("Mask area fraction (forged)")
plt.xlabel("mask_area_frac")
plt.show()

# %%
# Celula 6 — Preprocess (resize+pad para 512x512) + máscara binária
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TARGET_SIZE = 512


def load_image_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.array(img)


def load_mask_union(path: str | Path | None, shape_hw: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape_hw, dtype=np.uint8)
    masks = np.load(Path(path))
    if masks.ndim == 2:
        union = masks
    else:
        union = masks.max(axis=0)
    union = (union > 0).astype(np.uint8)
    if union.shape != shape_hw:
        raise ValueError(f"Mask shape {union.shape} does not match image shape {shape_hw}")
    return union


def resize_pad_to_square(
    image: np.ndarray,
    mask: np.ndarray,
    size: int = TARGET_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    scale = size / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image_rs = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_rs = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_h = size - new_h
    pad_w = size - new_w
    image_pad = np.pad(image_rs, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    mask_pad = np.pad(mask_rs, ((0, pad_h), (0, pad_w)), mode="constant")
    valid = np.zeros((size, size), dtype=np.uint8)
    valid[:new_h, :new_w] = 1
    return image_pad, mask_pad, valid


row = df[df["label"] == "forged"].iloc[0]
img = load_image_rgb(row["img_path"])
mask = load_mask_union(row["mask_path"], img.shape[:2])
img_p, mask_p, valid = resize_pad_to_square(img, mask, size=TARGET_SIZE)
print("Original:", img.shape, "Processed:", img_p.shape, "mask sum:", int(mask.sum()), "proc mask sum:", int(mask_p.sum()))


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    mask = mask.astype(bool)
    out = image.copy().astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return out.astype(np.uint8)


fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(img)
axes[0].set_title("original")
axes[0].axis("off")
axes[1].imshow(overlay_mask(img, mask))
axes[1].set_title("original + mask")
axes[1].axis("off")
axes[2].imshow(overlay_mask(img_p, mask_p))
axes[2].set_title("processed 512 + mask")
axes[2].axis("off")
plt.tight_layout()
plt.show()

# %%
# Celula 7 — Augmentations (imagem + máscara + valid_mask) + preview
train_aug = A.Compose(
    [
        # Random crop + resize (mantem saida 512x512)
        A.RandomResizedCrop(
            size=(TARGET_SIZE, TARGET_SIZE),
            scale=(0.6, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_AREA,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.5,
        ),
        # Geometria (flips + rotacao 0-360 + zoom in/out)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.75, 1.25),
            translate_percent=(-0.1, 0.1),
            rotate=(-180, 180),
            shear=(-8, 8),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.8,
        ),
        # Deformacoes locais leves
        A.ElasticTransform(
            alpha=1.0,
            sigma=30.0,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.15,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=(-0.03, 0.03),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.1,
        ),
        # Cor/contraste/ruido (robustez a intensidades e artefatos)
        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.05, 0.05),
                    p=1.0,
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ],
            p=0.25,
        ),
        A.GaussNoise(std_range=(0.01, 0.08), p=0.2),
        A.ImageCompression(quality_range=(60, 100), p=0.2),
        # Cutout: aplica na imagem, mas NAO altera mask/valid (fill_mask=None)
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.05, 0.25),
            hole_width_range=(0.05, 0.25),
            fill=0,
            fill_mask=None,
            p=0.3,
        ),
    ],
    additional_targets={"valid": "mask"},
)

aug = train_aug(image=img_p, mask=mask_p, valid=valid)
img_a = aug["image"]
mask_a = aug["mask"]
valid_a = aug["valid"]
print("Augmented shapes:", img_a.shape, mask_a.shape, valid_a.shape)
print("Aug config: aggressive geometric + color + local distort + cutout")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(overlay_mask(img_p, mask_p))
axes[0].set_title("processed + mask")
axes[0].axis("off")
axes[1].imshow(overlay_mask(img_a, mask_a))
axes[1].set_title("augmented + mask")
axes[1].axis("off")
axes[2].imshow(overlay_mask(img_a, (1 - valid_a).astype(np.uint8), color=(0, 255, 255)))
axes[2].set_title("augmented padding area")
axes[2].axis("off")
plt.tight_layout()
plt.show()

# %%
# Celula 8 — K-fold (5 folds) sem sklearn (estratificado por label + bins de area)
AREA_BINS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.2, 1.0]

df["area_bin"] = pd.cut(df["mask_area_frac"].fillna(0), bins=AREA_BINS, include_lowest=True, labels=False)
df["area_bin"] = df["area_bin"].fillna(0).astype(int)
df["stratify_key"] = df["label"].astype(str) + "_b" + df["area_bin"].astype(str)
df["fold"] = -1

rng = np.random.default_rng(RANDOM_SEED)
for key, idxs in df.groupby("stratify_key").indices.items():
    idxs = np.array(list(idxs), dtype=int)
    rng.shuffle(idxs)
    for i, idx in enumerate(idxs):
        df.loc[idx, "fold"] = int(i % 5)

assert int((df["fold"] < 0).sum()) == 0
fold_counts = pd.crosstab(df["fold"], df["label"])
display(fold_counts)
print("Fold sizes:", df["fold"].value_counts().sort_index().to_dict())

plt.figure(figsize=(6, 4))
df[df["label"] == "forged"].groupby("fold")["mask_area_frac"].mean().plot(kind="bar")
plt.title("Mean mask_area_frac (forged) por fold")
plt.ylabel("mean mask_area_frac")
plt.show()
