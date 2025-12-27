# %% [markdown]
# # Fase 3 — Treino do segmentador (máscara de duplicação)
#
# Objetivo:
# - Treinar um segmentador binário (1 canal) para localizar regiões duplicadas.
# - Salvar checkpoints em `/kaggle/working/outputs/models_seg/` (Kaggle) ou `outputs/models_seg/` (local).
#
# **Regras**
# - Notebook-only (sem importar código do projeto).
# - Internet pode estar OFF; use wheels offline se necessário.
#
# ---

# %%
# Célula 1 — Sanidade Kaggle (lembrete)
print("Kaggle submission constraints (lembrete):")
print("- Submissions via Notebook")
print("- Runtime <= 4h (CPU/GPU)")
print("- Internet: OFF no submit")

# %%
# Célula 2 — Imports + ambiente
import os
import random
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.simplefilter("default")


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SEED = 42
set_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

print("python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", DEVICE)

# %%
# Célula 2b — Instalação offline (opcional): wheels via Kaggle Dataset (sem internet)
import subprocess


def _find_offline_bundle() -> Path | None:
    if not is_kaggle():
        return None
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None

    candidates: list[Path] = []
    for ds in sorted(kaggle_input.glob("*")):
        for base in (ds, ds / "recodai_bundle"):
            if (base / "wheels").exists():
                candidates.append(base)

    if not candidates:
        return None
    if len(candidates) > 1:
        print("[OFFLINE INSTALL] múltiplos bundles com wheels encontrados; usando o primeiro:")
        for c in candidates:
            print(" -", c)
    return candidates[0]


OFFLINE_BUNDLE = _find_offline_bundle()
if OFFLINE_BUNDLE is None:
    print("[OFFLINE INSTALL] nenhum bundle com `wheels/` encontrado em `/kaggle/input`.")
else:
    wheel_dir = OFFLINE_BUNDLE / "wheels"
    whls = sorted(str(p) for p in wheel_dir.glob("*.whl"))
    print("[OFFLINE INSTALL] bundle:", OFFLINE_BUNDLE)
    print("[OFFLINE INSTALL] wheels:", len(whls))
    if whls:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheel_dir),
            *whls,
        ]
        print("[OFFLINE INSTALL] executando:", " ".join(cmd[:8]), "...", f"(+{len(whls)} wheels)")
        subprocess.check_call(cmd)
        print("[OFFLINE INSTALL] OK.")
    else:
        print("[OFFLINE INSTALL] aviso: `wheels/` existe, mas está vazio.")


def _is_competition_dataset_dir(path: Path) -> bool:
    return (path / "train_images").exists() or (path / "test_images").exists() or (path / "train_masks").exists()


def _candidate_python_roots(base: Path) -> list[Path]:
    roots = [
        base,
        base / "src",
        base / "vendor",
        base / "third_party",
        base / "recodai_bundle",
        base / "recodai_bundle" / "src",
        base / "recodai_bundle" / "vendor",
        base / "recodai_bundle" / "third_party",
    ]
    return [r for r in roots if r.exists()]


def add_local_package_to_syspath(package_dir_name: str) -> list[Path]:
    added: list[Path] = []
    if not is_kaggle():
        return added

    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return added

    for ds in sorted(kaggle_input.glob("*")):
        if _is_competition_dataset_dir(ds):
            continue
        for root in _candidate_python_roots(ds):
            pkg = root / package_dir_name
            if (pkg / "__init__.py").exists():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                    added.append(root)
                continue
            try:
                for child in sorted(p for p in root.glob("*") if p.is_dir()):
                    pkg2 = child / package_dir_name
                    if (pkg2 / "__init__.py").exists():
                        if str(child) not in sys.path:
                            sys.path.insert(0, str(child))
                            added.append(child)
            except Exception:
                continue

    if added:
        uniq = []
        for p in added:
            if p not in uniq:
                uniq.append(p)
        print(f"[LOCAL IMPORT] adicionado ao sys.path para '{package_dir_name}':")
        for p in uniq[:10]:
            print(" -", p)
        if len(uniq) > 10:
            print(" ...")
        return uniq

    print(f"[LOCAL IMPORT] não encontrei '{package_dir_name}/__init__.py' em `/kaggle/input/*` (fora do dataset da competição).")
    return added

# %%
# Célula 3 — Dataset paths (Kaggle/local)


def find_dataset_root() -> Path:
    if is_kaggle():
        base = Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection")
        if base.exists():
            return base
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for ds in sorted(kaggle_input.glob("*")):
                if (ds / "train_images").exists() and (ds / "test_images").exists():
                    return ds

    base = Path("data").resolve()
    if (base / "train_images").exists() and (base / "test_images").exists():
        return base

    raise FileNotFoundError("Dataset não encontrado. No Kaggle: anexe o dataset da competição.")


DATA_ROOT = find_dataset_root()
TRAIN_IMAGES = DATA_ROOT / "train_images"
TRAIN_MASKS = DATA_ROOT / "train_masks"
print("DATA_ROOT:", DATA_ROOT)

# %%
# Célula 4 — Index (train) para segmentação (máscara binária)


@dataclass(frozen=True)
class SegSample:
    case_id: str
    image_path: Path
    mask_path: Path | None
    label: int  # 0 authentic, 1 forged


def build_seg_index(train_images_dir: Path, train_masks_dir: Path) -> list[SegSample]:
    samples: list[SegSample] = []
    for label_name, y in [("authentic", 0), ("forged", 1)]:
        for img_path in sorted((train_images_dir / label_name).glob("*.png")):
            case_id = img_path.stem
            mask_path = train_masks_dir / f"{case_id}.npy" if label_name == "forged" else None
            samples.append(SegSample(case_id=case_id, image_path=img_path, mask_path=mask_path, label=int(y)))
    if not samples:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {train_images_dir}")
    return samples


train_samples = build_seg_index(TRAIN_IMAGES, TRAIN_MASKS)
y = np.array([s.label for s in train_samples], dtype=np.int64)
print("train samples:", len(train_samples))
print("authentic:", int((y == 0).sum()), "forged:", int((y == 1).sum()))

# %%
# Célula 5 — Split (5-fold estratificado)
N_FOLDS = 5
FOLD = 0

try:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(train_samples), dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(len(train_samples)), y)):
        folds[val_idx] = int(fold_id)
except Exception:
    print("[ERRO] Falha ao usar scikit-learn para split; usando split simples (não estratificado).")
    traceback.print_exc()
    folds = np.arange(len(train_samples), dtype=np.int64) % int(N_FOLDS)

train_idx = np.where(folds != int(FOLD))[0]
val_idx = np.where(folds == int(FOLD))[0]

print(f"fold={FOLD}: train={len(train_idx)} val={len(val_idx)}")

# %%
# Célula 6 — Dataset/DataLoader (patch-based, sem depender de libs externas)
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    print("[WARN] tqdm indisponível; usando loop simples.")

    def tqdm(x, **kwargs):  # type: ignore
        return x


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PATCH_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 2 if is_kaggle() else 4


def load_image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def load_mask_binary(path: Path | None, shape_hw: tuple[int, int]) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros(shape_hw, dtype=np.uint8)

    arr = np.load(path)
    arr = np.asarray(arr)

    if arr.ndim == 2:
        m = arr
    elif arr.ndim == 3:
        # comum: (N, H, W) instâncias ou (H, W, C)
        if arr.shape[0] < 32 and arr.shape[1:] == shape_hw:
            m = arr.max(axis=0)
        elif arr.shape[:2] == shape_hw:
            m = arr.max(axis=2)
        else:
            # fallback conservador
            m = arr.max(axis=0)
    else:
        raise ValueError(f"mask array com shape inesperado: {arr.shape}")

    return (np.asarray(m) > 0).astype(np.uint8)


def _pad_to_min(image: np.ndarray, mask: np.ndarray, min_h: int, min_w: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    pad_h = max(int(min_h) - int(h), 0)
    pad_w = max(int(min_w) - int(w), 0)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image_p = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    mask_p = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return image_p, mask_p


def _random_crop(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_min(image, mask, size, size)
    h, w = image.shape[:2]
    y0 = random.randint(0, h - size)
    x0 = random.randint(0, w - size)
    return image[y0 : y0 + size, x0 : x0 + size], mask[y0 : y0 + size, x0 : x0 + size]


def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])
    if random.random() < 0.20:
        k = random.randint(0, 3)
        if k:
            image = np.ascontiguousarray(np.rot90(image, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
    return image, mask


def normalize_image(image: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    x = image.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (x - mean) / std


class SegPatchDataset(Dataset):
    def __init__(self, samples: list[SegSample], train: bool):
        self.samples = samples
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[int(idx)]
        image = load_image_rgb(s.image_path)
        h, w = image.shape[:2]
        mask = load_mask_binary(s.mask_path, shape_hw=(h, w))

        image, mask = _random_crop(image, mask, int(PATCH_SIZE))
        if self.train:
            image, mask = _augment(image, mask)

        image = normalize_image(image)
        x = torch.from_numpy(image).permute(2, 0, 1).float()
        y = torch.from_numpy(mask[None, :, :]).float()
        return x, y


ds_train = SegPatchDataset([train_samples[i] for i in train_idx.tolist()], train=True)
ds_val = SegPatchDataset([train_samples[i] for i in val_idx.tolist()], train=False)

dl_train = DataLoader(
    ds_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    drop_last=True,
)
dl_val = DataLoader(
    ds_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    drop_last=False,
)

print("dl_train batches:", len(dl_train), "dl_val batches:", len(dl_val))

# %%
# Célula 7 — Modelo (SMP preferencial; fallback torchvision)
try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None
    print("[WARN] segmentation_models_pytorch indisponível.")
    traceback.print_exc()
    # tenta vendor via Kaggle Dataset GitHub
    add_local_package_to_syspath("segmentation_models_pytorch")
    add_local_package_to_syspath("segmentation_models_pytorch".replace("-", "_"))
    try:
        import segmentation_models_pytorch as smp  # type: ignore
    except Exception:
        smp = None


def _tv_deeplabv3_resnet50() -> nn.Module:
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50

        # compat com versões diferentes do torchvision
        try:
            m = deeplabv3_resnet50(weights=None, weights_backbone=None)
        except TypeError:
            m = deeplabv3_resnet50(pretrained=False)

        # troca head para 1 canal
        head = m.classifier[-1]
        m.classifier[-1] = nn.Conv2d(head.in_channels, 1, kernel_size=1)

        class _Wrap(nn.Module):
            def __init__(self, base: nn.Module):
                super().__init__()
                self.base = base

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.base(x)
                if isinstance(out, dict):
                    out = out["out"]
                return out

        return _Wrap(m)
    except Exception:
        print("[ERRO] falha ao criar torchvision deeplabv3_resnet50.")
        traceback.print_exc()
        raise


def build_seg_model() -> tuple[nn.Module, dict]:
    if smp is not None:
        try:
            m = smp.UnetPlusPlus(
                encoder_name="efficientnet-b4",
                encoder_weights=None,  # evita download
                classes=1,
                activation=None,
            )
            cfg = {
                "backend": "smp",
                "arch": "UnetPlusPlus",
                "encoder_name": "efficientnet-b4",
                "classes": 1,
            }
            return m, cfg
        except Exception:
            print("[ERRO] falha ao criar SMP UnetPlusPlus.")
            traceback.print_exc()
            raise

    m = _tv_deeplabv3_resnet50()
    cfg = {"backend": "torchvision", "arch": "deeplabv3_resnet50", "classes": 1}
    return m, cfg


model, model_cfg = build_seg_model()
model = model.to(DEVICE)
print("model cfg:", model_cfg)

# %%
# Célula 8 — Loss (BCE + Dice)


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    den = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    d = dice_loss_with_logits(logits, targets)
    return bce + d


@torch.no_grad()
def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs >= float(threshold)).float()
    inter = (pred * targets).sum().item()
    den = pred.sum().item() + targets.sum().item()
    return float((2.0 * inter + eps) / (den + eps))

# %%
# Célula 9 — Treino (com progresso) + checkpoint
from time import time

LR = 1e-3
EPOCHS = 15
WEIGHT_DECAY = 1e-2

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


def output_root() -> Path:
    if is_kaggle():
        return Path("/kaggle/working")
    return Path(".").resolve()


MODEL_ID = "unetpp_effb4" if model_cfg.get("backend") == "smp" else "deeplabv3_r50"
SAVE_DIR = output_root() / "outputs" / "models_seg" / MODEL_ID / f"fold_{int(FOLD)}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = SAVE_DIR / "best.pt"

best_dice = -1.0


def train_one_epoch(model: nn.Module, loader: DataLoader) -> float:
    model.train()
    losses: list[float] = []
    for x, yb in tqdm(loader, desc="train", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss = bce_dice_loss(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    losses: list[float] = []
    dices: list[float] = []
    for x, yb in tqdm(loader, desc="val", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(x)
        losses.append(float(bce_dice_loss(logits, yb).item()))
        dices.append(dice_score_from_logits(logits, yb, threshold=0.5))
    return {"loss": float(np.mean(losses)) if losses else float("nan"), "dice@0.5": float(np.mean(dices)) if dices else 0.0}


for epoch in range(1, int(EPOCHS) + 1):
    t0 = time()
    tr_loss = train_one_epoch(model, dl_train)
    val = evaluate(model, dl_val)
    elapsed = time() - t0
    print(
        f"epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_loss={val['loss']:.4f} | "
        f"dice@0.5={val['dice@0.5']:.4f} | time={elapsed:.1f}s"
    )

    if float(val["dice@0.5"]) > best_dice:
        best_dice = float(val["dice@0.5"])
        ckpt = {
            "model_state": model.state_dict(),
            "config": {
                **model_cfg,
                "model_id": MODEL_ID,
                "patch_size": int(PATCH_SIZE),
                "fold": int(FOLD),
                "seed": int(SEED),
            },
            "score": float(best_dice),
        }
        torch.save(ckpt, BEST_PATH)
        print("  saved best ->", BEST_PATH)

print("best dice:", best_dice)
print("saved dir:", SAVE_DIR)

