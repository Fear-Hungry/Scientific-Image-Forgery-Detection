# %% [markdown]
# # Pipeline único — Fases 1→4 (Kaggle / Offline)
#
# Este notebook junta tudo em um só lugar:
#
# 1) **Setup offline + checagens**
# 2) **Treino do classificador** (authentic vs forged) *(opcional)*
# 3) **Treino do segmentador** (máscara de duplicação) *(opcional)*
# 4) **Inferência + submissão** (`submission.csv`)
#
# ## Regras / Decisões
# - Notebook-only (não importa `src/forgeryseg/`).
# - Compatível com Kaggle **internet OFF** (instala wheels locais se existirem).
# - Não esconde erros: exceções e tracebacks aparecem explicitamente.
#
# ---

# %%
# Fase 1 — Célula 1: Sanidade Kaggle (lembrete)
print("Kaggle submission constraints (lembrete):")
print("- Submissions via Notebook")
print("- Runtime <= 4h (CPU/GPU)")
print("- Internet: OFF no submit")
print("- Output: /kaggle/working/submission.csv")

# %%
# Fase 1 — Célula 2: Imports + ambiente
import csv
import json
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
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


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
# Fase 1 — Célula 2b: Instalação offline (wheels) — NÃO resolve deps (evita puxar nvidia-cuda-*)
#
# Estruturas suportadas:
# - `/kaggle/input/<dataset>/wheels/*.whl`
# - `/kaggle/input/<dataset>/recodai_bundle/wheels/*.whl`
#
# Observação: instalamos com `--no-deps` para não tentar instalar dependências do torch offline.
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
    if not whls:
        print("[OFFLINE INSTALL] aviso: diretório `wheels/` existe mas não há `.whl`.")
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheel_dir),
            "--no-deps",
            *whls,
        ]
        print("[OFFLINE INSTALL] executando:", " ".join(cmd[:9]), "...", f"(+{len(whls)} wheels)")
        subprocess.check_call(cmd)
        print("[OFFLINE INSTALL] OK.")


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
    """
    Procura por `package_dir_name/__init__.py` em `/kaggle/input/*` (exceto o dataset da competição)
    e adiciona o root correspondente ao `sys.path`.
    """
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
# Fase 1 — Célula 3: Checagem de dependências (não esconde erro)


def _try_import(module_name: str) -> None:
    try:
        mod = __import__(module_name)
        ver = getattr(mod, "__version__", None)
        print(f"[OK] import {module_name}" + (f" ({ver})" if ver else ""))
    except Exception:
        print(f"[ERRO] falha ao importar: {module_name}")
        traceback.print_exc()


for pkg in [
    "numpy",
    "torch",
    "torchvision",
    "albumentations",
    "cv2",
    "timm",
    "segmentation_models_pytorch",
    "scipy",
    "sklearn",
]:
    _try_import(pkg)

# %%
# Fase 1 — Célula 4: Dataset root (Kaggle/local)


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

    raise FileNotFoundError(
        "Dataset não encontrado.\n"
        "- No Kaggle: anexe o dataset da competição (Add data).\n"
        "- Local: espere `data/train_images` e `data/train_masks`."
    )


DATA_ROOT = find_dataset_root()
TRAIN_IMAGES = DATA_ROOT / "train_images"
TRAIN_MASKS = DATA_ROOT / "train_masks"
TEST_IMAGES = DATA_ROOT / "test_images"

num_auth = len(list((TRAIN_IMAGES / "authentic").glob("*.png")))
num_forged = len(list((TRAIN_IMAGES / "forged").glob("*.png")))
num_masks = len(list(TRAIN_MASKS.glob("*.npy")))
num_test = len(list(TEST_IMAGES.glob("*.png")))

print("DATA_ROOT:", DATA_ROOT)
print("train/authentic:", num_auth)
print("train/forged:", num_forged)
print("train_masks:", num_masks)
print("test_images:", num_test)
if num_auth == 0 and num_forged == 0:
    raise FileNotFoundError(
        "Nenhuma imagem encontrada em `train_images/authentic` e `train_images/forged`.\n"
        f"DATA_ROOT={DATA_ROOT}\n"
        "No Kaggle, isso normalmente significa que o dataset da competição não foi anexado."
    )

# %%
# Fase 1 — Célula 5: Config global (liga/desliga)
RUN_TRAIN_CLS = False  # mude para True se quiser treinar classificador aqui
RUN_TRAIN_SEG = False  # mude para True se quiser treinar segmentação aqui
RUN_SUBMISSION = True  # deixe True no submit

N_FOLDS = 5
FOLD = 0  # qual fold usar para validar/treinar (quando não for ensemble)

print("RUN_TRAIN_CLS:", RUN_TRAIN_CLS)
print("RUN_TRAIN_SEG:", RUN_TRAIN_SEG)
print("RUN_SUBMISSION:", RUN_SUBMISSION)

# %%
# Fase 2 — Célula 6: Classificador (index + split)


@dataclass(frozen=True)
class ClsSample:
    case_id: str
    image_path: Path
    label: int  # 0 authentic, 1 forged


def build_cls_index(train_images_dir: Path) -> list[ClsSample]:
    samples: list[ClsSample] = []
    for label_name, y in [("authentic", 0), ("forged", 1)]:
        for img_path in sorted((train_images_dir / label_name).glob("*.png")):
            samples.append(ClsSample(case_id=img_path.stem, image_path=img_path, label=int(y)))
    if not samples:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {train_images_dir}")
    return samples


cls_samples = build_cls_index(TRAIN_IMAGES)
cls_y = np.array([s.label for s in cls_samples], dtype=np.int64)
print("cls samples:", len(cls_samples), "auth:", int((cls_y == 0).sum()), "forged:", int((cls_y == 1).sum()))

try:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cls_folds = np.zeros(len(cls_samples), dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(len(cls_samples)), cls_y)):
        cls_folds[val_idx] = int(fold_id)
except Exception:
    print("[ERRO] scikit-learn falhou (StratifiedKFold). Usando split simples.")
    traceback.print_exc()
    cls_folds = np.arange(len(cls_samples), dtype=np.int64) % int(N_FOLDS)

cls_train_idx = np.where(cls_folds != int(FOLD))[0]
cls_val_idx = np.where(cls_folds == int(FOLD))[0]
print(f"cls fold={FOLD}: train={len(cls_train_idx)} val={len(cls_val_idx)}")

# %%
# Fase 2 — Célula 7: Classificador (dataset/dataloader + modelo + treino)
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import torchvision.transforms as T
except Exception:
    print("[ERRO] torchvision falhou no import.")
    traceback.print_exc()
    raise

try:
    from tqdm.auto import tqdm
except Exception:
    print("[WARN] tqdm indisponível; usando loop simples.")

    def tqdm(x, **kwargs):  # type: ignore
        return x


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLS_IMAGE_SIZE = 384
CLS_BATCH_SIZE = 32
CLS_EPOCHS = 10
CLS_LR = 3e-4
CLS_WEIGHT_DECAY = 1e-2
CLS_MODEL_NAME = "tf_efficientnet_b4_ns"


def build_transform(train: bool) -> T.Compose:
    aug = []
    if train:
        aug += [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]
    aug += [
        T.Resize((CLS_IMAGE_SIZE, CLS_IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(aug)


class ClsDataset(Dataset):
    def __init__(self, samples: list[ClsSample], transform: T.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[int(idx)]
        img = Image.open(s.image_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([float(s.label)], dtype=torch.float32)
        return x, y


cls_ds_train = ClsDataset([cls_samples[i] for i in cls_train_idx.tolist()], build_transform(train=True))
cls_ds_val = ClsDataset([cls_samples[i] for i in cls_val_idx.tolist()], build_transform(train=False))

# Nota: fora de notebook (ex.: rodando .py), multiprocess pode quebrar; aqui preferimos 0 fora do Kaggle.
CLS_NUM_WORKERS = 2 if is_kaggle() else 0

cls_dl_train = DataLoader(
    cls_ds_train,
    batch_size=CLS_BATCH_SIZE,
    shuffle=True,
    num_workers=CLS_NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    drop_last=True,
)
cls_dl_val = DataLoader(
    cls_ds_val,
    batch_size=CLS_BATCH_SIZE,
    shuffle=False,
    num_workers=CLS_NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    drop_last=False,
)

try:
    import timm
except Exception:
    timm = None
    print("[WARN] timm indisponível; usando fallback torchvision.")
    traceback.print_exc()


def build_cls_model(model_name: str) -> nn.Module:
    if timm is not None:
        return timm.create_model(model_name, pretrained=False, num_classes=1)
    from torchvision.models import resnet50

    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def _compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)


cls_model = build_cls_model(CLS_MODEL_NAME).to(DEVICE)
cls_pos_weight = _compute_pos_weight(cls_y[cls_train_idx]).to(DEVICE)
cls_criterion = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)
cls_optimizer = torch.optim.AdamW(cls_model.parameters(), lr=CLS_LR, weight_decay=CLS_WEIGHT_DECAY)
cls_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


def output_root() -> Path:
    if is_kaggle():
        return Path("/kaggle/working")
    return Path(".").resolve()


CLS_SAVE_DIR = output_root() / "outputs" / "models_cls" / f"fold_{int(FOLD)}"
CLS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CLS_BEST_PATH = CLS_SAVE_DIR / "best.pt"


@torch.no_grad()
def cls_evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    losses: list[float] = []
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for x, yb in tqdm(loader, desc="cls val", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(x).view(-1, 1)
        loss = cls_criterion(logits, yb)
        losses.append(float(loss.item()))
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0).reshape(-1)
    targets_np = np.concatenate(all_targets, axis=0).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    pred = (probs >= 0.5).astype(np.int64)
    acc = float((pred == targets_np.astype(np.int64)).mean())

    out = {"loss": float(np.mean(losses)) if losses else float("nan"), "acc@0.5": acc}
    try:
        from sklearn.metrics import roc_auc_score

        out["auc"] = float(roc_auc_score(targets_np, probs))
    except Exception:
        print("[WARN] falha ao calcular AUC (roc_auc_score).")
        traceback.print_exc()
    return out


def cls_train_one_epoch(model: nn.Module, loader: DataLoader) -> float:
    model.train()
    losses: list[float] = []
    for x, yb in tqdm(loader, desc="cls train", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        cls_optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(x).view(-1, 1)
            loss = cls_criterion(logits, yb)
        cls_scaler.scale(loss).backward()
        cls_scaler.step(cls_optimizer)
        cls_scaler.update()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


if RUN_TRAIN_CLS:
    best_score = -1.0
    for epoch in range(1, int(CLS_EPOCHS) + 1):
        train_loss = cls_train_one_epoch(cls_model, cls_dl_train)
        val = cls_evaluate(cls_model, cls_dl_val)
        score = float(val.get("auc", -val["loss"]))
        print(
            f"[CLS] epoch {epoch:02d}/{CLS_EPOCHS} | train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | acc@0.5={val['acc@0.5']:.4f} | "
            + (f"auc={val.get('auc', float('nan')):.4f}" if "auc" in val else "")
        )
        if score > best_score:
            best_score = score
            ckpt = {
                "model_state": cls_model.state_dict(),
                "config": {
                    "backend": "timm" if timm is not None else "torchvision",
                    "model_name": CLS_MODEL_NAME,
                    "image_size": int(CLS_IMAGE_SIZE),
                    "fold": int(FOLD),
                    "seed": int(SEED),
                },
                "score": float(best_score),
            }
            torch.save(ckpt, CLS_BEST_PATH)
            print("[CLS] saved best ->", CLS_BEST_PATH)
    print("[CLS] done. best score:", best_score)
else:
    print("[CLS] treino desativado (RUN_TRAIN_CLS=False).")

# %%
# Fase 3 — Célula 8: Segmentação (index + split)


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


seg_samples = build_seg_index(TRAIN_IMAGES, TRAIN_MASKS)
seg_y = np.array([s.label for s in seg_samples], dtype=np.int64)

try:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    seg_folds = np.zeros(len(seg_samples), dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(len(seg_samples)), seg_y)):
        seg_folds[val_idx] = int(fold_id)
except Exception:
    print("[ERRO] scikit-learn falhou (StratifiedKFold). Usando split simples.")
    traceback.print_exc()
    seg_folds = np.arange(len(seg_samples), dtype=np.int64) % int(N_FOLDS)

seg_train_idx = np.where(seg_folds != int(FOLD))[0]
seg_val_idx = np.where(seg_folds == int(FOLD))[0]
print("seg samples:", len(seg_samples), "fold:", FOLD, "train:", len(seg_train_idx), "val:", len(seg_val_idx))

# %%
# Fase 3 — Célula 9: Segmentação (dataset patch + modelo + treino)
SEG_PATCH_SIZE = 512
SEG_BATCH_SIZE = 8
SEG_EPOCHS = 15
SEG_LR = 1e-3
SEG_WEIGHT_DECAY = 1e-2


def load_image_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask_binary(path: Path | None, shape_hw: tuple[int, int]) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros(shape_hw, dtype=np.uint8)

    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        m = arr
    elif arr.ndim == 3:
        if arr.shape[0] < 64 and arr.shape[1:] == shape_hw:
            m = arr.max(axis=0)
        elif arr.shape[:2] == shape_hw:
            m = arr.max(axis=2)
        else:
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


def _augment_seg(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        image, mask = _random_crop(image, mask, int(SEG_PATCH_SIZE))
        if self.train:
            image, mask = _augment_seg(image, mask)
        image = normalize_image(image)
        x = torch.from_numpy(image).permute(2, 0, 1).float()
        y = torch.from_numpy(mask[None, :, :]).float()
        return x, y


seg_ds_train = SegPatchDataset([seg_samples[i] for i in seg_train_idx.tolist()], train=True)
seg_ds_val = SegPatchDataset([seg_samples[i] for i in seg_val_idx.tolist()], train=False)

SEG_NUM_WORKERS = 2 if is_kaggle() else 0
seg_dl_train = DataLoader(
    seg_ds_train,
    batch_size=SEG_BATCH_SIZE,
    shuffle=True,
    num_workers=SEG_NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    drop_last=True,
)
seg_dl_val = DataLoader(
    seg_ds_val,
    batch_size=SEG_BATCH_SIZE,
    shuffle=False,
    num_workers=SEG_NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    drop_last=False,
)

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None
    print("[WARN] segmentation_models_pytorch indisponível.")
    traceback.print_exc()


def _tv_deeplabv3_resnet50() -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    try:
        base = deeplabv3_resnet50(weights=None, weights_backbone=None)
    except TypeError:
        base = deeplabv3_resnet50(pretrained=False)

    head = base.classifier[-1]
    base.classifier[-1] = nn.Conv2d(head.in_channels, 1, kernel_size=1)

    class _Wrap(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.m(x)
            if isinstance(out, dict):
                out = out["out"]
            return out

    return _Wrap(base)


def build_seg_model() -> tuple[nn.Module, dict]:
    if smp is not None:
        m = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None, classes=1, activation=None)
        cfg = {"backend": "smp", "arch": "UnetPlusPlus", "encoder_name": "efficientnet-b4", "classes": 1}
        return m, cfg
    m = _tv_deeplabv3_resnet50()
    cfg = {"backend": "torchvision", "arch": "deeplabv3_resnet50", "classes": 1}
    return m, cfg


seg_model, seg_model_cfg = build_seg_model()
seg_model = seg_model.to(DEVICE)
print("seg model cfg:", seg_model_cfg)


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


seg_optimizer = torch.optim.AdamW(seg_model.parameters(), lr=SEG_LR, weight_decay=SEG_WEIGHT_DECAY)
seg_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

SEG_MODEL_ID = "unetpp_effb4" if seg_model_cfg.get("backend") == "smp" else "deeplabv3_r50"
SEG_SAVE_DIR = output_root() / "outputs" / "models_seg" / SEG_MODEL_ID / f"fold_{int(FOLD)}"
SEG_SAVE_DIR.mkdir(parents=True, exist_ok=True)
SEG_BEST_PATH = SEG_SAVE_DIR / "best.pt"


def seg_train_one_epoch(model: nn.Module, loader: DataLoader) -> float:
    model.train()
    losses: list[float] = []
    for x, yb in tqdm(loader, desc="seg train", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        seg_optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss = bce_dice_loss(logits, yb)
        seg_scaler.scale(loss).backward()
        seg_scaler.step(seg_optimizer)
        seg_scaler.update()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def seg_evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    losses: list[float] = []
    dices: list[float] = []
    for x, yb in tqdm(loader, desc="seg val", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(x)
        losses.append(float(bce_dice_loss(logits, yb).item()))
        dices.append(dice_score_from_logits(logits, yb, threshold=0.5))
    return {"loss": float(np.mean(losses)) if losses else float("nan"), "dice@0.5": float(np.mean(dices)) if dices else 0.0}


if RUN_TRAIN_SEG:
    best_dice = -1.0
    for epoch in range(1, int(SEG_EPOCHS) + 1):
        tr = seg_train_one_epoch(seg_model, seg_dl_train)
        val = seg_evaluate(seg_model, seg_dl_val)
        print(f"[SEG] epoch {epoch:02d}/{SEG_EPOCHS} | train_loss={tr:.4f} | val_loss={val['loss']:.4f} | dice@0.5={val['dice@0.5']:.4f}")
        if float(val["dice@0.5"]) > best_dice:
            best_dice = float(val["dice@0.5"])
            ckpt = {
                "model_state": seg_model.state_dict(),
                "config": {**seg_model_cfg, "model_id": SEG_MODEL_ID, "patch_size": int(SEG_PATCH_SIZE), "fold": int(FOLD), "seed": int(SEED)},
                "score": float(best_dice),
            }
            torch.save(ckpt, SEG_BEST_PATH)
            print("[SEG] saved best ->", SEG_BEST_PATH)
    print("[SEG] done. best dice:", best_dice)
else:
    print("[SEG] treino desativado (RUN_TRAIN_SEG=False).")

# %%
# Fase 4 — Célula 10: Inferência (load checkpoints + submission)


@dataclass(frozen=True)
class TestSample:
    case_id: str
    image_path: Path


def build_test_index(test_images_dir: Path) -> list[TestSample]:
    paths = sorted(test_images_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {test_images_dir}")
    return [TestSample(case_id=p.stem, image_path=p) for p in paths]


test_samples = build_test_index(TEST_IMAGES)
print("test samples:", len(test_samples))


def _find_models_dir(dir_name: str) -> Path | None:
    # Preferência: Kaggle Dataset anexado contendo outputs/
    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            candidates = []
            for ds in sorted(ki.glob("*")):
                for base in (ds, ds / "recodai_bundle"):
                    cand = base / "outputs" / dir_name
                    if cand.exists():
                        candidates.append(cand)
            if candidates:
                if len(candidates) > 1:
                    print(f"[CKPT] múltiplos candidatos para outputs/{dir_name}; usando o primeiro:")
                    for c in candidates:
                        print(" -", c)
                return candidates[0]

    local = output_root() / "outputs" / dir_name
    if local.exists():
        return local
    return None


MODELS_SEG_DIR = _find_models_dir("models_seg")
MODELS_CLS_DIR = _find_models_dir("models_cls")
print("MODELS_SEG_DIR:", MODELS_SEG_DIR)
print("MODELS_CLS_DIR:", MODELS_CLS_DIR)


def _load_checkpoint(path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("config", {})
    return ckpt, {}


def build_seg_from_config(cfg: dict) -> nn.Module:
    backend = str(cfg.get("backend", "torchvision"))
    arch = str(cfg.get("arch", cfg.get("model_id", "deeplabv3_resnet50")))

    if backend == "smp":
        try:
            import segmentation_models_pytorch as smp  # type: ignore
        except Exception:
            print("[ERRO] checkpoint pede SMP, mas segmentation_models_pytorch não está disponível.")
            traceback.print_exc()
            raise

        encoder_name = str(cfg.get("encoder_name", "efficientnet-b4"))
        classes = int(cfg.get("classes", 1))

        if arch.lower() in {"unetplusplus", "unetpp"}:
            return smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, classes=classes, activation=None)
        if arch.lower() == "unet":
            return smp.Unet(encoder_name=encoder_name, encoder_weights=None, classes=classes, activation=None)
        if arch.lower() in {"deeplabv3plus", "deeplabv3+"}:
            return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None, classes=classes, activation=None)
        raise ValueError(f"Arquitetura SMP desconhecida no cfg: {arch!r}")

    return _tv_deeplabv3_resnet50()


def build_cls_from_config(cfg: dict) -> tuple[nn.Module, int]:
    backend = str(cfg.get("backend", "torchvision"))
    model_name = str(cfg.get("model_name", "resnet50"))
    image_size = int(cfg.get("image_size", CLS_IMAGE_SIZE))

    if backend == "timm":
        if timm is None:
            raise RuntimeError("checkpoint pede timm, mas timm é None")
        return timm.create_model(model_name, pretrained=False, num_classes=1), image_size

    from torchvision.models import resnet50

    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m, image_size


SEG_MODELS: list[nn.Module] = []
if MODELS_SEG_DIR is None:
    print("[ERRO] MODELS_SEG_DIR não encontrado. Treine e salve em outputs/models_seg/... ou anexe dataset com isso.")
else:
    seg_ckpts = sorted(MODELS_SEG_DIR.glob("*/*/best.pt"))
    print("seg checkpoints encontrados:", len(seg_ckpts))
    for p in seg_ckpts:
        try:
            state, cfg = _load_checkpoint(p)
            m = build_seg_from_config(cfg)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            SEG_MODELS.append(m)
        except Exception:
            print("[ERRO] falha ao carregar seg checkpoint:", p)
            traceback.print_exc()

if not SEG_MODELS:
    raise RuntimeError("Nenhum modelo de segmentação foi carregado. Veja os erros acima.")


CLS_MODELS: list[nn.Module] = []
CLS_INFER_IMAGE_SIZE = CLS_IMAGE_SIZE
CLS_SKIP_THRESHOLD = 0.30

if MODELS_CLS_DIR is not None:
    cls_ckpts = sorted(MODELS_CLS_DIR.glob("fold_*/best.pt"))
    print("cls checkpoints encontrados:", len(cls_ckpts))
    for p in cls_ckpts:
        try:
            state, cfg = _load_checkpoint(p)
            m, image_size = build_cls_from_config(cfg)
            CLS_INFER_IMAGE_SIZE = int(image_size)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            CLS_MODELS.append(m)
        except Exception:
            print("[ERRO] falha ao carregar cls checkpoint:", p)
            traceback.print_exc()

print("loaded cls models:", len(CLS_MODELS), "CLS_INFER_IMAGE_SIZE:", CLS_INFER_IMAGE_SIZE)


@torch.no_grad()
def predict_prob_forged(image: np.ndarray) -> float:
    if not CLS_MODELS:
        raise RuntimeError("CLS_MODELS vazio")
    import torch.nn.functional as F

    img = normalize_image(image)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    if CLS_INFER_IMAGE_SIZE and x.shape[-2:] != (CLS_INFER_IMAGE_SIZE, CLS_INFER_IMAGE_SIZE):
        x = F.interpolate(x, size=(CLS_INFER_IMAGE_SIZE, CLS_INFER_IMAGE_SIZE), mode="bilinear", align_corners=False)

    probs: list[float] = []
    for m in CLS_MODELS:
        logits = m(x).view(-1)
        probs.append(float(torch.sigmoid(logits)[0].item()))
    return float(np.mean(probs))


try:
    from scipy.ndimage import label as _cc_label
except Exception:
    _cc_label = None

try:
    import cv2
except Exception:
    cv2 = None


def extract_components(mask: np.ndarray, min_area: int = 0) -> list[np.ndarray]:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if mask.max() == 0:
        return []

    instances: list[np.ndarray] = []
    if _cc_label is not None:
        labeled, num = _cc_label(mask > 0)
        for idx in range(1, int(num) + 1):
            comp = (labeled == idx)
            if min_area and int(comp.sum()) < int(min_area):
                continue
            instances.append(comp.astype(np.uint8))
        return instances

    if cv2 is None:
        raise ImportError("connected components requires scipy or opencv-python")

    num_labels, labels = cv2.connectedComponents(mask, connectivity=4)
    for idx in range(1, int(num_labels)):
        comp = (labels == idx)
        if min_area and int(comp.sum()) < int(min_area):
            continue
        instances.append(comp.astype(np.uint8))
    return instances


AUTHENTIC_LABEL = "authentic"


def rle_encode(mask: np.ndarray) -> list[int]:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.max() == 0:
        return []
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return changes.tolist()


def encode_instances(instances: list[np.ndarray]) -> str:
    if not instances:
        return AUTHENTIC_LABEL
    parts = [json.dumps(rle_encode(m)) for m in instances]
    return ";".join(parts)


def _tile_coords(length: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    stride = int(tile_size) - int(overlap)
    if stride <= 0:
        raise ValueError("tile_size must be larger than overlap")
    if length <= tile_size:
        return [(0, tile_size)]
    coords = list(range(0, length - tile_size + 1, stride))
    if coords[-1] != length - tile_size:
        coords.append(length - tile_size)
    return [(int(start), int(start + tile_size)) for start in coords]


def _pad_image(image: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    return padded, (pad_h, pad_w)


@torch.no_grad()
def _predict_tensor(model: nn.Module, tensor: torch.Tensor, device: str) -> torch.Tensor:
    tensor = tensor.to(device)
    logits = model(tensor)
    return torch.sigmoid(logits)


def predict_image(
    model: nn.Module,
    image: np.ndarray,
    device: str,
    tile_size: int = 0,
    overlap: int = 0,
    max_size: int = 0,
) -> np.ndarray:
    import torch.nn.functional as F

    orig_h, orig_w = image.shape[:2]

    if tile_size and int(tile_size) > 0:
        padded, _ = _pad_image(image, int(tile_size), int(tile_size))
        pad_h, pad_w = padded.shape[0], padded.shape[1]
        pred_sum = np.zeros((pad_h, pad_w), dtype=np.float32)
        pred_count = np.zeros((pad_h, pad_w), dtype=np.float32)

        ys = _tile_coords(padded.shape[0], int(tile_size), int(overlap))
        xs = _tile_coords(padded.shape[1], int(tile_size), int(overlap))
        for y0, y1 in ys:
            for x0, x1 in xs:
                tile = padded[y0:y1, x0:x1]
                tile_norm = normalize_image(tile)
                tile_tensor = torch.from_numpy(tile_norm).permute(2, 0, 1).unsqueeze(0)
                probs = _predict_tensor(model, tile_tensor, device)
                prob_tile = probs.squeeze(0).squeeze(0).cpu().numpy()
                pred_sum[y0:y1, x0:x1] += prob_tile
                pred_count[y0:y1, x0:x1] += 1.0

        pred = pred_sum / np.maximum(pred_count, 1.0)
        return pred[:orig_h, :orig_w]

    image_norm = normalize_image(image)
    tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
    if max_size and max(orig_h, orig_w) > int(max_size):
        scale = int(max_size) / float(max(orig_h, orig_w))
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    probs = _predict_tensor(model, tensor, device)
    if probs.shape[-2:] != (orig_h, orig_w):
        probs = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return probs.squeeze(0).squeeze(0).cpu().numpy()


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(mask) >= float(threshold)).astype(np.uint8)


def postprocess_binary_mask(mask: np.ndarray, max_hole_area: int = 64, morph_kernel: int = 0) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.max() == 0 or cv2 is None:
        return m
    try:
        if max_hole_area > 0:
            inv = (1 - m).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
            for i in range(1, int(n)):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area <= int(max_hole_area):
                    m[labels == i] = 1
        if morph_kernel and int(morph_kernel) > 1:
            k = int(morph_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    except Exception:
        print("[PP] erro no pós-processamento; seguindo sem. Erro abaixo:")
        traceback.print_exc()
    return (m > 0).astype(np.uint8)


TILE_SIZE = 1024
OVERLAP = 128
MAX_SIZE = 0
THRESHOLD = 0.50
MIN_AREA = 32
USE_TTA = True
TTA_MODES = ("none", "hflip", "vflip")
PP_MAX_HOLE_AREA = 64
PP_MORPH_KERNEL = 0


def _apply_tta(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image
    if mode == "hflip":
        return np.ascontiguousarray(image[:, ::-1])
    if mode == "vflip":
        return np.ascontiguousarray(image[::-1, :])
    raise ValueError(f"tta mode inválido: {mode}")


def _undo_tta(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return mask
    if mode == "hflip":
        return np.ascontiguousarray(mask[:, ::-1])
    if mode == "vflip":
        return np.ascontiguousarray(mask[::-1, :])
    raise ValueError(f"tta mode inválido: {mode}")


def predict_seg_ensemble_prob(image: np.ndarray) -> np.ndarray:
    probs_sum = None
    count = 0
    modes = TTA_MODES if USE_TTA else ("none",)
    for mode in modes:
        img_t = _apply_tta(image, mode)
        ens = None
        for m in SEG_MODELS:
            p = predict_image(m, img_t, DEVICE, tile_size=TILE_SIZE, overlap=OVERLAP, max_size=MAX_SIZE)
            ens = p if ens is None else (ens + p)
        ens = ens / float(len(SEG_MODELS))
        ens = _undo_tta(ens, mode)
        probs_sum = ens if probs_sum is None else (probs_sum + ens)
        count += 1
    return probs_sum / float(max(count, 1))


def predict_instances(image: np.ndarray) -> list[np.ndarray]:
    prob = predict_seg_ensemble_prob(image)
    bin_mask = binarize(prob, threshold=THRESHOLD)
    bin_mask = postprocess_binary_mask(bin_mask, max_hole_area=int(PP_MAX_HOLE_AREA), morph_kernel=int(PP_MORPH_KERNEL))
    return extract_components(bin_mask, min_area=int(MIN_AREA))


SUBMISSION_PATH = Path("/kaggle/working/submission.csv") if is_kaggle() else (output_root() / "submission.csv")
if RUN_SUBMISSION:
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSION_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()
        for s in tqdm(test_samples, desc="infer"):
            img = load_image_rgb(s.image_path)
            if CLS_MODELS:
                p_forged = predict_prob_forged(img)
                if float(p_forged) < float(CLS_SKIP_THRESHOLD):
                    writer.writerow({"case_id": s.case_id, "annotation": AUTHENTIC_LABEL})
                    continue
            inst = predict_instances(img)
            writer.writerow({"case_id": s.case_id, "annotation": encode_instances(inst)})

    print("wrote:", SUBMISSION_PATH)
else:
    print("[SUBMISSION] RUN_SUBMISSION=False; não gerou CSV.")

