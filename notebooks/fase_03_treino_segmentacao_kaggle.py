# %% [markdown]
# # Fase 3 — Treino de segmentação (Kaggle)
#
# Objetivo:
# - Treinar um modelo de segmentação para localizar regiões duplicadas.
# - Salvar checkpoints em `outputs/models_seg/<model_id>/fold_<k>/best.pt`.
#
# **Importante**
# - Este notebook **importa o código do projeto** em `src/forgeryseg/` (modularizado).
# - Internet pode estar OFF: use wheels offline se necessário.
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
from pathlib import Path

import numpy as np
import torch

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
# Célula 2b — Instalação offline (opcional): wheels via Kaggle Dataset (sem internet)
#
# Instala TODAS as wheels encontradas com `--no-deps` para evitar o pip tentar puxar
# dependências do torch (ex.: nvidia-cuda-*) quando o Kaggle está offline.
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


OFFLINE_BUNDLE = _find_offline_bundle()
if OFFLINE_BUNDLE is None:
    print("[OFFLINE INSTALL] nenhum bundle com `wheels/` encontrado em `/kaggle/input`.")
else:
    wheel_dir = OFFLINE_BUNDLE / "wheels"
    whls = sorted(str(p) for p in wheel_dir.glob("*.whl"))
    print("[OFFLINE INSTALL] bundle:", OFFLINE_BUNDLE)
    print("[OFFLINE INSTALL] wheels:", len(whls))
    if not whls:
        print("[OFFLINE INSTALL] aviso: `wheels/` existe, mas está vazio.")
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

# %%
# Célula 2c — Import do projeto (src/forgeryseg)
try:
    import forgeryseg  # type: ignore
except Exception:
    local_src = Path("src").resolve()
    if (local_src / "forgeryseg" / "__init__.py").exists() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    if is_kaggle():
        add_local_package_to_syspath("forgeryseg")
    import forgeryseg  # type: ignore

print("forgeryseg:", Path(forgeryseg.__file__).resolve())

from torch.utils.data import DataLoader  # noqa: E402

from forgeryseg.augment import get_train_augment, get_val_augment  # noqa: E402
from forgeryseg.dataset import PatchDataset, build_train_index  # noqa: E402
from forgeryseg.losses import BCETverskyLoss  # noqa: E402
from forgeryseg.models import builders  # noqa: E402
from forgeryseg.train import train_one_epoch, validate  # noqa: E402

# %%
# Célula 3 — Dataset root + index


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
train_samples = build_train_index(DATA_ROOT, strict=True)
labels = np.array([0 if s.is_authentic else 1 for s in train_samples], dtype=np.int64)
print("DATA_ROOT:", DATA_ROOT)
print("train samples:", len(train_samples), "auth:", int((labels == 0).sum()), "forged:", int((labels == 1).sum()))

# %%
# Célula 4 — Split (5-fold estratificado)
N_FOLDS = 5
FOLD = 0

try:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(train_samples), dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(len(train_samples)), labels)):
        folds[val_idx] = int(fold_id)
except Exception:
    print("[ERRO] scikit-learn falhou (StratifiedKFold). Usando split simples.")
    traceback.print_exc()
    folds = np.arange(len(train_samples), dtype=np.int64) % int(N_FOLDS)

train_idx = np.where(folds != int(FOLD))[0]
val_idx = np.where(folds == int(FOLD))[0]
print(f"fold={FOLD}: train={len(train_idx)} val={len(val_idx)}")

# %%
# Célula 5 — Config (patch + augs + dataloader)
PATCH_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-2

POSITIVE_PROB = 0.70
MIN_POS_PIXELS = 1

NUM_WORKERS = 2 if is_kaggle() else 0

train_aug = get_train_augment(patch_size=PATCH_SIZE, copy_move_prob=0.20)
val_aug = get_val_augment()

ds_train = PatchDataset(
    [train_samples[i] for i in train_idx.tolist()],
    patch_size=PATCH_SIZE,
    train=True,
    augment=train_aug,
    positive_prob=POSITIVE_PROB,
    min_pos_pixels=MIN_POS_PIXELS,
    seed=SEED,
)
ds_val = PatchDataset(
    [train_samples[i] for i in val_idx.tolist()],
    patch_size=PATCH_SIZE,
    train=False,
    augment=val_aug,
    seed=SEED,
)

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
# Célula 6 — Modelo (escolha aqui)
#
# Importante (offline): use `encoder_weights=None`.
MODEL_ID = "unetpp_effb7"
ENCODER_NAME = "efficientnet-b7"

try:
    model = builders.build_unetplusplus(encoder_name=ENCODER_NAME, encoder_weights=None, classes=1, strict_weights=True)
except Exception:
    print("[ERRO] falha ao construir modelo (provável falta de segmentation_models_pytorch ou encoder).")
    traceback.print_exc()
    raise

model = model.to(DEVICE)
print("model:", MODEL_ID, "encoder:", ENCODER_NAME)

# %%
# Célula 7 — Treino + checkpoint


def output_root() -> Path:
    if is_kaggle():
        return Path("/kaggle/working")
    return Path(".").resolve()


SAVE_DIR = output_root() / "outputs" / "models_seg" / MODEL_ID / f"fold_{int(FOLD)}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = SAVE_DIR / "best.pt"

criterion = BCETverskyLoss(alpha=0.7, beta=0.3, tversky_weight=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

use_amp = (DEVICE == "cuda")
best_dice = -1.0

for epoch in range(1, int(EPOCHS) + 1):
    tr = train_one_epoch(model, dl_train, criterion, optimizer, DEVICE, use_amp=use_amp, progress=True, desc="seg train")
    val_stats, val_dice = validate(model, dl_val, criterion, DEVICE, progress=True, desc="seg val")
    print(
        f"epoch {epoch:02d}/{EPOCHS} | train_loss={tr.loss:.4f} | val_loss={val_stats.loss:.4f} | dice@0.5={val_dice:.4f}"
    )

    if float(val_dice) > best_dice:
        best_dice = float(val_dice)
        ckpt = {
            "model_state": model.state_dict(),
            "config": {
                "backend": "smp",
                "arch": "unetplusplus",
                "encoder_name": ENCODER_NAME,
                "encoder_weights": None,
                "classes": 1,
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

