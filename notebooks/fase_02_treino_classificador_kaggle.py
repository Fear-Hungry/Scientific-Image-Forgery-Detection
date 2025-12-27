# %% [markdown]
# # Fase 2 — Treino do classificador (authentic vs forged)
#
# Objetivo:
# - Treinar um classificador binário para filtrar imagens provavelmente autênticas.
# - Salvar checkpoints em `/kaggle/working/outputs/models_cls/` (Kaggle) ou `outputs/models_cls/` (local).
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
print("DATA_ROOT:", DATA_ROOT)

# %%
# Célula 4 — Index (train) para classificação


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


train_samples = build_cls_index(TRAIN_IMAGES)
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
# Célula 6 — Dataset/DataLoader + transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import torchvision.transforms as T
except Exception:
    print("[ERRO] torchvision não disponível (ou falhou no import).")
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

IMAGE_SIZE = 384
BATCH_SIZE = 32
NUM_WORKERS = 2 if is_kaggle() else 4


def build_transform(train: bool) -> T.Compose:
    aug = []
    if train:
        aug += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
    aug += [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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


ds_train = ClsDataset([train_samples[i] for i in train_idx.tolist()], build_transform(train=True))
ds_val = ClsDataset([train_samples[i] for i in val_idx.tolist()], build_transform(train=False))

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
# Célula 7 — Modelo (timm preferencial; fallback torchvision)
try:
    import timm
except Exception:
    timm = None
    print("[WARN] timm indisponível; vou usar fallback do torchvision.")
    traceback.print_exc()


def build_cls_model(model_name: str) -> nn.Module:
    if timm is not None:
        try:
            m = timm.create_model(model_name, pretrained=False, num_classes=1)
            return m
        except Exception:
            print(f"[ERRO] falha ao criar modelo timm='{model_name}'.")
            traceback.print_exc()
            raise

    try:
        from torchvision.models import resnet50

        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    except Exception:
        print("[ERRO] falha ao criar fallback torchvision (resnet50).")
        traceback.print_exc()
        raise


CLS_MODEL_NAME = "tf_efficientnet_b4_ns"  # se timm disponível
model = build_cls_model(CLS_MODEL_NAME).to(DEVICE)

# %%
# Célula 8 — Treino (com progresso) + checkpoint
from time import time

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None
    print("[WARN] scikit-learn (roc_auc_score) indisponível; vou reportar só loss/accuracy.")
    traceback.print_exc()


def _compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)


pos_weight = _compute_pos_weight(y[train_idx]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

LR = 3e-4
EPOCHS = 10
WEIGHT_DECAY = 1e-2

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    losses: list[float] = []
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for x, yb in tqdm(loader, desc="val", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(x).view(-1, 1)
        loss = criterion(logits, yb)
        losses.append(float(loss.item()))
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0).reshape(-1)
    targets_np = np.concatenate(all_targets, axis=0).reshape(-1)
    probs = _sigmoid_np(logits_np)
    pred = (probs >= 0.5).astype(np.int64)
    acc = float((pred == targets_np.astype(np.int64)).mean())

    out = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc@0.5": acc,
    }
    if roc_auc_score is not None:
        try:
            out["auc"] = float(roc_auc_score(targets_np, probs))
        except Exception:
            print("[WARN] falha ao calcular AUC (roc_auc_score).")
            traceback.print_exc()
    return out


def train_one_epoch(model: nn.Module, loader: DataLoader) -> float:
    model.train()
    losses: list[float] = []
    for x, yb in tqdm(loader, desc="train", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(x).view(-1, 1)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def output_root() -> Path:
    if is_kaggle():
        return Path("/kaggle/working")
    return Path(".").resolve()


OUTPUT_DIR = output_root() / "outputs" / "models_cls" / f"fold_{int(FOLD)}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = OUTPUT_DIR / "best.pt"

best_score = -1.0

for epoch in range(1, int(EPOCHS) + 1):
    t0 = time()
    train_loss = train_one_epoch(model, dl_train)
    val_metrics = evaluate(model, dl_val)
    elapsed = time() - t0

    score = float(val_metrics.get("auc", -val_metrics["loss"]))
    print(
        f"epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
        f"acc@0.5={val_metrics['acc@0.5']:.4f} | "
        + (f"auc={val_metrics.get('auc', float('nan')):.4f} | " if "auc" in val_metrics else "")
        + f"time={elapsed:.1f}s"
    )

    if score > best_score:
        best_score = score
        ckpt = {
            "model_state": model.state_dict(),
            "config": {
                "backend": "timm" if timm is not None else "torchvision",
                "model_name": CLS_MODEL_NAME,
                "image_size": int(IMAGE_SIZE),
                "fold": int(FOLD),
                "seed": int(SEED),
            },
            "score": float(best_score),
        }
        torch.save(ckpt, BEST_PATH)
        print("  saved best ->", BEST_PATH)

print("best score:", best_score)

# %%
# Célula 9 — Tuning de limiar (favorecer recall em forged)

THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

if roc_auc_score is None:
    print("[INFO] scikit-learn indisponível; tuning simples vai usar apenas accuracy.")

try:
    from sklearn.metrics import precision_recall_fscore_support
except Exception:
    precision_recall_fscore_support = None
    print("[WARN] scikit-learn indisponível (precision_recall_fscore_support).")
    traceback.print_exc()


@torch.no_grad()
def collect_val_probs(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for x, yb in tqdm(loader, desc="collect", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x).view(-1).detach().cpu().numpy()
        probs = _sigmoid_np(logits)
        all_probs.append(probs)
        all_targets.append(yb.view(-1).cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


probs_val, y_val = collect_val_probs(model, dl_val)
print("val probs shape:", probs_val.shape)

best = None
for t in THRESHOLDS:
    pred = (probs_val >= float(t)).astype(np.int64)
    if precision_recall_fscore_support is None:
        acc = float((pred == y_val.astype(np.int64)).mean())
        print(f"t={t:.2f} acc={acc:.4f}")
        continue

    # classe 1 (forged)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, pred, labels=[1], average=None, zero_division=0)
    prec1, rec1, f11 = float(prec[0]), float(rec[0]), float(f1[0])
    print(f"t={t:.2f}  forged: recall={rec1:.4f} precision={prec1:.4f} f1={f11:.4f}")
    cand = (rec1, -prec1, t)
    if best is None or cand > best:
        best = cand

if best is not None:
    print("\nSugestão (max recall, desempate por menor precision):")
    print("recall=", best[0], "precision=-", -best[1], "threshold=", best[2])

