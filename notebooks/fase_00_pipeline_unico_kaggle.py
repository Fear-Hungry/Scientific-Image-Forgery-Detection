# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Treino + Submissão — Pipeline Kaggle (Offline)
#
# Este notebook é focado em **submissão no Kaggle** (internet OFF) e **treina por padrão**:
#
# 1) **Setup offline + checagens**
# 2) **Treino** (classificador + segmentadores)
# 3) **Inferência + `submission.csv`** via `scripts/submit_ensemble.py`
#
# ## Regras / Decisões
# - Importa código do projeto em `src/forgeryseg/` (modularizado).
# - Compatível com Kaggle **internet OFF** (instala wheels locais se existirem).
# - Não esconde erros: exceções e tracebacks aparecem.
#
# ---

# %%
# Fase 1 — Célula 1: Sanidade Kaggle (lembrete)
print("Kaggle constraints (lembrete):")
print("- Runtime <= 4h (CPU/GPU)")
print("- Internet: OFF no submit")
print("- Outputs: /kaggle/working/outputs (checkpoints)")

# %%
# Fase 1 — Célula 2: Imports + ambiente
import os
import random
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.simplefilter("default")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
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

# Em notebooks, multiprocessing pode gerar warnings/instabilidade; use 0 por padrão.
NUM_WORKERS = int(os.environ.get("FORGERYSEG_NUM_WORKERS", "0"))
if NUM_WORKERS > 0:
    print("[WARN] FORGERYSEG_NUM_WORKERS>0 em notebooks pode gerar warnings/instabilidade.")
print("NUM_WORKERS:", NUM_WORKERS)

print("python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", DEVICE)

# %%
# Fase 1 — Célula 2b: Instalação offline (wheels) — NÃO resolve deps
#
# Estruturas suportadas:
# - `/kaggle/input/<dataset>/wheels/*.whl`
# - `/kaggle/input/<dataset>/recodai_bundle/wheels/*.whl`
#
# Observação: instalamos com `--no-deps` para não tentar instalar dependências do torch offline.
import subprocess

_install_wheels_env = os.environ.get("FORGERYSEG_INSTALL_WHEELS", "")
if _install_wheels_env == "":
    INSTALL_WHEELS = bool(is_kaggle())
else:
    INSTALL_WHEELS = _install_wheels_env.strip().lower() in {"1", "true", "yes", "y", "on"}


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
    Procura por `package_dir_name/__init__.py` em `/kaggle/input/*` e adiciona o root correspondente ao `sys.path`.

    Nota: não excluímos o dataset da competição, pois alguns usuários empacotam o código junto com os dados
    em um único Kaggle Dataset. A busca é rasa (não percorre imagens), então o custo é baixo.
    """
    added: list[Path] = []
    if not is_kaggle():
        return added

    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return added

    for ds in sorted(kaggle_input.glob("*")):
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

    print(f"[LOCAL IMPORT] não encontrei '{package_dir_name}/__init__.py' em `/kaggle/input/*`.")
    return added


OFFLINE_BUNDLE = _find_offline_bundle()
if not INSTALL_WHEELS:
    print("[OFFLINE INSTALL] FORGERYSEG_INSTALL_WHEELS=0; pulando instalação de wheels.")
elif OFFLINE_BUNDLE is None:
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
        try:
            subprocess.check_call(cmd)
            print("[OFFLINE INSTALL] OK.")
        except Exception:
            print("[OFFLINE INSTALL] falhou; seguindo sem wheels. Verifique compatibilidade das wheels.")
            traceback.print_exc()

# %%
# Fase 1 — Célula 2c: Import do projeto (src/forgeryseg)

def _maybe_add_src_to_syspath(src_root: Path) -> bool:
    if (src_root / "forgeryseg" / "__init__.py").exists() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
        return True
    return False


try:
    import forgeryseg  # type: ignore
except Exception:
    _maybe_add_src_to_syspath(Path("src").resolve())
    if is_kaggle():
        add_local_package_to_syspath("forgeryseg")
    import forgeryseg  # type: ignore

FORGERYSEG_FILE = Path(forgeryseg.__file__).resolve()
print("forgeryseg:", FORGERYSEG_FILE)

PROJECT_ROOT: Path | None = None
try:
    if FORGERYSEG_FILE.parent.name == "forgeryseg" and FORGERYSEG_FILE.parent.parent.name == "src":
        PROJECT_ROOT = FORGERYSEG_FILE.parents[2]
except Exception:
    PROJECT_ROOT = None
print("PROJECT_ROOT:", PROJECT_ROOT)

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from forgeryseg.augment import get_train_augment, get_val_augment  # noqa: E402
from forgeryseg.dataset import PatchDataset, build_train_index  # noqa: E402
from forgeryseg.losses import BCETverskyLoss  # noqa: E402
from forgeryseg.models import builders  # noqa: E402
from forgeryseg.models.classifier import build_classifier, compute_pos_weight  # noqa: E402
from forgeryseg.offline import configure_cache_dirs  # noqa: E402
from forgeryseg.train import train_one_epoch, validate  # noqa: E402

# %%
# Fase 1 — Célula 2d: Cache dirs (offline weights)
#
# Para usar pesos pré-treinados no Kaggle com internet OFF, anexe um Dataset contendo caches e aponte aqui.
# Exemplo de estrutura sugerida:
# - <CACHE_ROOT>/torch  (TORCH_HOME)
# - <CACHE_ROOT>/hf     (HF_HOME)
CACHE_ROOT = os.environ.get("FORGERYSEG_CACHE_ROOT", "")
if CACHE_ROOT:
    configure_cache_dirs(CACHE_ROOT)
    print("[CACHE] FORGERYSEG_CACHE_ROOT:", CACHE_ROOT)
else:
    print("[CACHE] FORGERYSEG_CACHE_ROOT vazio (seguindo sem caches).")

# %%
# Fase 1 — Célula 3: Dataset root + contagens


def find_dataset_root() -> Path:
    def _looks_like_root(p: Path) -> bool:
        return (p / "train_images").exists() and (p / "test_images").exists()

    if is_kaggle():
        base = Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection")
        if _looks_like_root(base):
            return base
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for ds in sorted(kaggle_input.glob("*")):
                if _looks_like_root(ds):
                    return ds

    local_candidates = [
        Path("data/recodai").resolve(),
        Path("data").resolve(),
    ]
    for cand in local_candidates:
        if _looks_like_root(cand):
            return cand

    raise FileNotFoundError("Dataset não encontrado. No Kaggle: anexe o dataset da competição.")


DATA_ROOT = find_dataset_root()
train_samples = build_train_index(DATA_ROOT, strict=False)
train_labels = np.array([0 if s.is_authentic else 1 for s in train_samples], dtype=np.int64)

print("DATA_ROOT:", DATA_ROOT)
print("train samples:", len(train_samples), "auth:", int((train_labels == 0).sum()), "forged:", int((train_labels == 1).sum()))

# %%
# Fase 1 — Célula 4: Config de treino (liga/desliga)
def _has_any_ckpt(dir_name: str, pattern: str) -> bool:
    # Procura primeiro em /kaggle/input (datasets anexados), depois em outputs/ local.
    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                for base in (ds, ds / "recodai_bundle"):
                    cand = base / "outputs" / dir_name
                    if cand.exists():
                        if any(cand.glob(pattern)):
                            return True
    local = (Path("/kaggle/working") if is_kaggle() else Path(".").resolve()) / "outputs" / dir_name
    return local.exists() and any(local.glob(pattern))


HAS_SEG_CKPT = _has_any_ckpt("models_seg", "*/*/best.pt")
HAS_CLS_CKPT = _has_any_ckpt("models_cls", "fold_*/best.pt")

# Utils
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if v == "":
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


ALLOW_DOWNLOAD = _env_bool("FORGERYSEG_ALLOW_DOWNLOAD", default=not is_kaggle())
# No Kaggle, a internet é OFF por padrão. Permita downloads apenas se o usuário pedir explicitamente.
OFFLINE_NO_DOWNLOAD = bool(is_kaggle() and not ALLOW_DOWNLOAD)
if OFFLINE_NO_DOWNLOAD:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("[OFFLINE] downloads disabled (Kaggle offline).")


N_FOLDS = int(os.environ.get("FORGERYSEG_N_FOLDS", "5"))
FOLD = int(os.environ.get("FORGERYSEG_FOLD", "0"))

FAST_TRAIN = _env_bool("FORGERYSEG_FAST_TRAIN", default=bool(is_kaggle() and not HAS_SEG_CKPT))

print("FAST_TRAIN:", FAST_TRAIN)
print("HAS_SEG_CKPT:", HAS_SEG_CKPT)
print("HAS_CLS_CKPT:", HAS_CLS_CKPT)

# Em notebook de submissão Kaggle, por padrão TREINAMOS.
RUN_TRAIN_CLS = _env_bool("FORGERYSEG_RUN_TRAIN_CLS", default=True)
RUN_TRAIN_SEG = _env_bool("FORGERYSEG_RUN_TRAIN_SEG", default=True)

print("RUN_TRAIN_CLS:", RUN_TRAIN_CLS)
print("RUN_TRAIN_SEG:", RUN_TRAIN_SEG)
print("N_FOLDS:", N_FOLDS)
print("FOLD:", FOLD)

# %%
# Fase 2 — Célula 5: Split (folds)
try:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(train_samples), dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(len(train_samples)), train_labels)):
        folds[val_idx] = int(fold_id)
except Exception:
    print("[ERRO] scikit-learn falhou (StratifiedKFold). Usando split simples.")
    traceback.print_exc()
    folds = np.arange(len(train_samples), dtype=np.int64) % int(N_FOLDS)

train_idx = np.where(folds != int(FOLD))[0]
val_idx = np.where(folds == int(FOLD))[0]
print(f"fold={FOLD}: train={len(train_idx)} val={len(val_idx)}")

# %%
# Fase 2 — Célula 6: Treino do classificador (opcional)
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

CLS_MODEL_NAME = "tf_efficientnet_b4_ns"
CLS_IMAGE_SIZE = 384
CLS_BATCH_SIZE = 32
CLS_EPOCHS = int(os.environ.get("FORGERYSEG_CLS_EPOCHS", "15"))
CLS_PATIENCE = 3
CLS_LR = 3e-4
CLS_WEIGHT_DECAY = 1e-2
# Preferir recall (evitar falsos negativos): só pule a segmentação quando tiver alta confiança de autenticidade.
CLS_SKIP_THRESHOLD = float(os.environ.get("FORGERYSEG_CLS_SKIP_THRESHOLD", "0.30"))
# Scheduler (PDF sugere ReduceLROnPlateau ou cosine; usamos ReduceLROnPlateau por padrão).
CLS_USE_SCHEDULER = _env_bool("FORGERYSEG_CLS_USE_SCHEDULER", default=True)
CLS_LR_SCHED_PATIENCE = int(os.environ.get("FORGERYSEG_CLS_LR_SCHED_PATIENCE", "2"))
CLS_LR_SCHED_FACTOR = float(os.environ.get("FORGERYSEG_CLS_LR_SCHED_FACTOR", "0.5"))
# Pesos pré-treinados ajudam; em Kaggle offline usamos cache local e fazemos fallback se faltar.
CLS_PRETRAINED = _env_bool("FORGERYSEG_CLS_PRETRAINED", default=True)


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
    def __init__(self, samples, transform: T.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[int(idx)]
        from PIL import Image

        img = Image.open(s.image_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([0.0 if s.is_authentic else 1.0], dtype=torch.float32)
        return x, y


if RUN_TRAIN_CLS:
    try:
        ds_cls_train = ClsDataset([train_samples[i] for i in train_idx.tolist()], build_transform(train=True))
        ds_cls_val = ClsDataset([train_samples[i] for i in val_idx.tolist()], build_transform(train=False))
    
        num_workers = NUM_WORKERS
        dl_cls_train = DataLoader(ds_cls_train, batch_size=CLS_BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE == "cuda"), drop_last=True)
        dl_cls_val = DataLoader(ds_cls_val, batch_size=CLS_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE == "cuda"), drop_last=False)
    
        try:
            cls_model = build_classifier(model_name=CLS_MODEL_NAME, pretrained=CLS_PRETRAINED, num_classes=1).to(DEVICE)
        except Exception:
            if CLS_PRETRAINED:
                print("[CLS] falha ao carregar pesos pré-treinados; fallback para pretrained=False.")
                traceback.print_exc()
                cls_model = build_classifier(model_name=CLS_MODEL_NAME, pretrained=False, num_classes=1).to(DEVICE)
            else:
                raise
        pos_weight = torch.tensor(compute_pos_weight(train_labels[train_idx]), dtype=torch.float32, device=DEVICE)
        cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        cls_optimizer = torch.optim.AdamW(cls_model.parameters(), lr=CLS_LR, weight_decay=CLS_WEIGHT_DECAY)
        cls_scheduler = None
        if CLS_USE_SCHEDULER:
            cls_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                cls_optimizer,
                mode="min",
                patience=int(CLS_LR_SCHED_PATIENCE),
                factor=float(CLS_LR_SCHED_FACTOR),
            )
        cls_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    
        @torch.no_grad()
        def cls_eval() -> dict:
            cls_model.eval()
            losses = []
            logits_all = []
            y_all = []
            for x, yb in tqdm(dl_cls_val, desc="cls val", leave=False):
                x = x.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                logits = cls_model(x).view(-1, 1)
                loss = cls_criterion(logits, yb)
                losses.append(float(loss.item()))
                logits_all.append(logits.detach().cpu().numpy())
                y_all.append(yb.detach().cpu().numpy())
            logits_np = np.concatenate(logits_all, axis=0).reshape(-1)
            y_np = np.concatenate(y_all, axis=0).reshape(-1)
            probs = 1.0 / (1.0 + np.exp(-logits_np))
            acc = float(((probs >= 0.5).astype(np.int64) == y_np.astype(np.int64)).mean())
            out = {"loss": float(np.mean(losses)) if losses else float("nan"), "acc@0.5": acc}
            try:
                from sklearn.metrics import roc_auc_score
    
                out["auc"] = float(roc_auc_score(y_np, probs))
            except Exception:
                traceback.print_exc()
            return out
    
        def cls_train_one_epoch() -> float:
            cls_model.train()
            losses = []
            for x, yb in tqdm(dl_cls_train, desc="cls train", leave=False):
                x = x.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                cls_optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    logits = cls_model(x).view(-1, 1)
                    loss = cls_criterion(logits, yb)
                cls_scaler.scale(loss).backward()
                cls_scaler.step(cls_optimizer)
                cls_scaler.update()
                losses.append(float(loss.item()))
            return float(np.mean(losses)) if losses else float("nan")
    
        def output_root() -> Path:
            return Path("/kaggle/working") if is_kaggle() else Path(".").resolve()
    
        cls_save_dir = output_root() / "outputs" / "models_cls" / f"fold_{int(FOLD)}"
        cls_save_dir.mkdir(parents=True, exist_ok=True)
        cls_best_path = cls_save_dir / "best.pt"
    
        best_score = -1.0
        best_epoch = 0
        for epoch in range(1, int(CLS_EPOCHS) + 1):
            tr_loss = cls_train_one_epoch()
            val = cls_eval()
            if cls_scheduler is not None:
                cls_scheduler.step(float(val["loss"]))
            score = float(val.get("auc", -val["loss"]))
            print(f"[CLS] epoch {epoch:02d}/{CLS_EPOCHS} | train_loss={tr_loss:.4f} | val={val}")
            if score > best_score:
                best_score = score
                best_epoch = int(epoch)
                ckpt = {
                    "model_state": cls_model.state_dict(),
                    "config": {
                        "backend": "timm",
                        "model_name": CLS_MODEL_NAME,
                        "image_size": int(CLS_IMAGE_SIZE),
                        "pretrained": bool(CLS_PRETRAINED),
                        "fold": int(FOLD),
                        "seed": int(SEED),
                    },
                    "score": float(best_score),
                }
                torch.save(ckpt, cls_best_path)
                print("[CLS] saved best ->", cls_best_path)
            if CLS_PATIENCE and best_epoch and (int(epoch) - int(best_epoch) >= int(CLS_PATIENCE)):
                print(f"[CLS] early stopping: sem melhora por {CLS_PATIENCE} épocas (best_epoch={best_epoch}).")
                break
    
        print("[CLS] done. best score:", best_score)
    except Exception:
        print("[CLS] erro no treino; seguindo para próxima fase.")
        traceback.print_exc()
else:
    print("[CLS] RUN_TRAIN_CLS=False (pulando).")

# %%
# Fase 3 — Célula 7: Treino de segmentação (opcional)
SEG_PATCH_SIZE = int(os.environ.get("FORGERYSEG_SEG_PATCH_SIZE", "512"))
SEG_COPY_MOVE_PROB = 0.20
SEG_BATCH_SIZE = int(os.environ.get("FORGERYSEG_SEG_BATCH_SIZE", "4"))
SEG_EPOCHS = int(os.environ.get("FORGERYSEG_SEG_EPOCHS", "40"))
SEG_LR = 1e-3
SEG_WEIGHT_DECAY = 1e-2
SEG_PATIENCE = 3
# Scheduler (PDF sugere ReduceLROnPlateau ou cosine; usamos ReduceLROnPlateau por padrão).
SEG_USE_SCHEDULER = _env_bool("FORGERYSEG_SEG_USE_SCHEDULER", default=True)
SEG_LR_SCHED_PATIENCE = int(os.environ.get("FORGERYSEG_SEG_LR_SCHED_PATIENCE", "2"))
SEG_LR_SCHED_FACTOR = float(os.environ.get("FORGERYSEG_SEG_LR_SCHED_FACTOR", "0.5"))
# Pesos pré-treinados: preferível (offline via cache); fallback para None se falhar.
_seg_weights_env = os.environ.get("FORGERYSEG_SEG_ENCODER_WEIGHTS", "imagenet")
if str(_seg_weights_env).strip().lower() in {"", "none", "null", "false", "0"}:
    SEG_ENCODER_WEIGHTS = None
else:
    SEG_ENCODER_WEIGHTS = str(_seg_weights_env)

# Para performance máxima, treine mais de uma arquitetura e faça ensemble na inferência.
# Preset inspirado no PDF "Pipeline Completo..." (Unet++ + DeepLabV3+ + SegFormer).
SEG_TRAIN_SPECS = [
    {
        "model_id": "unetpp_effnet_b7",
        "arch": "unetplusplus",
        "encoder_name": "efficientnet-b7",
        "encoder_fallback": "efficientnet-b4",
    },
    {
        "model_id": "deeplabv3p_tu_resnest101e",
        "arch": "deeplabv3plus",
        "encoder_name": "tu-resnest101e",
        "encoder_fallback": "resnet101",
    },
    {
        "model_id": "segformer_mit_b3",
        "arch": "segformer",
        "encoder_name": "mit_b3",
        "encoder_fallback": "mit_b2",
    },
]

if FAST_TRAIN:
    print("[SEG] FAST_TRAIN=True -> preset rápido (1 modelo / poucas épocas).")
    SEG_EPOCHS = min(int(SEG_EPOCHS), 2)
    if OFFLINE_NO_DOWNLOAD:
        SEG_TRAIN_SPECS = [
            {"model_id": "unet_resnet34", "arch": "unet", "encoder_name": "resnet34", "encoder_weights": None},
        ]
    else:
        SEG_TRAIN_SPECS = [
            {"model_id": "unet_tu_convnext_small", "arch": "unet", "encoder_name": "tu-convnext_small"},
        ]

if RUN_TRAIN_SEG:
    try:
        train_aug = get_train_augment(patch_size=SEG_PATCH_SIZE, copy_move_prob=SEG_COPY_MOVE_PROB)
        val_aug = get_val_augment()
    
        ds_seg_train = PatchDataset([train_samples[i] for i in train_idx.tolist()], patch_size=SEG_PATCH_SIZE, train=True, augment=train_aug, positive_prob=0.7, seed=SEED)
        ds_seg_val = PatchDataset([train_samples[i] for i in val_idx.tolist()], patch_size=SEG_PATCH_SIZE, train=False, augment=val_aug, seed=SEED)
    
        num_workers = NUM_WORKERS
        dl_seg_train = DataLoader(ds_seg_train, batch_size=SEG_BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE == "cuda"), drop_last=True)
        dl_seg_val = DataLoader(ds_seg_val, batch_size=SEG_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE == "cuda"), drop_last=False)
    
        def output_root() -> Path:
            return Path("/kaggle/working") if is_kaggle() else Path(".").resolve()
    
        use_amp = (DEVICE == "cuda")
    
        def build_seg_model(arch: str, encoder_name: str, encoder_weights: str | None) -> nn.Module:
            arch = str(arch).lower()
            if arch == "unet":
                return builders.build_unet(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    classes=1,
                    strict_weights=True,
                )
            if arch in {"unetplusplus", "unetpp"}:
                return builders.build_unetplusplus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    classes=1,
                    strict_weights=True,
                )
            if arch in {"deeplabv3plus", "deeplabv3+", "deeplabv3p"}:
                return builders.build_deeplabv3plus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    classes=1,
                    strict_weights=True,
                )
            if arch in {"segformer", "mit"}:
                return builders.build_segformer(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    classes=1,
                    strict_weights=True,
                )
            raise ValueError(f"SEG arch inválida: {arch!r}")
    
        try:
            available_encoders = set(builders.available_encoders())
        except Exception:
            available_encoders = set()

        for spec in SEG_TRAIN_SPECS:
            model_id = str(spec["model_id"])
            arch = str(spec.get("arch", "unetplusplus"))
            encoder_name = str(spec.get("encoder_name", "efficientnet-b4"))
            encoder_fallback = spec.get("encoder_fallback", None)
            encoder_weights: str | None = spec.get("encoder_weights", SEG_ENCODER_WEIGHTS)

            if available_encoders and encoder_name not in available_encoders:
                if encoder_fallback and str(encoder_fallback) in available_encoders:
                    print(
                        f"[SEG] encoder {encoder_name!r} indisponível; usando fallback {encoder_fallback!r}."
                    )
                    encoder_name = str(encoder_fallback)
                else:
                    print(f"[SEG] encoder {encoder_name!r} não listado em SMP; tentando mesmo assim.")
    
            try:
                seg_model = build_seg_model(arch, encoder_name, encoder_weights).to(DEVICE)
            except Exception:
                if encoder_weights is not None:
                    print(f"[SEG] falha ao carregar encoder_weights={encoder_weights!r}; fallback para None.")
                    traceback.print_exc()
                    encoder_weights = None
                    seg_model = build_seg_model(arch, encoder_name, encoder_weights).to(DEVICE)
                else:
                    raise
    
            seg_criterion = BCETverskyLoss(alpha=0.7, beta=0.3, tversky_weight=1.0)
            seg_optimizer = torch.optim.AdamW(seg_model.parameters(), lr=SEG_LR, weight_decay=SEG_WEIGHT_DECAY)
            seg_scheduler = None
            if SEG_USE_SCHEDULER:
                seg_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    seg_optimizer,
                    mode="min",
                    patience=int(SEG_LR_SCHED_PATIENCE),
                    factor=float(SEG_LR_SCHED_FACTOR),
                )
    
            seg_save_dir = output_root() / "outputs" / "models_seg" / model_id / f"fold_{int(FOLD)}"
            seg_save_dir.mkdir(parents=True, exist_ok=True)
            seg_best_path = seg_save_dir / "best.pt"
    
            best_dice = -1.0
            best_epoch = 0
            for epoch in range(1, int(SEG_EPOCHS) + 1):
                tr = train_one_epoch(seg_model, dl_seg_train, seg_criterion, seg_optimizer, DEVICE, use_amp=use_amp, progress=True, desc=f"seg train ({model_id})")
                val_stats, val_dice = validate(seg_model, dl_seg_val, seg_criterion, DEVICE, progress=True, desc=f"seg val ({model_id})")
                if seg_scheduler is not None:
                    seg_scheduler.step(float(val_stats.loss))
                print(f"[SEG {model_id}] epoch {epoch:02d}/{SEG_EPOCHS} | train_loss={tr.loss:.4f} | val_loss={val_stats.loss:.4f} | dice@0.5={val_dice:.4f}")
                if float(val_dice) > best_dice:
                    best_dice = float(val_dice)
                    best_epoch = int(epoch)
                    ckpt = {
                        "model_state": seg_model.state_dict(),
                        "config": {
                            "backend": "smp",
                            "arch": arch,
                            "encoder_name": encoder_name,
                            "encoder_weights": encoder_weights,
                            "classes": 1,
                            "model_id": model_id,
                            "patch_size": int(SEG_PATCH_SIZE),
                            "fold": int(FOLD),
                            "seed": int(SEED),
                        },
                        "score": float(best_dice),
                    }
                    torch.save(ckpt, seg_best_path)
                    print("[SEG] saved best ->", seg_best_path)
                if SEG_PATIENCE and best_epoch and (int(epoch) - int(best_epoch) >= int(SEG_PATIENCE)):
                    print(f"[SEG {model_id}] early stopping: sem melhora por {SEG_PATIENCE} épocas (best_epoch={best_epoch}).")
                    break
    
            print(f"[SEG {model_id}] done. best dice:", best_dice)
    except Exception:
        print("[SEG] erro no treino; seguindo para submissão.")
        traceback.print_exc()
else:
    print("[SEG] RUN_TRAIN_SEG=False (pulando).")

# %% [markdown]
# ## Fase 4 — Geração de `submission.csv` (roteiro oficial)
#
# A competição pede **segmentação** de regiões de copy-move e usa uma variante do **F1-score**,
# portanto o foco é equilibrar precisão e recall. A métrica oficial usa **RLE (Run-Length Encoding)**.
#
# Abaixo está o **roteiro completo** para montar o notebook de submissão:
#
# ### 1) Importar bibliotecas e ler dados
# - Define os caminhos de treino e teste no Kaggle.
# - Lista as imagens de teste para gerar o CSV.
#
# ### 2) Funções de codificação RLE
# - Usa RLE para converter máscaras binárias em string.
#
# ### 3) Lógica de predição (baseline)
# - Baseline simples: assume todas as imagens como `authentic`.
# - Opcional: gerar máscara via modelo e converter para RLE.
#
# ### 4) Gerar e salvar o arquivo de submissão
# - Salva `submission.csv` em `/kaggle/working/`.

# %%
# Fase 4 — Célula 8: Imports + leitura de dados (roteiro oficial)
import json
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# No Kaggle, o dataset costuma ficar aqui; se não existir, usa o DATA_ROOT detectado.
DATA_DIR = Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection")
if not DATA_DIR.exists():
    DATA_DIR = DATA_ROOT

TRAIN_DIR = DATA_DIR / "train_images"
TEST_DIR = DATA_DIR / "test_images"
TRAIN_MASKS = DATA_DIR / "train_masks"  # se houver

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
test_images = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in VALID_EXTS])
test_by_id = {p.stem: p for p in test_images}

sample_submission_path = DATA_DIR / "sample_submission.csv"
sample_submission = None
case_ids = [p.stem for p in test_images]
if sample_submission_path.exists():
    sample_submission = pd.read_csv(sample_submission_path)
    if "case_id" in sample_submission.columns:
        case_ids = sample_submission["case_id"].tolist()

print("DATA_DIR:", DATA_DIR)
print("TEST_DIR:", TEST_DIR)
print("#test images:", len(test_images))

# %%
# Fase 4 — Célula 9: Funções RLE (roteiro oficial)
# A métrica oficial usa JSON para a lista de pares [start, length, ...]
# e suporta múltiplas instâncias separadas por ';'.
def _rle_encode_single(mask: np.ndarray, fg_val: int = 1) -> list[int]:
    # Kaggle oficial usa ordem coluna-major (Fortran).
    dots = np.where(mask.flatten(order="F") == fg_val)[0]
    run_lengths: list[int] = []
    prev = -2
    for b in dots:
        b = int(b)
        if b > prev + 1:
            run_lengths.extend([b + 1, 0])  # start 1-based
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def rle_encode(masks: list[np.ndarray] | np.ndarray, fg_val: int = 1) -> str:
    if isinstance(masks, np.ndarray):
        masks = [masks]
    parts: list[str] = []
    for m in masks:
        runs = _rle_encode_single(m, fg_val=fg_val)
        if runs:
            parts.append(json.dumps(runs))
    if not parts:
        return "authentic"
    return ";".join(parts)


def encode_submission(mask_union: np.ndarray) -> str:
    if int(mask_union.sum()) <= 0:
        return "authentic"
    return rle_encode(mask_union.astype(np.uint8))


def rle_decode(annotation: str, shape: tuple[int, int]) -> list[np.ndarray]:
    text = annotation.strip()
    if text == "" or text.lower() == "authentic":
        return []
    masks: list[np.ndarray] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        runs = json.loads(part)
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, length in zip(runs[0::2], runs[1::2]):
            if int(length) <= 0:
                continue
            start_index = int(start) - 1  # 1-based -> 0-based
            end_index = start_index + int(length)
            mask[start_index:end_index] = 1
        masks.append(mask.reshape(shape, order="F"))
    return masks

# %%
# Fase 4 — Célula 10: Baseline simples (tudo authentic)
if sample_submission is not None:
    submissions = sample_submission.copy()
    submissions["annotation"] = "authentic"
else:
    submissions = pd.DataFrame(
        {
            "case_id": case_ids,
            "annotation": ["authentic"] * len(case_ids),
        }
    )

submissions.head()

# %%
# Fase 4 — Célula 11: Exemplo de loop com modelo (opcional)
# Ative com FORGERYSEG_USE_MODEL_SUBMISSION=1 e defina `model`.
USE_MODEL_SUBMISSION = _env_bool("FORGERYSEG_USE_MODEL_SUBMISSION", default=False)
THRESHOLD = float(os.environ.get("FORGERYSEG_SUBMISSION_THRESHOLD", "0.5"))

submissions_from_model = None
if USE_MODEL_SUBMISSION:
    if "model" not in globals():
        raise RuntimeError("Defina a variável `model` antes de ativar FORGERYSEG_USE_MODEL_SUBMISSION=1.")

    annotations = []
    for case_id in case_ids:
        key = str(case_id)
        if key not in test_by_id:
            raise FileNotFoundError(f"Não encontrei imagem para case_id={case_id!r}")
        img_path = test_by_id[key]
        # carregue e processe a imagem
        img = np.array(Image.open(img_path)) / 255.0
        # modelo deve gerar um mapa de probabilidade ou máscara
        pred_prob = model.predict(img[None])[0]  # exemplo
        binary_mask = (pred_prob > THRESHOLD).astype(np.uint8)
        annotations.append(encode_submission(binary_mask))

    submissions_from_model = pd.DataFrame(
        {
            "case_id": case_ids,
            "annotation": annotations,
        }
    )

    submissions_from_model.head()

# %%
# Fase 4 — Célula 12: Salvar submission.csv (roteiro oficial)
def output_root() -> Path:
    return Path("/kaggle/working") if is_kaggle() else Path(".").resolve()


def _write_submission_csv(submissions_to_save: pd.DataFrame) -> Path:
    submission_path = output_root() / "submission.csv"
    pd.DataFrame(submissions_to_save).to_csv(submission_path, index=False)
    return submission_path


RUN_SUBMISSION_SIMPLE = _env_bool("FORGERYSEG_RUN_SUBMISSION_SIMPLE", default=False)
print("RUN_SUBMISSION_SIMPLE:", RUN_SUBMISSION_SIMPLE)

if RUN_SUBMISSION_SIMPLE:
    submissions_to_save = submissions_from_model if USE_MODEL_SUBMISSION else submissions
    submission_path = _write_submission_csv(submissions_to_save)
    print("Wrote:", submission_path)

# %% [markdown]
# ## Fase 4b — Submissão via `submit_ensemble.py` (opcional)
#
# - Usa os checkpoints em `outputs/models_seg/...`.
# - Respeita o `configs/infer_ensemble.json` (inclui gate do classificador e pesos do ensemble).
#
# Para desligar/ligar: `FORGERYSEG_RUN_SUBMISSION_SCRIPT=0|1`.

# %%
# Fase 4b — Célula 13: Gerar submission.csv via script (opcional)
RUN_SUBMISSION_SCRIPT = _env_bool("FORGERYSEG_RUN_SUBMISSION_SCRIPT", default=bool(is_kaggle()))
print("RUN_SUBMISSION_SCRIPT:", RUN_SUBMISSION_SCRIPT)


def _find_submit_ensemble_script() -> Path:
    candidates: list[Path] = []
    if PROJECT_ROOT is not None:
        candidates.append(PROJECT_ROOT / "scripts" / "submit_ensemble.py")
    candidates.append(Path("scripts/submit_ensemble.py").resolve())

    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                for base in (ds, ds / "recodai_bundle"):
                    p = base / "scripts" / "submit_ensemble.py"
                    if p.exists():
                        candidates.append(p)

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Não encontrei `scripts/submit_ensemble.py`.\n"
        "- Solução (Kaggle): anexe o dataset do repositório (bundle) contendo `scripts/`.\n"
        "- Solução (local): rode a partir do root do repo (onde existe `scripts/`)."
    )


def _find_infer_cfg_path() -> Path | None:
    candidates: list[Path] = []
    if PROJECT_ROOT is not None:
        candidates.append(PROJECT_ROOT / "configs" / "infer_ensemble.json")
    candidates.append(Path("configs/infer_ensemble.json").resolve())

    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                for base in (ds, ds / "recodai_bundle"):
                    p = base / "configs" / "infer_ensemble.json"
                    if p.exists():
                        candidates.append(p)

    for p in candidates:
        if p.exists():
            return p
    return None


def _find_models_dir_with_ckpt() -> Path | None:
    candidates: list[Path] = []
    if PROJECT_ROOT is not None:
        candidates.append(PROJECT_ROOT / "outputs" / "models_seg")
    candidates.append(Path("outputs/models_seg").resolve())

    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                for base in (ds, ds / "recodai_bundle"):
                    cand = base / "outputs" / "models_seg"
                    if cand.exists():
                        candidates.append(cand)

    for cand in candidates:
        if any(cand.glob("*/*/best.pt")):
            return cand
    return None


if RUN_SUBMISSION_SCRIPT:
    try:
        submit_script = _find_submit_ensemble_script()
        infer_cfg_path = _find_infer_cfg_path()

        submission_path = output_root() / "submission.csv"
        models_dir = _find_models_dir_with_ckpt()

        cmd = [
            sys.executable,
            str(submit_script),
            "--data-root",
            str(DATA_ROOT),
            "--out-csv",
            str(submission_path),
        ]
        if models_dir is not None:
            cmd += ["--models-dir", str(models_dir)]
            if infer_cfg_path is not None:
                cmd += ["--config", str(infer_cfg_path)]

            print("[SUBMISSION] script:", submit_script)
            if infer_cfg_path is not None:
                print("[SUBMISSION] cfg:", infer_cfg_path)
            print("[SUBMISSION] running:", " ".join(cmd))
            subprocess.check_call(cmd)
            print("[SUBMISSION] wrote:", submission_path)
        else:
            print("[SUBMISSION] nenhum checkpoint encontrado; pulando submit_ensemble.py.")
            RUN_SUBMISSION_SIMPLE = True
    except Exception:
        print("[SUBMISSION] erro ao rodar submit_ensemble.py; fallback para baseline 'authentic'.")
        traceback.print_exc()
        submission_path = _write_submission_csv(submissions)
        print("[SUBMISSION] wrote fallback:", submission_path)
        RUN_SUBMISSION_SIMPLE = True
else:
    print("[SUBMISSION] RUN_SUBMISSION_SCRIPT=False (pulando).")
