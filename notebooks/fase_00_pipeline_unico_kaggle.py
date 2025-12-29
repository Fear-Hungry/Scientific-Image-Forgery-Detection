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
from forgeryseg.models import builders, dinov2  # noqa: E402
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

# DINO-only: pipeline simples e 100% offline (sem timm / sem downloads).
DINO_ONLY = _env_bool("FORGERYSEG_DINO_ONLY", default=bool(is_kaggle()))

# Em notebook de submissão Kaggle, por padrão TREINAMOS.
RUN_TRAIN_CLS = _env_bool("FORGERYSEG_RUN_TRAIN_CLS", default=not DINO_ONLY)
RUN_TRAIN_SEG = _env_bool("FORGERYSEG_RUN_TRAIN_SEG", default=not DINO_ONLY)
RUN_TRAIN_DINO = _env_bool("FORGERYSEG_RUN_TRAIN_DINO", default=bool(DINO_ONLY))

print("DINO_ONLY:", DINO_ONLY)
print("RUN_TRAIN_CLS:", RUN_TRAIN_CLS)
print("RUN_TRAIN_SEG:", RUN_TRAIN_SEG)
print("RUN_TRAIN_DINO:", RUN_TRAIN_DINO)
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
CLS_BACKEND = os.environ.get("FORGERYSEG_CLS_BACKEND", "timm").strip().lower()
CLS_HF_MODEL_ID = os.environ.get("FORGERYSEG_CLS_HF_MODEL_ID", "metaresearch/dinov2")
CLS_FREEZE_ENCODER = _env_bool("FORGERYSEG_CLS_FREEZE_ENCODER", default=True)
CLS_LOCAL_FILES_ONLY = _env_bool("FORGERYSEG_CLS_LOCAL_FILES_ONLY", default=OFFLINE_NO_DOWNLOAD)
CLS_CLASSIFIER_HIDDEN = int(os.environ.get("FORGERYSEG_CLS_HIDDEN", "0"))
CLS_CLASSIFIER_DROPOUT = float(os.environ.get("FORGERYSEG_CLS_DROPOUT", "0.1"))
CLS_USE_CLS_TOKEN = _env_bool("FORGERYSEG_CLS_USE_CLS_TOKEN", default=True)
_cls_default_size = "392" if CLS_BACKEND in {"dinov2", "hf"} else "384"
CLS_IMAGE_SIZE = int(os.environ.get("FORGERYSEG_CLS_IMAGE_SIZE", _cls_default_size))
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
# Pesos pré-treinados ajudam; em Kaggle offline use cache local (falha se faltar).
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


if RUN_TRAIN_CLS and not DINO_ONLY:
    ds_cls_train = ClsDataset([train_samples[i] for i in train_idx.tolist()], build_transform(train=True))
    ds_cls_val = ClsDataset([train_samples[i] for i in val_idx.tolist()], build_transform(train=False))

    num_workers = NUM_WORKERS
    dl_cls_train = DataLoader(ds_cls_train, batch_size=CLS_BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE == "cuda"), drop_last=True)
    dl_cls_val = DataLoader(ds_cls_val, batch_size=CLS_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE == "cuda"), drop_last=False)

    cls_kwargs = {}
    if CLS_BACKEND in {"dinov2", "hf"}:
        cls_kwargs = {
            "hf_model_id": CLS_HF_MODEL_ID,
            "local_files_only": CLS_LOCAL_FILES_ONLY,
            "freeze_encoder": CLS_FREEZE_ENCODER,
            "classifier_hidden": CLS_CLASSIFIER_HIDDEN,
            "classifier_dropout": CLS_CLASSIFIER_DROPOUT,
            "use_cls_token": CLS_USE_CLS_TOKEN,
        }
    cls_model = build_classifier(
        model_name=CLS_MODEL_NAME,
        pretrained=CLS_PRETRAINED,
        num_classes=1,
        backend=CLS_BACKEND,
        **cls_kwargs,
    ).to(DEVICE)
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
            ckpt_config = {
                "backend": CLS_BACKEND,
                "model_name": CLS_MODEL_NAME,
                "image_size": int(CLS_IMAGE_SIZE),
                "pretrained": bool(CLS_PRETRAINED),
                "fold": int(FOLD),
                "seed": int(SEED),
            }
            if CLS_BACKEND in {"dinov2", "hf"}:
                ckpt_config.update(
                    {
                        "hf_model_id": CLS_HF_MODEL_ID,
                        "local_files_only": CLS_LOCAL_FILES_ONLY,
                        "freeze_encoder": CLS_FREEZE_ENCODER,
                        "classifier_hidden": CLS_CLASSIFIER_HIDDEN,
                        "classifier_dropout": CLS_CLASSIFIER_DROPOUT,
                        "use_cls_token": CLS_USE_CLS_TOKEN,
                    }
                )
            ckpt = {
                "model_state": cls_model.state_dict(),
                "config": ckpt_config,
                "score": float(best_score),
            }
            torch.save(ckpt, cls_best_path)
            print("[CLS] saved best ->", cls_best_path)
        if CLS_PATIENCE and best_epoch and (int(epoch) - int(best_epoch) >= int(CLS_PATIENCE)):
            print(f"[CLS] early stopping: sem melhora por {CLS_PATIENCE} épocas (best_epoch={best_epoch}).")
            break

    print("[CLS] done. best score:", best_score)
else:
    if DINO_ONLY:
        print("[CLS] DINO_ONLY=True (pulando treino do classificador).")
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
# Pesos pré-treinados: preferível (offline via cache); falha se faltar.
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
    },
    {
        "model_id": "deeplabv3p_tu_resnest101e",
        "arch": "deeplabv3plus",
        "encoder_name": "tu-resnest101e",
    },
    {
        "model_id": "segformer_mit_b3",
        "arch": "segformer",
        "encoder_name": "mit_b3",
    },
]

USE_DINOV2 = _env_bool("FORGERYSEG_USE_DINOV2", default=False)
if USE_DINOV2:
    SEG_TRAIN_SPECS.append(
        {
            "model_id": "dinov2_base_light",
            "backend": "dinov2",
            "arch": "dinov2",
            "hf_model_id": os.environ.get("FORGERYSEG_SEG_HF_MODEL_ID", "metaresearch/dinov2"),
            "freeze_encoder": _env_bool("FORGERYSEG_SEG_FREEZE_ENCODER", default=True),
            "decoder_channels": [256, 128, 64],
            "decoder_dropout": float(os.environ.get("FORGERYSEG_SEG_DECODER_DROPOUT", "0.0")),
            "local_files_only": OFFLINE_NO_DOWNLOAD,
        }
    )

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

if RUN_TRAIN_SEG and not DINO_ONLY:
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

    def build_seg_model(spec: dict, encoder_weights: str | None) -> nn.Module:
        backend = str(spec.get("backend", "smp")).lower()
        if backend == "smp":
            arch = str(spec.get("arch", "unet")).lower()
            encoder_name = str(spec.get("encoder_name", "efficientnet-b4"))
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

        if backend in {"dinov2", "hf"}:
            model_id = str(spec.get("hf_model_id", "metaresearch/dinov2"))
            return dinov2.build_dinov2_segmenter(
                model_id=model_id,
                decoder_channels=spec.get("decoder_channels", (256, 128, 64)),
                decoder_dropout=float(spec.get("decoder_dropout", 0.0)),
                pretrained=True,
                freeze_encoder=bool(spec.get("freeze_encoder", True)),
                local_files_only=bool(spec.get("local_files_only", OFFLINE_NO_DOWNLOAD)),
            )

        raise ValueError(f"SEG backend inválido: {backend!r}")

    available_encoders = set(builders.available_encoders())

    for spec in SEG_TRAIN_SPECS:
        model_id = str(spec["model_id"])
        backend = str(spec.get("backend", "smp")).lower()
        arch = str(spec.get("arch", "unetplusplus"))
        encoder_name = str(spec.get("encoder_name", spec.get("hf_model_id", "efficientnet-b4")))
        encoder_weights: str | None = spec.get("encoder_weights", SEG_ENCODER_WEIGHTS)

        if backend == "smp" and available_encoders and encoder_name not in available_encoders:
            raise ValueError(f"[SEG] encoder {encoder_name!r} não listado em SMP.")

        seg_model = build_seg_model(spec, encoder_weights).to(DEVICE)

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
                ckpt_config = {
                    "backend": backend,
                    "arch": arch,
                    "encoder_name": encoder_name,
                    "encoder_weights": encoder_weights,
                    "classes": 1,
                    "model_id": model_id,
                    "patch_size": int(SEG_PATCH_SIZE),
                    "fold": int(FOLD),
                    "seed": int(SEED),
                }
                if backend in {"dinov2", "hf"}:
                    ckpt_config.update(
                        {
                            "hf_model_id": spec.get("hf_model_id", encoder_name),
                            "freeze_encoder": bool(spec.get("freeze_encoder", True)),
                            "decoder_channels": spec.get("decoder_channels", (256, 128, 64)),
                            "decoder_dropout": float(spec.get("decoder_dropout", 0.0)),
                            "local_files_only": bool(spec.get("local_files_only", OFFLINE_NO_DOWNLOAD)),
                        }
                    )
                ckpt = {
                    "model_state": seg_model.state_dict(),
                    "config": ckpt_config,
                    "score": float(best_dice),
                }
                torch.save(ckpt, seg_best_path)
                print("[SEG] saved best ->", seg_best_path)
            if SEG_PATIENCE and best_epoch and (int(epoch) - int(best_epoch) >= int(SEG_PATIENCE)):
                print(f"[SEG {model_id}] early stopping: sem melhora por {SEG_PATIENCE} épocas (best_epoch={best_epoch}).")
                break

        print(f"[SEG {model_id}] done. best dice:", best_dice)
else:
    if DINO_ONLY:
        print("[SEG] DINO_ONLY=True (pulando treino SMP/SegFormer).")
    else:
        print("[SEG] RUN_TRAIN_SEG=False (pulando).")

# %% [markdown]
# ## Fase 3b — DINOv2 (offline) + head leve + TTA + pós-processamento
#
# Pipeline simples e robusto para Kaggle offline:
# - Encoder DINOv2 (congelado, pesos locais)
# - Head conv leve (3 camadas)
# - TTA + pós-processamento adaptativo
#
# Para ativar: `FORGERYSEG_DINO_ONLY=1` (default no Kaggle).

# %%
if DINO_ONLY:
    try:
        import cv2
    except Exception:
        print("[ERRO] OpenCV (cv2) não disponível. Instale ou inclua no bundle.")
        raise
    try:
        from PIL import Image
    except Exception:
        print("[ERRO] Pillow (PIL) não disponível. Inclua no bundle offline.")
        raise

    def _parse_version_tuple(ver: str) -> tuple[int, ...]:
        parts = []
        for chunk in str(ver).replace("+", ".").split("."):
            try:
                parts.append(int(chunk))
            except Exception:
                break
        return tuple(parts)

    def _ensure_hf_hub_compat() -> None:
        try:
            from importlib import metadata as importlib_metadata
        except Exception:
            import importlib_metadata  # type: ignore

        try:
            hub_ver = importlib_metadata.version("huggingface-hub")
        except Exception:
            hub_ver = None

        def _is_ok(v: str | None) -> bool:
            if not v:
                return False
            vt = _parse_version_tuple(v)
            return vt >= (0, 34, 0) and vt < (1, 0, 0)

        if _is_ok(hub_ver):
            return

        print(f"[DINO] huggingface-hub incompatível (versão atual={hub_ver}). Tentando instalar wheel offline (<1.0).")
        if OFFLINE_BUNDLE is None:
            raise RuntimeError("[DINO] bundle offline não encontrado; adicione wheels de huggingface-hub 0.34.x.")

        wheel_dir = OFFLINE_BUNDLE / "wheels"
        if not wheel_dir.exists():
            raise RuntimeError(f"[DINO] diretório de wheels não encontrado: {wheel_dir}")

        hub_wheels = sorted(wheel_dir.glob("huggingface_hub-*.whl"))
        if not hub_wheels:
            raise RuntimeError("[DINO] wheel de huggingface-hub não encontrada. Inclua huggingface-hub==0.34.* no bundle.")

        # escolher a wheel mais alta <1.0
        def _wheel_version(p: Path) -> tuple[int, ...]:
            name = p.name.replace("huggingface_hub-", "")
            ver = name.split("-")[0]
            return _parse_version_tuple(ver)

        candidates = [(p, _wheel_version(p)) for p in hub_wheels]
        candidates = [c for c in candidates if c[1] >= (0, 34, 0) and c[1] < (1, 0, 0)]
        if not candidates:
            raise RuntimeError("[DINO] nenhuma wheel compatível de huggingface-hub (<1.0) encontrada no bundle.")

        candidates.sort(key=lambda x: x[1], reverse=True)
        wheel_path = candidates[0][0]
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheel_dir),
            "--no-deps",
            str(wheel_path),
        ]
        print("[DINO] instalando:", " ".join(cmd))
        subprocess.check_call(cmd)

        try:
            hub_ver = importlib_metadata.version("huggingface-hub")
        except Exception:
            hub_ver = None
        if not _is_ok(hub_ver):
            raise RuntimeError(f"[DINO] huggingface-hub ainda incompatível após instalação (versão={hub_ver}).")
        print("[DINO] huggingface-hub OK:", hub_ver)

    _ensure_hf_hub_compat()

    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception:
        print("[ERRO] transformers não disponível. Inclua no bundle offline.")
        raise

    DINO_PATH = os.environ.get("FORGERYSEG_DINO_PATH", "/kaggle/input/dinov2/pytorch/base/1")
    DINO_IMAGE_SIZE = int(os.environ.get("FORGERYSEG_DINO_IMAGE_SIZE", "512"))
    DINO_BATCH_SIZE = int(os.environ.get("FORGERYSEG_DINO_BATCH_SIZE", "4"))
    DINO_EPOCHS = int(os.environ.get("FORGERYSEG_DINO_EPOCHS", "5"))
    DINO_LR = float(os.environ.get("FORGERYSEG_DINO_LR", "3e-4"))
    DINO_WEIGHT_DECAY = float(os.environ.get("FORGERYSEG_DINO_WEIGHT_DECAY", "1e-2"))
    DINO_LOCAL_FILES_ONLY = _env_bool("FORGERYSEG_DINO_LOCAL_ONLY", default=True)
    DINO_DECODER_DROPOUT = float(os.environ.get("FORGERYSEG_DINO_DECODER_DROPOUT", "0.0"))
    DINO_SAVE_DIR = (Path("/kaggle/working") if is_kaggle() else Path(".").resolve()) / "outputs" / "models_dino"
    DINO_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DINO_CKPT_PATH = DINO_SAVE_DIR / "best.pt"

    # Pós-processamento (adaptativo)
    DINO_THR_FACTOR = float(os.environ.get("FORGERYSEG_DINO_THR_FACTOR", "0.3"))
    DINO_MIN_AREA = int(os.environ.get("FORGERYSEG_DINO_MIN_AREA", "30"))
    DINO_MIN_AREA_PERCENT = float(os.environ.get("FORGERYSEG_DINO_MIN_AREA_PERCENT", "0.0005"))
    DINO_MIN_CONFIDENCE = float(os.environ.get("FORGERYSEG_DINO_MIN_CONFIDENCE", "0.33"))
    DINO_MORPH_CLOSE_K = int(os.environ.get("FORGERYSEG_DINO_CLOSE_K", "5"))
    DINO_MORPH_OPEN_K = int(os.environ.get("FORGERYSEG_DINO_OPEN_K", "3"))
    DINO_MORPH_ITERS = int(os.environ.get("FORGERYSEG_DINO_MORPH_ITERS", "1"))
    DINO_USE_TTA = _env_bool("FORGERYSEG_DINO_TTA", default=True)

    print("DINO_PATH:", DINO_PATH)
    print("DINO_IMAGE_SIZE:", DINO_IMAGE_SIZE)
    print("DINO_BATCH_SIZE:", DINO_BATCH_SIZE)
    print("DINO_EPOCHS:", DINO_EPOCHS)
    print("DINO_THR_FACTOR:", DINO_THR_FACTOR)
    print("DINO_MIN_AREA:", DINO_MIN_AREA, "DINO_MIN_AREA_PERCENT:", DINO_MIN_AREA_PERCENT)
    print("DINO_MIN_CONFIDENCE:", DINO_MIN_CONFIDENCE)
    print("DINO_USE_TTA:", DINO_USE_TTA)

    if str(DINO_PATH).startswith("/") and not Path(DINO_PATH).exists():
        raise FileNotFoundError(f"[DINO] caminho não encontrado: {DINO_PATH}")

    class DinoSeg(nn.Module):
        def __init__(self, dino_path: str, out_ch: int = 1):
            super().__init__()
            self.processor = AutoImageProcessor.from_pretrained(
                dino_path,
                local_files_only=DINO_LOCAL_FILES_ONLY,
            )
            self.encoder = AutoModel.from_pretrained(
                dino_path,
                local_files_only=DINO_LOCAL_FILES_ONLY,
            )
            for p in self.encoder.parameters():
                p.requires_grad = False

            hidden_size = int(getattr(self.encoder.config, "hidden_size", 768))
            self.head = nn.Sequential(
                nn.Conv2d(hidden_size, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=float(DINO_DECODER_DROPOUT)),
                nn.Conv2d(256, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 1),
            )

        @torch.no_grad()
        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, C, H, W] float32 em [0,1]
            imgs = (x * 255.0).clamp(0, 255).to(torch.uint8)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            inputs = self.processor(
                images=list(imgs),
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            ).to(x.device)
            feats = self.encoder(**inputs).last_hidden_state  # B, N, C
            feats = feats[:, 1:, :]
            b, n, c = feats.shape
            s = int(np.sqrt(n))
            fmap = feats.permute(0, 2, 1).reshape(b, c, s, s)
            return fmap

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            fmap = self.forward_features(x)
            logits = self.head(
                torch.nn.functional.interpolate(
                    fmap,
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            return logits

    def _load_union_mask(mask_path: Path | None, out_size: int) -> np.ndarray:
        if mask_path is None:
            return np.zeros((out_size, out_size), dtype=np.uint8)
        masks = np.load(mask_path)
        if masks.ndim == 2:
            union = masks
        else:
            union = masks.max(axis=0)
        union = (union > 0).astype(np.uint8)
        if union.shape != (out_size, out_size):
            union = cv2.resize(union, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        return union

    class DinoSegDataset(Dataset):
        def __init__(self, samples, image_size: int, train: bool):
            self.samples = samples
            self.image_size = int(image_size)
            self.train = bool(train)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            s = self.samples[int(idx)]
            img = np.array(Image.open(s.image_path).convert("RGB"))
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            mask = _load_union_mask(s.mask_path, self.image_size)

            if self.train:
                if np.random.rand() < 0.5:
                    img = np.ascontiguousarray(img[:, ::-1])
                    mask = np.ascontiguousarray(mask[:, ::-1])
                if np.random.rand() < 0.5:
                    img = np.ascontiguousarray(img[::-1, :])
                    mask = np.ascontiguousarray(mask[::-1, :])
                # rotações 90°
                if np.random.rand() < 0.25:
                    img = np.ascontiguousarray(np.rot90(img, k=1))
                    mask = np.ascontiguousarray(np.rot90(mask, k=1))

            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            y = torch.from_numpy(mask).unsqueeze(0).float()
            return x, y

    def _dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
        probs = torch.sigmoid(logits)
        preds = (probs > thr).float()
        inter = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * inter + eps) / (union + eps)
        return float(dice.mean().item())

    def _adaptive_threshold_value(prob: np.ndarray, factor: float = 0.3, eps: float = 1e-6) -> float:
        mean = float(np.mean(prob))
        std = float(np.std(prob))
        thr = mean + float(factor) * std
        thr = max(float(eps), min(1.0 - float(eps), thr))
        return float(thr)

    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        if int(min_area) <= 0:
            return mask
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num <= 1:
            return mask
        out = np.zeros_like(mask, dtype=np.uint8)
        for idx in range(1, num):
            area = stats[idx, cv2.CC_STAT_AREA]
            if int(area) >= int(min_area):
                out[labels == idx] = 1
        return out

    def _postprocess_prob(prob: np.ndarray) -> np.ndarray:
        thr = _adaptive_threshold_value(prob, factor=DINO_THR_FACTOR)
        mask = (prob >= thr).astype(np.uint8)

        if DINO_MORPH_CLOSE_K > 0:
            kernel = np.ones((DINO_MORPH_CLOSE_K, DINO_MORPH_CLOSE_K), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(DINO_MORPH_ITERS))
        if DINO_MORPH_OPEN_K > 0:
            kernel = np.ones((DINO_MORPH_OPEN_K, DINO_MORPH_OPEN_K), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(DINO_MORPH_ITERS))

        min_area = int(DINO_MIN_AREA)
        if float(DINO_MIN_AREA_PERCENT) > 0:
            min_area = max(min_area, int(float(DINO_MIN_AREA_PERCENT) * float(mask.size)))
        mask = _remove_small_components(mask, min_area=min_area)

        if int(mask.sum()) <= 0:
            return mask

        if float(DINO_MIN_CONFIDENCE) > 0:
            conf = float(prob[mask > 0].mean())
            if conf < float(DINO_MIN_CONFIDENCE):
                return np.zeros_like(mask, dtype=np.uint8)
        return mask

    dino_model: DinoSeg | None = None
    if RUN_TRAIN_DINO:
        ds_dino_train = DinoSegDataset([train_samples[i] for i in train_idx.tolist()], DINO_IMAGE_SIZE, train=True)
        ds_dino_val = DinoSegDataset([train_samples[i] for i in val_idx.tolist()], DINO_IMAGE_SIZE, train=False)
        dl_dino_train = DataLoader(ds_dino_train, batch_size=DINO_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), drop_last=True)
        dl_dino_val = DataLoader(ds_dino_val, batch_size=DINO_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), drop_last=False)

        dino_model = DinoSeg(DINO_PATH).to(DEVICE)
        dino_optimizer = torch.optim.AdamW(dino_model.head.parameters(), lr=DINO_LR, weight_decay=DINO_WEIGHT_DECAY)
        dino_loss = nn.BCEWithLogitsLoss()
        dino_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

        best_dice = -1.0
        best_epoch = 0
        for epoch in range(1, int(DINO_EPOCHS) + 1):
            dino_model.train()
            tr_losses = []
            for xb, yb in tqdm(dl_dino_train, desc="dino train", leave=False):
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                dino_optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    logits = dino_model(xb)
                    loss = dino_loss(logits, yb)
                dino_scaler.scale(loss).backward()
                dino_scaler.step(dino_optimizer)
                dino_scaler.update()
                tr_losses.append(float(loss.item()))

            dino_model.eval()
            val_dices = []
            with torch.no_grad():
                for xb, yb in tqdm(dl_dino_val, desc="dino val", leave=False):
                    xb = xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)
                    logits = dino_model(xb)
                    val_dices.append(_dice_from_logits(logits, yb))
            mean_dice = float(np.mean(val_dices)) if val_dices else float("nan")
            print(f"[DINO] epoch {epoch:02d}/{DINO_EPOCHS} | train_loss={np.mean(tr_losses):.4f} | dice@0.5={mean_dice:.4f}")
            if float(mean_dice) > best_dice:
                best_dice = float(mean_dice)
                best_epoch = int(epoch)
                ckpt = {
                    "head_state": dino_model.head.state_dict(),
                    "config": {
                        "dino_path": str(DINO_PATH),
                        "image_size": int(DINO_IMAGE_SIZE),
                        "thr_factor": float(DINO_THR_FACTOR),
                        "min_area": int(DINO_MIN_AREA),
                        "min_area_percent": float(DINO_MIN_AREA_PERCENT),
                        "min_confidence": float(DINO_MIN_CONFIDENCE),
                    },
                    "score": float(best_dice),
                }
                torch.save(ckpt, DINO_CKPT_PATH)
                print("[DINO] saved best ->", DINO_CKPT_PATH)
            if best_epoch and (int(epoch) - int(best_epoch) >= 3):
                print("[DINO] early stopping: sem melhora por 3 épocas.")
                break

        print("[DINO] done. best dice:", best_dice)
    else:
        print("[DINO] RUN_TRAIN_DINO=False (pulando treino).")

    if dino_model is None:
        if not DINO_CKPT_PATH.exists():
            raise FileNotFoundError(f"[DINO] checkpoint não encontrado: {DINO_CKPT_PATH}")
        dino_model = DinoSeg(DINO_PATH).to(DEVICE)
        ckpt = torch.load(DINO_CKPT_PATH, map_location=DEVICE)
        dino_model.head.load_state_dict(ckpt["head_state"])
        dino_model.eval()
        print("[DINO] carregado checkpoint ->", DINO_CKPT_PATH)

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

    USE_TTA = _env_bool("FORGERYSEG_TTA", default=True)
    TTA_MODES = ("none", "hflip", "vflip", "rot90", "rot180", "rot270")

    def _apply_tta(image: np.ndarray, mode: str) -> np.ndarray:
        if mode == "none":
            return image
        if mode == "hflip":
            return np.ascontiguousarray(image[:, ::-1])
        if mode == "vflip":
            return np.ascontiguousarray(image[::-1, :])
        if mode == "rot90":
            return np.ascontiguousarray(np.rot90(image, k=1))
        if mode == "rot180":
            return np.ascontiguousarray(np.rot90(image, k=2))
        if mode == "rot270":
            return np.ascontiguousarray(np.rot90(image, k=3))
        raise ValueError(f"TTA mode inválido: {mode}")

    def _undo_tta(mask: np.ndarray, mode: str) -> np.ndarray:
        if mode == "none":
            return mask
        if mode == "hflip":
            return np.ascontiguousarray(mask[:, ::-1])
        if mode == "vflip":
            return np.ascontiguousarray(mask[::-1, :])
        if mode == "rot90":
            return np.ascontiguousarray(np.rot90(mask, k=3))
        if mode == "rot180":
            return np.ascontiguousarray(np.rot90(mask, k=2))
        if mode == "rot270":
            return np.ascontiguousarray(np.rot90(mask, k=1))
        raise ValueError(f"TTA mode inválido: {mode}")

    def _default_predict_fn(image: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict"):
            pred = model.predict(image[None])[0]
        else:
            from forgeryseg.inference import predict_image

            pred = predict_image(model, image, DEVICE)
        pred = np.asarray(pred)
        if pred.ndim > 2:
            pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise ValueError(f"Predição esperada HxW, obtido shape={pred.shape}")
        return pred.astype(np.float32)

    def _tta_predict(image: np.ndarray) -> np.ndarray:
        if not USE_TTA:
            return _default_predict_fn(image)
        preds = []
        for mode in TTA_MODES:
            img_t = _apply_tta(image, mode)
            pred_t = _default_predict_fn(img_t)
            pred = _undo_tta(pred_t, mode)
            preds.append(pred)
        return np.mean(preds, axis=0)

    annotations = []
    for case_id in case_ids:
        key = str(case_id)
        if key not in test_by_id:
            raise FileNotFoundError(f"Não encontrei imagem para case_id={case_id!r}")
        img_path = test_by_id[key]
        # carregue e processe a imagem
        img = np.array(Image.open(img_path)) / 255.0
        # modelo deve gerar um mapa de probabilidade ou máscara
        pred_prob = _tta_predict(img)
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
# Fase 4 — Célula 11b: Submissão DINO-only (offline)
submissions_from_dino = None
if DINO_ONLY:
    if dino_model is None:
        raise RuntimeError("[DINO] modelo não carregado.")

    dino_model.eval()

    TTA_MODES = ("none", "hflip", "vflip", "rot90", "rot180", "rot270")

    def _apply_tta(img: np.ndarray, mode: str) -> np.ndarray:
        if mode == "none":
            return img
        if mode == "hflip":
            return np.ascontiguousarray(img[:, ::-1])
        if mode == "vflip":
            return np.ascontiguousarray(img[::-1, :])
        if mode == "rot90":
            return np.ascontiguousarray(np.rot90(img, k=1))
        if mode == "rot180":
            return np.ascontiguousarray(np.rot90(img, k=2))
        if mode == "rot270":
            return np.ascontiguousarray(np.rot90(img, k=3))
        raise ValueError(f"TTA mode inválido: {mode}")

    def _undo_tta(mask: np.ndarray, mode: str) -> np.ndarray:
        if mode == "none":
            return mask
        if mode == "hflip":
            return np.ascontiguousarray(mask[:, ::-1])
        if mode == "vflip":
            return np.ascontiguousarray(mask[::-1, :])
        if mode == "rot90":
            return np.ascontiguousarray(np.rot90(mask, k=3))
        if mode == "rot180":
            return np.ascontiguousarray(np.rot90(mask, k=2))
        if mode == "rot270":
            return np.ascontiguousarray(np.rot90(mask, k=1))
        raise ValueError(f"TTA mode inválido: {mode}")

    @torch.no_grad()
    def _dino_predict_prob(img_rgb: np.ndarray) -> np.ndarray:
        orig_h, orig_w = img_rgb.shape[:2]
        img_rs = cv2.resize(img_rgb, (DINO_IMAGE_SIZE, DINO_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img_rs).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(DEVICE)
        logits = dino_model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        if prob.shape != (orig_h, orig_w):
            prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return prob

    def _dino_tta_predict(img_rgb: np.ndarray) -> np.ndarray:
        if not DINO_USE_TTA:
            return _dino_predict_prob(img_rgb)
        preds = []
        for mode in TTA_MODES:
            img_t = _apply_tta(img_rgb, mode)
            prob_t = _dino_predict_prob(img_t)
            prob = _undo_tta(prob_t, mode)
            preds.append(prob)
        return np.mean(preds, axis=0).astype(np.float32)

    annotations = []
    for case_id in case_ids:
        key = str(case_id)
        if key not in test_by_id:
            raise FileNotFoundError(f"[DINO] imagem não encontrada para case_id={case_id!r}")
        img_path = test_by_id[key]
        img = np.array(Image.open(img_path).convert("RGB"))
        prob = _dino_tta_predict(img)
        mask = _postprocess_prob(prob)
        annotations.append(encode_submission(mask))

    submissions_from_dino = pd.DataFrame(
        {
            "case_id": case_ids,
            "annotation": annotations,
        }
    )

    submissions_from_dino.head()

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
    if DINO_ONLY and submissions_from_dino is not None:
        submissions_to_save = submissions_from_dino
    else:
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
RUN_SUBMISSION_SCRIPT = _env_bool("FORGERYSEG_RUN_SUBMISSION_SCRIPT", default=bool(is_kaggle() and not DINO_ONLY))
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
        if any(cand.glob("*/best.pt")):
            return cand
        if any(cand.glob("**/best.pt")):
            return cand
        if any(cand.glob("*/*/last.pt")):
            return cand
        if any(cand.glob("*/last.pt")):
            return cand
        if any(cand.glob("**/last.pt")):
            return cand
    return None


if RUN_SUBMISSION_SCRIPT:
    submit_script = _find_submit_ensemble_script()
    infer_cfg_path = _find_infer_cfg_path()

    submission_path = output_root() / "submission.csv"
    models_dir = _find_models_dir_with_ckpt()
    if models_dir is None:
        raise RuntimeError("[SUBMISSION] nenhum checkpoint encontrado em outputs/models_seg.")

    cmd = [
        sys.executable,
        str(submit_script),
        "--data-root",
        str(DATA_ROOT),
        "--out-csv",
        str(submission_path),
    ]
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
    print("[SUBMISSION] RUN_SUBMISSION_SCRIPT=False (pulando).")
