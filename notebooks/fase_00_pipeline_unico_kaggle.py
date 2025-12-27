# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---
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
# - Importa código do projeto em `src/forgeryseg/` (modularizado).
# - Compatível com Kaggle **internet OFF** (instala wheels locais se existirem).
# - Não esconde erros: exceções e tracebacks aparecem.
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
# Fase 1 — Célula 2b: Instalação offline (wheels) — NÃO resolve deps
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
from forgeryseg.checkpoints import build_classifier_from_config, build_segmentation_from_config, load_checkpoint  # noqa: E402
from forgeryseg.constants import AUTHENTIC_LABEL  # noqa: E402
from forgeryseg.data_analysis import quick_dataset_stats  # noqa: E402
from forgeryseg.dataset import PatchDataset, build_test_index, build_train_index, load_image  # noqa: E402
from forgeryseg.inference import normalize_image, predict_image  # noqa: E402
from forgeryseg.losses import BCETverskyLoss  # noqa: E402
from forgeryseg.models import builders  # noqa: E402
from forgeryseg.models.classifier import build_classifier, compute_pos_weight  # noqa: E402
from forgeryseg.offline import configure_cache_dirs  # noqa: E402
from forgeryseg.postprocess import binarize, extract_components  # noqa: E402
from forgeryseg.rle import encode_instances  # noqa: E402
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
test_samples = build_test_index(DATA_ROOT)

print("DATA_ROOT:", DATA_ROOT)
print("train samples:", len(train_samples), "auth:", int((train_labels == 0).sum()), "forged:", int((train_labels == 1).sum()))
print("test samples:", len(test_samples))

# %% [markdown]
# ## Análise dos Dados e Pré-processamento
#
# Antes de treinar modelos, fazemos uma exploração rápida do dataset:
#
# - Cada imagem tem um identificador único (`case_id` = nome do arquivo).
# - No treino, existem dois grupos: **authentic** e **forged**.
# - Para imagens **forged**, há uma máscara em `train_masks/<case_id>.npy` indicando pixels duplicados
#   (pode vir como máscara binária 2D ou como múltiplas instâncias 3D).
# - Para segmentação, montamos pares (**imagem**, **máscara-union**). Para imagens autênticas, a máscara é toda zero.
# - Para classificação, usamos rótulo binário: `0=authentic`, `1=forged`.
#
# ### Dimensionamento e formato
#
# - As imagens têm tamanhos variados e podem ser coloridas ou em escala de cinza. Padronizamos a leitura para **3 canais RGB**
#   (via `PIL.Image.convert("RGB")`, usado em `forgeryseg.dataset.load_image`).
# - Para segmentação, em vez de redimensionar agressivamente a imagem inteira, trabalhamos com **patches de tamanho fixo**
#   (`SEG_PATCH_SIZE`, ex.: 384/512) usando **padding + crop** durante o treino. Isso preserva detalhes finos de falsificações pequenas.
# - Em inferência, usamos **tiling** (`TILE_SIZE`/`OVERLAP`) para lidar com imagens grandes sem perder resolução.
#
# ### Normalização
#
# - Aplicamos **normalização padronizada por canal** (médias/desvios do ImageNet) para manter a escala esperada pelos modelos
#   — especialmente útil com backbones pré-treinados.
# - Existe variação de contraste entre artigos/figuras; equalização/normalização por imagem pode ajudar, mas pode também
#   alterar evidências sutis. Preferimos manter o pré-processamento **mínimo** e ganhar robustez com augmentations.
#
# ### Divisão de dados (validação local)
#
# - A competição avalia no **test set oculto**; para desenvolvimento usamos validação local a partir do treino.
# - Usamos **5-fold estratificado** (authentic vs forged) para melhor uso dos dados e para reduzir overfitting.
# - Na inferência, carregamos todos os checkpoints encontrados e fazemos **ensemble** (média das probabilidades).
#
# ### Data augmentation (aumento de dados)
#
# `forgeryseg.augment.get_train_augment(...)` aplica (imagem + máscara) transforms geométricos e (só na imagem) transforms
# fotométricos para robustez:
#
# - Flips/rotações (`HorizontalFlip`, `VerticalFlip`, `RandomRotate90`)
# - Affine/escala (`Affine`, `RandomResizedCrop` quando `patch_size` é definido)
# - Ruído/desfoque (`GaussNoise`, `GaussianBlur`, `MotionBlur`)
# - Brilho/contraste/CLAHE (`RandomBrightnessContrast`, `RandomGamma`, `CLAHE`)
# - Copy-move sintético (custom `CopyMoveTransform`) para gerar falsificações on-the-fly em máscaras vazias
#
# ### Modelo de classificação (forged vs authentic)
#
# Classificador opcional para (1) sinal global “tem fraude?” e (2) economizar tempo:
#
# - CNN pré-treinada (ex.: EfficientNet via `timm`) com saída binária (`num_classes=1`).
# - Loss: `BCEWithLogitsLoss` com `pos_weight`.
# - Inferência: se `p_forged < CLS_SKIP_THRESHOLD`, rotulamos como `AUTHENTIC_LABEL` e pulamos a segmentação.
#
# ### Modelo de segmentação (localização)
#
# Segmentação prevê logits (1 canal); aplicamos **sigmóide** para probabilidades. O pipeline suporta ensemble heterogêneo:
#
# - **U-Net++** (detalhe fino) + **DeepLabV3+** (contexto multi-escala) + opcional **SegFormer** (atenção global).
#
# ---

# %%
# Fase 1 — Célula 3b: Stats rápidos (tamanho / canais / máscaras)
RUN_DATA_ANALYSIS = not is_kaggle()
ANALYSIS_MAX_ITEMS = 200

if RUN_DATA_ANALYSIS:
    quick_dataset_stats(train_samples, max_items=ANALYSIS_MAX_ITEMS, seed=SEED, name="train")
    quick_dataset_stats(test_samples, max_items=ANALYSIS_MAX_ITEMS, seed=SEED, name="test")
else:
    print("[DATA] RUN_DATA_ANALYSIS=False (pulando).")

# %%
# Fase 1 — Célula 4: Config global (liga/desliga)
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


# Modos:
# - "infer_only": só inferência (exige checkpoints anexados)
# - "train_then_infer": força treino antes da submissão
# - "auto": se não houver checkpoints, treina o mínimo para conseguir gerar `submission.csv`
_default_pipeline_mode = "train_then_infer" if is_kaggle() else "auto"
PIPELINE_MODE = os.environ.get("FORGERYSEG_PIPELINE_MODE", _default_pipeline_mode).strip().lower()

# Defaults "rodáveis": por padrão, no Kaggle tentamos gerar `submission.csv` sem intervenção manual.
# (Você pode sobrescrever manualmente aqui.)
RUN_TRAIN_CLS = False
if PIPELINE_MODE == "infer_only":
    RUN_TRAIN_SEG = False
elif PIPELINE_MODE == "train_then_infer":
    RUN_TRAIN_SEG = True
else:
    RUN_TRAIN_SEG = not HAS_SEG_CKPT
RUN_SUBMISSION = True

N_FOLDS = 5
FOLD = 0

FAST_TRAIN = _env_bool("FORGERYSEG_FAST_TRAIN", default=bool(is_kaggle() and not HAS_SEG_CKPT))

print("PIPELINE_MODE:", PIPELINE_MODE)
print("FAST_TRAIN:", FAST_TRAIN)
print("RUN_TRAIN_CLS:", RUN_TRAIN_CLS)
print("RUN_TRAIN_SEG:", RUN_TRAIN_SEG)
print("RUN_SUBMISSION:", RUN_SUBMISSION)
print("HAS_SEG_CKPT:", HAS_SEG_CKPT)
print("HAS_CLS_CKPT:", HAS_CLS_CKPT)

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
CLS_EPOCHS = 10
CLS_PATIENCE = 3
CLS_LR = 3e-4
CLS_WEIGHT_DECAY = 1e-2
# Preferir recall (evitar falsos negativos): só pule a segmentação quando tiver alta confiança de autenticidade.
CLS_SKIP_THRESHOLD = 0.10
# Pesos pré-treinados ajudam, mas no Kaggle com internet OFF podem não estar disponíveis.
CLS_PRETRAINED = True


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
    ds_cls_train = ClsDataset([train_samples[i] for i in train_idx.tolist()], build_transform(train=True))
    ds_cls_val = ClsDataset([train_samples[i] for i in val_idx.tolist()], build_transform(train=False))

    num_workers = 2 if is_kaggle() else 0
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
else:
    print("[CLS] RUN_TRAIN_CLS=False (pulando).")

# %%
# Fase 3 — Célula 7: Treino de segmentação (opcional)
SEG_PATCH_SIZE = 384
SEG_COPY_MOVE_PROB = 0.20
SEG_BATCH_SIZE = 8
SEG_EPOCHS = 8
SEG_LR = 1e-3
SEG_WEIGHT_DECAY = 1e-2
SEG_PATIENCE = 3
# Pesos pré-treinados: preferível (offline via cache); fallback para None se falhar.
SEG_ENCODER_WEIGHTS = "imagenet"

# Para performance máxima, treine mais de uma arquitetura e faça ensemble na inferência.
SEG_TRAIN_SPECS = [
    {"model_id": "unetpp_tu_convnext_small", "arch": "unetplusplus", "encoder_name": "tu-convnext_small"},
    {"model_id": "deeplabv3p_tu_resnest101e", "arch": "deeplabv3plus", "encoder_name": "tu-resnest101e"},
    {"model_id": "segformer_mit_b2", "arch": "segformer", "encoder_name": "mit_b2"},
]

if FAST_TRAIN:
    print("[SEG] FAST_TRAIN=True -> preset rápido (1 modelo / poucas épocas).")
    SEG_EPOCHS = min(int(SEG_EPOCHS), 2)
    SEG_TRAIN_SPECS = [
        {"model_id": "unetpp_tu_convnext_small", "arch": "unetplusplus", "encoder_name": "tu-convnext_small"},
    ]

if RUN_TRAIN_SEG:
    train_aug = get_train_augment(patch_size=SEG_PATCH_SIZE, copy_move_prob=SEG_COPY_MOVE_PROB)
    val_aug = get_val_augment()

    ds_seg_train = PatchDataset([train_samples[i] for i in train_idx.tolist()], patch_size=SEG_PATCH_SIZE, train=True, augment=train_aug, positive_prob=0.7, seed=SEED)
    ds_seg_val = PatchDataset([train_samples[i] for i in val_idx.tolist()], patch_size=SEG_PATCH_SIZE, train=False, augment=val_aug, seed=SEED)

    num_workers = 2 if is_kaggle() else 0
    dl_seg_train = DataLoader(ds_seg_train, batch_size=SEG_BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE == "cuda"), drop_last=True)
    dl_seg_val = DataLoader(ds_seg_val, batch_size=SEG_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE == "cuda"), drop_last=False)

    def output_root() -> Path:
        return Path("/kaggle/working") if is_kaggle() else Path(".").resolve()

    use_amp = (DEVICE == "cuda")

    def build_seg_model(arch: str, encoder_name: str, encoder_weights: str | None) -> nn.Module:
        arch = str(arch).lower()
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

    for spec in SEG_TRAIN_SPECS:
        model_id = str(spec["model_id"])
        arch = str(spec.get("arch", "unetplusplus"))
        encoder_name = str(spec.get("encoder_name", "efficientnet-b4"))
        encoder_weights: str | None = spec.get("encoder_weights", SEG_ENCODER_WEIGHTS)

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

        seg_save_dir = output_root() / "outputs" / "models_seg" / model_id / f"fold_{int(FOLD)}"
        seg_save_dir.mkdir(parents=True, exist_ok=True)
        seg_best_path = seg_save_dir / "best.pt"

        best_dice = -1.0
        best_epoch = 0
        for epoch in range(1, int(SEG_EPOCHS) + 1):
            tr = train_one_epoch(seg_model, dl_seg_train, seg_criterion, seg_optimizer, DEVICE, use_amp=use_amp, progress=True, desc=f"seg train ({model_id})")
            val_stats, val_dice = validate(seg_model, dl_seg_val, seg_criterion, DEVICE, progress=True, desc=f"seg val ({model_id})")
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
else:
    print("[SEG] RUN_TRAIN_SEG=False (pulando).")

# %%
# Fase 4 — Célula 8: Carregar checkpoints (para inferência/submissão)


def output_root() -> Path:
    return Path("/kaggle/working") if is_kaggle() else Path(".").resolve()


def _find_models_dir(dir_name: str) -> Path | None:
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

SEG_MODELS: list[nn.Module] = []
if MODELS_SEG_DIR is not None:
    for ckpt_path in sorted(MODELS_SEG_DIR.glob("*/*/best.pt")):
        try:
            state, cfg = load_checkpoint(ckpt_path)
            m = build_segmentation_from_config(cfg)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            SEG_MODELS.append(m)
        except Exception:
            print("[ERRO] falha ao carregar seg:", ckpt_path)
            traceback.print_exc()

CLS_MODELS: list[nn.Module] = []
CLS_INFER_IMAGE_SIZE = CLS_IMAGE_SIZE
if MODELS_CLS_DIR is not None:
    for ckpt_path in sorted(MODELS_CLS_DIR.glob("fold_*/best.pt")):
        try:
            state, cfg = load_checkpoint(ckpt_path)
            m, image_size = build_classifier_from_config(cfg)
            CLS_INFER_IMAGE_SIZE = int(image_size)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            CLS_MODELS.append(m)
        except Exception:
            print("[ERRO] falha ao carregar cls:", ckpt_path)
            traceback.print_exc()

print("loaded seg models:", len(SEG_MODELS))
print("loaded cls models:", len(CLS_MODELS))

# %%
# Fase 4 — Célula 9: Inferência + submission
def _find_infer_cfg_path() -> Path | None:
    candidates: list[Path] = []
    if PROJECT_ROOT is not None:
        candidates.append(PROJECT_ROOT / "configs" / "infer_ensemble.json")
    candidates.append(Path("configs/infer_ensemble.json").resolve())

    for p in candidates:
        if p.exists():
            return p

    if is_kaggle():
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for ds in sorted(kaggle_input.glob("*")):
                for base in (ds, ds / "recodai_bundle"):
                    p = base / "configs" / "infer_ensemble.json"
                    if p.exists():
                        return p
    return None


INFER_CFG_PATH = _find_infer_cfg_path()
INFER_CFG: dict = {}
if INFER_CFG_PATH is not None:
    with INFER_CFG_PATH.open("r") as f:
        INFER_CFG = json.load(f)
    print("[INFER CFG] loaded:", INFER_CFG_PATH)
else:
    print("[INFER CFG] configs/infer_ensemble.json não encontrado; usando defaults do notebook.")

TILE_SIZE = int(INFER_CFG.get("tile_size", 1024))
OVERLAP = int(INFER_CFG.get("overlap", 128))
MAX_SIZE = int(INFER_CFG.get("max_size", 0))
THRESHOLD = float(INFER_CFG.get("threshold", 0.50))
MIN_AREA = int(INFER_CFG.get("min_area", 32))

if isinstance(INFER_CFG.get("tta_modes"), (list, tuple)):
    TTA_MODES = tuple(str(x) for x in INFER_CFG.get("tta_modes") if str(x).strip())
else:
    TTA_MODES = ("none", "hflip")

USE_TTA = bool(TTA_MODES)
if FAST_TRAIN:
    USE_TTA = False

CLS_SKIP_THRESHOLD = float(INFER_CFG.get("cls_skip_threshold", CLS_SKIP_THRESHOLD))


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


@torch.no_grad()
def predict_prob_forged(image: np.ndarray) -> float:
    import torch.nn.functional as F

    img = normalize_image(image)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    if CLS_INFER_IMAGE_SIZE and x.shape[-2:] != (CLS_INFER_IMAGE_SIZE, CLS_INFER_IMAGE_SIZE):
        x = F.interpolate(x, size=(CLS_INFER_IMAGE_SIZE, CLS_INFER_IMAGE_SIZE), mode="bilinear", align_corners=False)
    probs: list[float] = []
    for m in CLS_MODELS:
        logits = m(x).view(-1)
        probs.append(float(torch.sigmoid(logits)[0].item()))
    return float(np.mean(probs)) if probs else 0.0


def predict_seg_ensemble_prob(image: np.ndarray) -> np.ndarray:
    if not SEG_MODELS:
        msg = (
            "Nenhum modelo de segmentação carregado.\n"
            f"- MODELS_SEG_DIR={MODELS_SEG_DIR}\n"
            "- Esperado: `outputs/models_seg/<model_id>/fold_*/best.pt` (no Kaggle: /kaggle/working/outputs/...)\n"
            "- Soluções:\n"
            "  1) Rode treino aqui: defina `RUN_TRAIN_SEG=True` e execute as células de treino.\n"
            "  2) Anexe um Dataset com checkpoints em `outputs/models_seg/...` e rode novamente.\n"
        )
        raise RuntimeError(msg)
    probs_sum: np.ndarray | None = None
    count = 0
    modes = TTA_MODES if USE_TTA else ("none",)
    for mode in modes:
        img_t = _apply_tta(image, mode)
        ens: np.ndarray | None = None
        for m in SEG_MODELS:
            p = predict_image(m, img_t, DEVICE, tile_size=TILE_SIZE, overlap=OVERLAP, max_size=MAX_SIZE)
            ens = p if ens is None else (ens + p)
        assert ens is not None
        ens = ens / float(len(SEG_MODELS))
        ens = _undo_tta(ens, mode)
        probs_sum = ens if probs_sum is None else (probs_sum + ens)
        count += 1
    assert probs_sum is not None
    return probs_sum / float(max(count, 1))


def predict_instances(image: np.ndarray) -> list[np.ndarray]:
    prob = predict_seg_ensemble_prob(image)
    bin_mask = binarize(prob, threshold=THRESHOLD)
    return extract_components(bin_mask, min_area=int(MIN_AREA))


if RUN_SUBMISSION:
    if not SEG_MODELS:
        raise RuntimeError(
            "RUN_SUBMISSION=True, mas nenhum modelo de segmentação foi carregado.\n"
            f"- MODELS_SEG_DIR={MODELS_SEG_DIR}\n"
            "Treine (RUN_TRAIN_SEG=True) ou anexe um Dataset com `outputs/models_seg/<model_id>/fold_*/best.pt`."
        )
    SUBMISSION_PATH = Path("/kaggle/working/submission.csv") if is_kaggle() else (output_root() / "submission.csv")
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SUBMISSION_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
        writer.writeheader()

        for s in tqdm(test_samples, desc="infer"):
            img = load_image(s.image_path)
            if CLS_MODELS:
                p_forged = predict_prob_forged(img)
                if float(p_forged) < float(CLS_SKIP_THRESHOLD):
                    writer.writerow({"case_id": s.case_id, "annotation": AUTHENTIC_LABEL})
                    continue
            inst = predict_instances(img)
            writer.writerow({"case_id": s.case_id, "annotation": encode_instances(inst)})

    print("wrote:", SUBMISSION_PATH)
    try:
        with SUBMISSION_PATH.open("r") as f:
            header = f.readline().strip()
        if header != "case_id,annotation":
            raise ValueError(f"Header inesperado: {header!r}")
        print("[SUBMISSION] header OK:", header)
    except Exception:
        print("[SUBMISSION] WARN: não consegui validar o header do CSV.")
        traceback.print_exc()
else:
    print("[SUBMISSION] RUN_SUBMISSION=False; não gerou CSV.")
