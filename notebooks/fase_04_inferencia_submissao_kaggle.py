# %% [markdown]
# # Fase 4 — Inferência + submissão (Kaggle)
#
# Objetivo:
# - Carregar checkpoints treinados (segmentação + opcional classificador).
# - Rodar inferência no `test_images/` e gerar `submission.csv`.
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
print("- Output: /kaggle/working/submission.csv")

# %%
# Célula 2 — Imports + ambiente
import csv
import os
import random
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

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
    # Local
    local_src = Path("src").resolve()
    if (local_src / "forgeryseg" / "__init__.py").exists() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    # Kaggle (dataset com o repo)
    if is_kaggle():
        add_local_package_to_syspath("forgeryseg")
    import forgeryseg  # type: ignore

print("forgeryseg:", Path(forgeryseg.__file__).resolve())

from forgeryseg.checkpoints import build_classifier_from_config, build_segmentation_from_config, load_checkpoint  # noqa: E402
from forgeryseg.constants import AUTHENTIC_LABEL  # noqa: E402
from forgeryseg.dataset import build_test_index, load_image  # noqa: E402
from forgeryseg.inference import normalize_image, predict_image  # noqa: E402
from forgeryseg.postprocess import binarize, extract_components  # noqa: E402
from forgeryseg.rle import encode_instances  # noqa: E402

# %%
# Célula 3 — Dataset (test)


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
test_samples = build_test_index(DATA_ROOT)
print("DATA_ROOT:", DATA_ROOT)
print("test images:", len(test_samples))

# %%
# Célula 4 — Localizar diretórios de checkpoints (Kaggle/local)


def output_root() -> Path:
    if is_kaggle():
        return Path("/kaggle/working")
    return Path(".").resolve()


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

# %%
# Célula 5 — Carregar checkpoints

SEG_MODELS: list[torch.nn.Module] = []
SEG_MODEL_TAGS: list[str] = []

if MODELS_SEG_DIR is None:
    print("[ERRO] MODELS_SEG_DIR não encontrado. Treine e salve em outputs/models_seg/... ou anexe um dataset com isso.")
else:
    seg_ckpts = sorted(MODELS_SEG_DIR.glob("*/*/best.pt"))
    print("seg checkpoints encontrados:", len(seg_ckpts))
    for p in seg_ckpts:
        try:
            state, cfg = load_checkpoint(p)
            m = build_segmentation_from_config(cfg)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            SEG_MODELS.append(m)
            SEG_MODEL_TAGS.append(str(p.relative_to(MODELS_SEG_DIR)))
        except Exception:
            print("[ERRO] falha ao carregar seg checkpoint:", p)
            traceback.print_exc()

print("loaded seg models:", SEG_MODEL_TAGS[:10], ("..." if len(SEG_MODEL_TAGS) > 10 else ""))
if not SEG_MODELS:
    raise RuntimeError("Nenhum modelo de segmentação foi carregado. Veja os erros acima.")


CLS_MODELS: list[torch.nn.Module] = []
CLS_INFER_IMAGE_SIZE = 384
CLS_SKIP_THRESHOLD = 0.30  # mude aqui se quiser (favorece recall em forged com threshold mais baixo)

if MODELS_CLS_DIR is None:
    print("[CLS] MODELS_CLS_DIR não encontrado; gating desativado.")
else:
    cls_ckpts = sorted(MODELS_CLS_DIR.glob("fold_*/best.pt"))
    print("cls checkpoints encontrados:", len(cls_ckpts))
    for p in cls_ckpts:
        try:
            state, cfg = load_checkpoint(p)
            m, image_size = build_classifier_from_config(cfg)
            CLS_INFER_IMAGE_SIZE = int(image_size)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            CLS_MODELS.append(m)
        except Exception:
            print("[ERRO] falha ao carregar cls checkpoint:", p)
            traceback.print_exc()

print("loaded cls models:", len(CLS_MODELS), "CLS_INFER_IMAGE_SIZE:", CLS_INFER_IMAGE_SIZE)

# %%
# Célula 6 — Inferência (ensemble + TTA) + submissão
try:
    from tqdm.auto import tqdm
except Exception:
    print("[WARN] tqdm indisponível; usando loop simples.")

    def tqdm(x, **kwargs):  # type: ignore
        return x


TILE_SIZE = 1024
OVERLAP = 128
MAX_SIZE = 0

THRESHOLD = 0.50
MIN_AREA = 32

USE_TTA = True
TTA_MODES = ("none", "hflip", "vflip")


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


def predict_seg_ensemble_prob(image: np.ndarray) -> np.ndarray:
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
    # pós-processamento simples: remove componentes muito pequenas
    return extract_components(bin_mask, min_area=int(MIN_AREA))


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

