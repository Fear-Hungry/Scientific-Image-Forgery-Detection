# %% [markdown]
# # Fase 4 — Inferência + submissão (Kaggle)
#
# Objetivo:
# - Carregar checkpoints treinados (segmentação + opcional classificador).
# - Rodar inferência no `test_images/` e gerar `submission.csv`.
#
# **Importante**
# - Notebook-only, auto-contido.
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
# Célula 3 — Dataset (test) + IO
from PIL import Image


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
TEST_IMAGES = DATA_ROOT / "test_images"
print("DATA_ROOT:", DATA_ROOT)
print("test images:", len(list(TEST_IMAGES.glob("*.png"))))


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


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

# %%
# Célula 4 — Utilitários (normalize, tiling, CC, RLE)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_image(image: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    x = image.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (x - mean) / std


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
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
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
                tile_norm = normalize_image(tile, mean=mean, std=std)
                tile_tensor = torch.from_numpy(tile_norm).permute(2, 0, 1).unsqueeze(0)
                probs = _predict_tensor(model, tile_tensor, device)
                prob_tile = probs.squeeze(0).squeeze(0).cpu().numpy()
                pred_sum[y0:y1, x0:x1] += prob_tile
                pred_count[y0:y1, x0:x1] += 1.0

        pred = pred_sum / np.maximum(pred_count, 1.0)
        return pred[:orig_h, :orig_w]

    image_norm = normalize_image(image, mean=mean, std=std)
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


def postprocess_binary_mask(mask: np.ndarray, max_hole_area: int = 64, morph_kernel: int = 0) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.max() == 0:
        return m
    if cv2 is None:
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

# %%
# Célula 5 — Localizar diretórios de checkpoints (Kaggle/local)


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
# Célula 6 — Construção de modelos a partir do config salvo no checkpoint


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
            import segmentation_models_pytorch as smp
        except Exception:
            print("[ERRO] checkpoint pede SMP, mas segmentation_models_pytorch não está disponível.")
            traceback.print_exc()
            raise

        encoder_name = str(cfg.get("encoder_name", "efficientnet-b4"))
        classes = int(cfg.get("classes", 1))

        if arch.lower() in {"unetplusplus", "unetpp"}:
            m = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, classes=classes, activation=None)
            return m
        if arch.lower() == "unet":
            m = smp.Unet(encoder_name=encoder_name, encoder_weights=None, classes=classes, activation=None)
            return m
        if arch.lower() in {"deeplabv3plus", "deeplabv3+"}:
            m = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None, classes=classes, activation=None)
            return m

        raise ValueError(f"Arquitetura SMP desconhecida no cfg: {arch!r}")

    # torchvision fallback
    if "deeplab" in arch.lower():
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

    raise ValueError(f"Arquitetura torchvision desconhecida no cfg: {arch!r}")


def build_cls_from_config(cfg: dict) -> nn.Module:
    backend = str(cfg.get("backend", "torchvision"))
    model_name = str(cfg.get("model_name", "resnet50"))
    if backend == "timm":
        try:
            import timm
        except Exception:
            print("[ERRO] checkpoint pede timm, mas timm não está disponível.")
            traceback.print_exc()
            raise
        return timm.create_model(model_name, pretrained=False, num_classes=1)

    from torchvision.models import resnet50

    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m

# %%
# Célula 7 — Carregar checkpoints

SEG_MODELS: list[nn.Module] = []
SEG_MODEL_TAGS: list[str] = []

if MODELS_SEG_DIR is None:
    print("[ERRO] MODELS_SEG_DIR não encontrado. Treine e salve em outputs/models_seg/... ou anexe um dataset com isso.")
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
            SEG_MODEL_TAGS.append(str(p.relative_to(MODELS_SEG_DIR)))
        except Exception:
            print("[ERRO] falha ao carregar seg checkpoint:", p)
            traceback.print_exc()

print("loaded seg models:", SEG_MODEL_TAGS[:10], ("..." if len(SEG_MODEL_TAGS) > 10 else ""))
if not SEG_MODELS:
    raise RuntimeError("Nenhum modelo de segmentação foi carregado. Veja os erros acima.")


CLS_MODELS: list[nn.Module] = []
CLS_SKIP_THRESHOLD = 0.30  # mude aqui se quiser

if MODELS_CLS_DIR is None:
    print("[CLS] MODELS_CLS_DIR não encontrado; gating desativado.")
else:
    cls_ckpts = sorted(MODELS_CLS_DIR.glob("fold_*/best.pt"))
    print("cls checkpoints encontrados:", len(cls_ckpts))
    for p in cls_ckpts:
        try:
            state, cfg = _load_checkpoint(p)
            m = build_cls_from_config(cfg)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            CLS_MODELS.append(m)
            # tenta pegar threshold salvo (se existir)
            if "skip_threshold" in cfg:
                CLS_SKIP_THRESHOLD = float(cfg["skip_threshold"])
        except Exception:
            print("[ERRO] falha ao carregar cls checkpoint:", p)
            traceback.print_exc()

print("loaded cls models:", len(CLS_MODELS))

# %%
# Célula 8 — Inferência (ensemble) + submissão
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


def predict_prob_forged(image: np.ndarray) -> float:
    if not CLS_MODELS:
        raise RuntimeError("CLS_MODELS vazio")
    import torch.nn.functional as F

    x = normalize_image(image)
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    # usa image_size salvo se existir (senão, não força resize)
    probs: list[float] = []
    for m in CLS_MODELS:
        logits = m(t).view(-1)
        probs.append(float(torch.sigmoid(logits)[0].item()))
    return float(np.mean(probs))


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

