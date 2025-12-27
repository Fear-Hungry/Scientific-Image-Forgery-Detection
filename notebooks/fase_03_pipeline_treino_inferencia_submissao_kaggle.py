# %% [markdown]
# # Fase 3 — Pipeline completo (treino + inferência + submissão)
#
# Este notebook é um "guia executável" para:
# 1) Treinar um baseline de segmentação (opcional),
# 2) Validar com oF1 (opcional, depende de `scipy`),
# 3) Rodar inferência no `test_images/` e gerar `submission.csv`.
#
# **Modo Kaggle (Code Competition)**
# - Internet: OFF no momento da submissão.
# - Tempo típico: até 4h.
# - Saída esperada: `submission.csv` (ou `submission.parquet`).
#
# **Importante**
# - Este notebook é **auto-contido** (não importa módulos do projeto).
# - No Kaggle, você só precisa do dataset da competição (e opcionalmente um dataset com pesos/outputs).
#
# ---

# %%
# Célula 1 — Regras do Kaggle (sanidade)
print("Kaggle submission constraints (lembrete):")
print("- Submissions via Notebook")
print("- Runtime <= 4h (CPU/GPU)")
print("- Internet: OFF no submit")
print("- Output: submission.csv ou submission.parquet")

# %%
# Célula 2 — Imports + ambiente
import csv
import json
import os
import random
import sys
import traceback
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

print("python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
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

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Célula 2b — Instalação offline (opcional): wheels via Kaggle Dataset (sem internet)
#
# Se você anexar um Dataset que contenha `wheels/*.whl`, esta célula instala os pacotes **offline** via pip.
# Estruturas suportadas:
# - `/kaggle/input/<dataset>/wheels/*.whl`
# - `/kaggle/input/<dataset>/recodai_bundle/wheels/*.whl`
#
# Se nada for encontrado, nada é instalado (e eventuais imports vão falhar com erro explícito mais adiante).
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
    # Evita varrer o dataset grande da competição quando procurando código.
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
    e adiciona o *root* correspondente ao `sys.path`.

    Isso permite "vendorizar" libs puras Python via GitHub Dataset quando não há wheel disponível.
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
            # 1) package diretamente no root (root/<pkg>/__init__.py)
            pkg = root / package_dir_name
            if (pkg / "__init__.py").exists():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                    added.append(root)
                continue

            # 2) um nível abaixo (root/*/<pkg>/__init__.py) para repositórios dentro de vendor/
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
# Célula 3 — Utilitários (dataset + inferência + RLE + métrica) — auto-contido

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    import albumentations as A
except Exception:
    print("[IMPORT ERROR] albumentations não importou; augmentations ficarão desativadas.")
    traceback.print_exc()
    A = None

try:
    import cv2
except Exception:
    print("[IMPORT ERROR] opencv-python (cv2) não importou; alguns fallbacks/morfologia podem falhar.")
    traceback.print_exc()
    cv2 = None

try:
    from scipy.ndimage import label as _cc_label
except Exception:
    print("[IMPORT ERROR] scipy.ndimage.label não importou; connected components via SciPy ficará indisponível.")
    traceback.print_exc()
    _cc_label = None

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except Exception:
    print("[IMPORT ERROR] scipy.optimize.linear_sum_assignment não importou; métrica oF1 (Hungarian) ficará indisponível.")
    traceback.print_exc()
    _linear_sum_assignment = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_project_root() -> Path:
    """
    Heurística para achar a raiz do projeto local (para ler configs e gravar outputs).
    No Kaggle, normalmente isso vira o CWD (`/kaggle/working`).
    """
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "notebooks").exists() and (p / "README.md").exists():
            return p
        if (p / "configs").exists() and (p / "data").exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()
print("PROJECT_ROOT:", PROJECT_ROOT)


# -----------------------------
# Dataset / indexing
# -----------------------------


@dataclass(frozen=True)
class Sample:
    case_id: str
    image_path: Path
    mask_path: Optional[Path]
    is_authentic: Optional[bool]
    split: str
    label: Optional[str]
    rel_path: Path


def build_train_index(data_root: str | Path, strict: bool = False) -> List[Sample]:
    data_root = Path(data_root)
    train_root = data_root / "train_images"
    mask_root = data_root / "train_masks"

    samples: List[Sample] = []
    for label in ("authentic", "forged"):
        for image_path in sorted((train_root / label).glob("*.png")):
            case_id = image_path.stem
            mask_path = None
            if label == "forged":
                candidate = mask_root / f"{case_id}.npy"
                if candidate.exists():
                    mask_path = candidate
                elif strict:
                    raise FileNotFoundError(f"Missing mask for {case_id}")

            samples.append(
                Sample(
                    case_id=case_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    is_authentic=(label == "authentic"),
                    split="train",
                    label=label,
                    rel_path=image_path.relative_to(data_root),
                )
            )
    return samples


def build_supplemental_index(data_root: str | Path, strict: bool = False) -> List[Sample]:
    data_root = Path(data_root)
    image_root = data_root / "supplemental_images"
    mask_root = data_root / "supplemental_masks"

    samples: List[Sample] = []
    for image_path in sorted(image_root.glob("*.png")):
        case_id = image_path.stem
        mask_path = None
        candidate = mask_root / f"{case_id}.npy"
        if candidate.exists():
            mask_path = candidate
        elif strict:
            raise FileNotFoundError(f"Missing supplemental mask for {case_id}")

        samples.append(
            Sample(
                case_id=case_id,
                image_path=image_path,
                mask_path=mask_path,
                is_authentic=False if mask_path is not None else None,
                split="supplemental",
                label=None,
                rel_path=image_path.relative_to(data_root),
            )
        )
    return samples


def build_test_index(data_root: str | Path) -> List[Sample]:
    data_root = Path(data_root)
    test_root = data_root / "test_images"

    samples: List[Sample] = []
    for image_path in sorted(test_root.glob("*.png")):
        case_id = image_path.stem
        samples.append(
            Sample(
                case_id=case_id,
                image_path=image_path,
                mask_path=None,
                is_authentic=None,
                split="test",
                label=None,
                rel_path=image_path.relative_to(data_root),
            )
        )
    return samples


def load_image(image_path: str | Path, as_rgb: bool = True) -> np.ndarray:
    from PIL import Image

    image_path = Path(image_path)
    with Image.open(image_path) as img:
        if as_rgb:
            img = img.convert("RGB")
        return np.array(img)


def load_mask_instances(mask_path: str | Path) -> List[np.ndarray]:
    mask_path = Path(mask_path)
    masks = np.load(mask_path)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return [(m > 0).astype(np.uint8) for m in masks]


def _load_union_mask(mask_path: Optional[Path], shape: tuple[int, int]) -> np.ndarray:
    if mask_path is None:
        return np.zeros(shape, dtype=np.uint8)
    masks = np.load(mask_path)
    if masks.ndim == 2:
        union = masks
    else:
        union = masks.max(axis=0)
    union = (union > 0).astype(np.uint8)
    if union.shape != shape:
        raise ValueError(f"Mask shape {union.shape} does not match image shape {shape}")
    return union


def _pad_to_size(image: np.ndarray, mask: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    mask_pad = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return image_pad, mask_pad


def _random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_h: int,
    crop_w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    if h == crop_h and w == crop_w:
        return image, mask
    top = int(rng.integers(0, h - crop_h + 1))
    left = int(rng.integers(0, w - crop_w + 1))
    return image[top : top + crop_h, left : left + crop_w], mask[top : top + crop_h, left : left + crop_w]


def _center_crop(image: np.ndarray, mask: np.ndarray, crop_h: int, crop_w: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return image[top : top + crop_h, left : left + crop_w], mask[top : top + crop_h, left : left + crop_w]


def _positive_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_h: int,
    crop_w: int,
    rng: np.random.Generator,
    max_tries: int,
    min_pos_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return _random_crop(image, mask, crop_h, crop_w, rng)

    h, w = mask.shape
    for _ in range(max_tries):
        idx = int(rng.integers(0, len(ys)))
        center_y = int(ys[idx])
        center_x = int(xs[idx])
        top = max(min(center_y - crop_h // 2, h - crop_h), 0)
        left = max(min(center_x - crop_w // 2, w - crop_w), 0)
        crop_mask = mask[top : top + crop_h, left : left + crop_w]
        if int(crop_mask.sum()) >= min_pos_pixels:
            return image[top : top + crop_h, left : left + crop_w], crop_mask

    return _random_crop(image, mask, crop_h, crop_w, rng)


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: List[Sample],
        patch_size: int | tuple[int, int] = 512,
        train: bool = True,
        augment=None,
        positive_prob: float = 0.7,
        min_pos_pixels: int = 1,
        max_tries: int = 10,
        seed: int = 42,
        return_meta: bool = False,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        normalize: bool = True,
    ) -> None:
        self.samples = samples
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.train = train
        self.augment = augment
        self.positive_prob = float(positive_prob)
        self.min_pos_pixels = int(min_pos_pixels)
        self.max_tries = int(max_tries)
        self.seed = int(seed)
        self.return_meta = bool(return_meta)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + idx + worker_id * 100000)

        image = load_image(sample.image_path)
        mask = _load_union_mask(sample.mask_path, image.shape[:2])

        crop_h, crop_w = self.patch_size
        image, mask = _pad_to_size(image, mask, crop_h, crop_w)
        if self.train:
            wants_positive = (sample.is_authentic is False) and (rng.random() < self.positive_prob)
            if wants_positive:
                image, mask = _positive_crop(
                    image,
                    mask,
                    crop_h,
                    crop_w,
                    rng,
                    self.max_tries,
                    self.min_pos_pixels,
                )
            else:
                image, mask = _random_crop(image, mask, crop_h, crop_w, rng)
        else:
            image, mask = _center_crop(image, mask, crop_h, crop_w)

        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        if self.normalize:
            image = (image - self.mean) / self.std

        image = np.transpose(image, (2, 0, 1))
        mask = mask.astype(np.float32)[None, ...]

        x = torch.from_numpy(image)
        y = torch.from_numpy(mask)
        if self.return_meta:
            return x, y, sample
        return x, y


# -----------------------------
# Augmentations (Albumentations + copy-move sintético)
# -----------------------------


def _seg_as_hw(size: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return (int(size), int(size))
    return int(size[0]), int(size[1])


def _seg_has_param(transform_cls, name: str) -> bool:
    import inspect

    return name in inspect.signature(transform_cls).parameters


def _seg_fill_kwargs(transform_cls, fill_value: float | int = 0) -> dict:
    kwargs: dict[str, Any] = {}
    if _seg_has_param(transform_cls, "fill"):
        kwargs["fill"] = fill_value
    if _seg_has_param(transform_cls, "fill_mask"):
        kwargs["fill_mask"] = fill_value
    if _seg_has_param(transform_cls, "value"):
        kwargs["value"] = fill_value
    if _seg_has_param(transform_cls, "mask_value"):
        kwargs["mask_value"] = fill_value
    if _seg_has_param(transform_cls, "cval"):
        kwargs["cval"] = fill_value
    if _seg_has_param(transform_cls, "cval_mask"):
        kwargs["cval_mask"] = fill_value
    return kwargs


def _seg_random_resized_crop(size_hw: tuple[int, int], scale: tuple[float, float], ratio: tuple[float, float], p: float):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    if _seg_has_param(A.RandomResizedCrop, "size"):
        return A.RandomResizedCrop(size=size_hw, scale=scale, ratio=ratio, p=p)
    return A.RandomResizedCrop(height=size_hw[0], width=size_hw[1], scale=scale, ratio=ratio, p=p)


def _seg_image_compression(quality_range: tuple[int, int], p: float):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    if _seg_has_param(A.ImageCompression, "quality_range"):
        return A.ImageCompression(quality_range=quality_range, p=p)
    return A.ImageCompression(quality_lower=quality_range[0], quality_upper=quality_range[1], p=p)


def _seg_coarse_dropout(p: float):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    if _seg_has_param(A.CoarseDropout, "num_holes_range"):
        return A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.03, 0.20),
            hole_width_range=(0.03, 0.20),
            fill=0,
            p=p,
        )
    return A.CoarseDropout(
        max_holes=8,
        max_height=64,
        max_width=64,
        min_holes=1,
        min_height=8,
        min_width=8,
        fill_value=0,
        p=p,
    )


def _seg_gauss_noise(p: float):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    if _seg_has_param(A.GaussNoise, "std_range"):
        # Roughly matches var_limit=(5..50) for uint8 images (std ~= 1..5 px).
        return A.GaussNoise(std_range=(0.005, 0.02), p=p)
    return A.GaussNoise(var_limit=(5.0, 50.0), p=p)


if A is not None:
    from albumentations.core.transforms_interface import DualTransform

    class CopyMoveTransform(DualTransform):
        """Copy-move sintético (só quando a máscara chega vazia).

        Copia uma região da imagem e cola em outro lugar e marca origem+destino na máscara.
        """

        def __init__(
            self,
            min_area_frac: float = 0.05,
            max_area_frac: float = 0.20,
            rotation_limit: float = 15.0,
            scale_range: tuple[float, float] = (0.9, 1.1),
            irregular_prob: float = 0.5,
            max_tries: int = 10,
            only_if_empty_mask: bool = True,
            p: float = 0.25,
        ) -> None:
            super().__init__(p=float(p))
            self.min_area_frac = float(min_area_frac)
            self.max_area_frac = float(max_area_frac)
            self.rotation_limit = float(rotation_limit)
            self.scale_range = (float(scale_range[0]), float(scale_range[1]))
            self.irregular_prob = float(irregular_prob)
            self.max_tries = int(max_tries)
            self.only_if_empty_mask = bool(only_if_empty_mask)

            if not (0.0 < self.min_area_frac <= self.max_area_frac <= 1.0):
                raise ValueError("Expected 0 < min_area_frac <= max_area_frac <= 1")
            if self.max_tries <= 0:
                raise ValueError("max_tries must be > 0")
            if not (0.0 <= self.irregular_prob <= 1.0):
                raise ValueError("irregular_prob must be in [0, 1]")
            if self.scale_range[0] <= 0.0 or self.scale_range[1] <= 0.0:
                raise ValueError("scale_range values must be > 0")

        @property
        def targets_as_params(self):  # type: ignore[override]
            return ["image", "mask"]

        def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
            image = data["image"]
            mask = data.get("mask")
            if mask is None:
                return {"do": False}

            mask = np.asarray(mask)
            if mask.ndim != 2:
                return {"do": False}

            if self.only_if_empty_mask and mask.max() > 0:
                return {"do": False}

            h, w = mask.shape
            if h < 16 or w < 16:
                return {"do": False}

            rg = self.random_generator

            area_frac = float(rg.uniform(self.min_area_frac, self.max_area_frac))
            target_area = area_frac * float(h * w)

            aspect = float(np.exp(rg.uniform(np.log(0.75), np.log(1.3333333333333333))))
            patch_h = int(round(np.sqrt(target_area / aspect)))
            patch_w = int(round(patch_h * aspect))
            patch_h = int(np.clip(patch_h, 8, h - 1))
            patch_w = int(np.clip(patch_w, 8, w - 1))

            y_choices = h - patch_h + 1
            x_choices = w - patch_w + 1
            if y_choices <= 0 or x_choices <= 0:
                return {"do": False}
            if y_choices == 1 and x_choices == 1:
                return {"do": False}

            src_y = int(rg.integers(0, y_choices))
            src_x = int(rg.integers(0, x_choices))

            chosen: tuple[int, int] | None = None
            for _ in range(self.max_tries):
                cand_y = int(rg.integers(0, y_choices))
                cand_x = int(rg.integers(0, x_choices))
                if cand_y == src_y and cand_x == src_x:
                    continue

                if chosen is None:
                    chosen = (cand_y, cand_x)

                y_overlap = max(0, min(src_y + patch_h, cand_y + patch_h) - max(src_y, cand_y))
                x_overlap = max(0, min(src_x + patch_w, cand_x + patch_w) - max(src_x, cand_x))
                if (y_overlap * x_overlap) == 0:
                    chosen = (cand_y, cand_x)
                    break

            if chosen is None:
                dst_y = (src_y + 1) % y_choices if y_choices > 1 else src_y
                dst_x = (src_x + 1) % x_choices if x_choices > 1 else src_x
                if dst_y == src_y and dst_x == src_x:
                    return {"do": False}
            else:
                dst_y, dst_x = chosen

            if float(rg.random()) < self.irregular_prob:
                yy, xx = np.mgrid[0:patch_h, 0:patch_w]
                cy = float(patch_h - 1) / 2.0 + float(rg.uniform(-0.15, 0.15)) * patch_h
                cx = float(patch_w - 1) / 2.0 + float(rg.uniform(-0.15, 0.15)) * patch_w
                ry = max(2.0, float(rg.uniform(0.35, 0.55)) * patch_h)
                rx = max(2.0, float(rg.uniform(0.35, 0.55)) * patch_w)
                mask_src = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0).astype(np.uint8)
            else:
                mask_src = np.ones((patch_h, patch_w), dtype=np.uint8)

            if int(mask_src.sum()) == 0:
                mask_src = np.ones((patch_h, patch_w), dtype=np.uint8)

            angle = float(rg.uniform(-self.rotation_limit, self.rotation_limit)) if self.rotation_limit > 0 else 0.0
            scale = float(rg.uniform(self.scale_range[0], self.scale_range[1]))

            if cv2 is not None and (abs(angle) > 1e-6 or abs(scale - 1.0) > 1e-6):
                center = (float(patch_w) / 2.0, float(patch_h) / 2.0)
                mat = cv2.getRotationMatrix2D(center, angle, scale)
                mask_dst = cv2.warpAffine(
                    mask_src,
                    mat,
                    dsize=(patch_w, patch_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(np.uint8)
                if int(mask_dst.sum()) == 0:
                    mask_dst = mask_src
            else:
                mat = None
                mask_dst = mask_src

            return {
                "do": True,
                "src_y": src_y,
                "src_x": src_x,
                "dst_y": dst_y,
                "dst_x": dst_x,
                "patch_h": patch_h,
                "patch_w": patch_w,
                "mask_src": mask_src,
                "mask_dst": mask_dst,
                "mat": mat,
            }

        def apply(self, img: np.ndarray, *args, **params) -> np.ndarray:
            if not params.get("do", False):
                return img

            src_y = int(params["src_y"])
            src_x = int(params["src_x"])
            dst_y = int(params["dst_y"])
            dst_x = int(params["dst_x"])
            patch_h = int(params["patch_h"])
            patch_w = int(params["patch_w"])
            mask_dst = np.asarray(params["mask_dst"]).astype(bool)
            mat = params.get("mat", None)

            out = img.copy()
            src_patch = out[src_y : src_y + patch_h, src_x : src_x + patch_w].copy()
            if mat is not None and cv2 is not None:
                src_patch = cv2.warpAffine(
                    src_patch,
                    mat,
                    dsize=(patch_w, patch_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )

            dst_patch = out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w]
            dst_patch[mask_dst] = src_patch[mask_dst]
            out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = dst_patch
            return out

        def apply_to_mask(self, mask: np.ndarray, *args, **params) -> np.ndarray:
            if not params.get("do", False):
                return mask

            src_y = int(params["src_y"])
            src_x = int(params["src_x"])
            dst_y = int(params["dst_y"])
            dst_x = int(params["dst_x"])
            patch_h = int(params["patch_h"])
            patch_w = int(params["patch_w"])
            mask_src = np.asarray(params["mask_src"]).astype(bool)
            mask_dst = np.asarray(params["mask_dst"]).astype(bool)

            out = (np.asarray(mask) > 0).astype(np.uint8)
            out[src_y : src_y + patch_h, src_x : src_x + patch_w][mask_src] = 1
            out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w][mask_dst] = 1
            return out

else:
    CopyMoveTransform = None


def get_train_augment(
    patch_size: int | tuple[int, int] | None = None,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    copy_move_prob: float = 0.0,
    copy_move_min_area_frac: float = 0.05,
    copy_move_max_area_frac: float = 0.20,
    copy_move_rotation_limit: float = 15.0,
    copy_move_scale_range: tuple[float, float] = (0.9, 1.1),
):
    if A is None:
        raise ImportError("albumentations is required for augmentations")

    hw = _seg_as_hw(patch_size)
    border_mode = cv2.BORDER_CONSTANT if cv2 is not None else 0

    affine_kwargs: dict[str, Any] = {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.05, 0.05),
        "rotate": (-20, 20),
        "p": 0.75,
        **_seg_fill_kwargs(A.Affine, fill_value=0),
    }
    if _seg_has_param(A.Affine, "interpolation"):
        affine_kwargs["interpolation"] = cv2.INTER_LINEAR if cv2 is not None else 1
    if _seg_has_param(A.Affine, "mask_interpolation"):
        affine_kwargs["mask_interpolation"] = cv2.INTER_NEAREST if cv2 is not None else 0
    if _seg_has_param(A.Affine, "border_mode"):
        affine_kwargs["border_mode"] = border_mode
    elif _seg_has_param(A.Affine, "mode"):
        affine_kwargs["mode"] = border_mode

    transforms = [
        *(
            [
                CopyMoveTransform(
                    min_area_frac=copy_move_min_area_frac,
                    max_area_frac=copy_move_max_area_frac,
                    rotation_limit=copy_move_rotation_limit,
                    scale_range=copy_move_scale_range,
                    p=float(copy_move_prob),
                )
            ]
            if CopyMoveTransform is not None and float(copy_move_prob) > 0.0
            else []
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.25),
        A.Affine(**affine_kwargs),
    ]

    if hw is not None:
        crop_h, crop_w = hw
        transforms.append(_seg_random_resized_crop((crop_h, crop_w), scale=(0.75, 1.0), ratio=(0.85, 1.15), p=0.35))

    transforms.extend(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.CLAHE(clip_limit=(1.0, 3.0), p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    _seg_gauss_noise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.25,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1.0,
                        sigma=40.0,
                        border_mode=border_mode,
                        p=1.0,
                        **_seg_fill_kwargs(A.ElasticTransform, fill_value=0),
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.05,
                        border_mode=border_mode,
                        p=1.0,
                        **_seg_fill_kwargs(A.GridDistortion, fill_value=0),
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.05,
                        border_mode=border_mode,
                        p=1.0,
                        **_seg_fill_kwargs(A.OpticalDistortion, fill_value=0),
                        **({"shift_limit": 0.05} if _seg_has_param(A.OpticalDistortion, "shift_limit") else {}),
                    ),
                ],
                p=0.10,
            ),
            _seg_coarse_dropout(p=0.20),
            _seg_image_compression(quality_range=(60, 100), p=0.10),
        ]
    )

    return A.Compose(transforms)


def get_val_augment(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if A is None:
        raise ImportError("albumentations is required for augmentations")
    return A.Compose([])


# -----------------------------
# Inferência (tile + resize)
# -----------------------------


def normalize_image(image: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (image - mean) / std


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
def _predict_tensor(model, tensor: torch.Tensor, device: str) -> torch.Tensor:
    tensor = tensor.to(device)
    logits = model(tensor)
    return torch.sigmoid(logits)


def predict_image(
    model,
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


# -----------------------------
# Pós-processamento (binário + CC)
# -----------------------------


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(mask) >= float(threshold)).astype(np.uint8)


def extract_components(mask: np.ndarray, min_area: int = 0) -> List[np.ndarray]:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if mask.max() == 0:
        return []

    instances: List[np.ndarray] = []
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


# -----------------------------
# RLE (formato oficial)
# -----------------------------

AUTHENTIC_LABEL = "authentic"


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
    return (arr > 0).astype(np.uint8)


def rle_encode(mask: np.ndarray) -> List[int]:
    mask = _normalize_mask(mask)
    if mask.max() == 0:
        return []

    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return changes.tolist()


def rle_decode(rle: Sequence[int] | str | None, shape: tuple[int, int]) -> np.ndarray:
    if rle is None:
        return np.zeros(shape, dtype=np.uint8)

    if isinstance(rle, str):
        text = rle.strip()
        if text == "" or text.lower() == AUTHENTIC_LABEL:
            return np.zeros(shape, dtype=np.uint8)
        if text.startswith("["):
            rle = json.loads(text)
        else:
            rle = [int(x) for x in text.split()]

    rle = list(rle)
    if len(rle) % 2 != 0:
        raise ValueError("RLE length must be even (start, length pairs)")

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(rle[0::2], rle[1::2]):
        if length <= 0:
            continue
        start_index = int(start) - 1
        end_index = start_index + int(length)
        mask[start_index:end_index] = 1

    return mask.reshape(shape, order="F")


def _normalize_instances(masks: Iterable[np.ndarray] | np.ndarray | None) -> List[np.ndarray]:
    if masks is None:
        return []
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            return [_normalize_mask(masks)]
        if masks.ndim == 3:
            return [_normalize_mask(m) for m in masks]
    if isinstance(masks, (list, tuple)):
        return [_normalize_mask(m) for m in masks]
    raise ValueError("Unsupported mask container for RLE encoding")


def encode_instances(masks: Iterable[np.ndarray] | np.ndarray | None) -> str:
    instances = _normalize_instances(masks)
    parts: List[str] = []
    for mask in instances:
        runs = rle_encode(mask)
        if runs:
            parts.append(json.dumps(runs))
    if not parts:
        return AUTHENTIC_LABEL
    return ";".join(parts)


def decode_annotation(annotation: str | None, shape: tuple[int, int]) -> List[np.ndarray]:
    if annotation is None:
        return []

    text = annotation.strip()
    if text == "" or text.lower() == AUTHENTIC_LABEL:
        return []

    masks: List[np.ndarray] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        masks.append(rle_decode(part, shape))
    return masks


# -----------------------------
# Métrica (oF1 por instância)
# -----------------------------


def _as_instance_list(masks: Iterable[np.ndarray] | np.ndarray | None) -> List[np.ndarray]:
    if masks is None:
        return []
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            return extract_components(masks)
        if masks.ndim == 3:
            return [(masks[i] > 0).astype(np.uint8) for i in range(masks.shape[0])]
    if isinstance(masks, (list, tuple)):
        return [(np.asarray(m) > 0).astype(np.uint8) for m in masks]
    raise ValueError("Unsupported mask container")


def _build_f1_matrix(gt_instances: List[np.ndarray], pred_instances: List[np.ndarray]) -> np.ndarray:
    gt_count = len(gt_instances)
    pred_count = len(pred_instances)
    if gt_count == 0 or pred_count == 0:
        return np.zeros((gt_count, pred_count), dtype=np.float32)

    gt_sums = [int(m.sum()) for m in gt_instances]
    pred_sums = [int(m.sum()) for m in pred_instances]

    f1_matrix = np.zeros((gt_count, pred_count), dtype=np.float32)
    for i, gt_mask in enumerate(gt_instances):
        if gt_sums[i] == 0:
            continue
        for j, pred_mask in enumerate(pred_instances):
            if pred_sums[j] == 0:
                continue
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            if intersection == 0:
                continue
            f1_matrix[i, j] = (2.0 * intersection) / (gt_sums[i] + pred_sums[j])
    return f1_matrix


def score_image(gt_masks: Iterable[np.ndarray] | np.ndarray | None, pred_masks: Iterable[np.ndarray] | np.ndarray | None) -> float:
    gt_instances = _as_instance_list(gt_masks)
    pred_instances = _as_instance_list(pred_masks)

    gt_count = len(gt_instances)
    pred_count = len(pred_instances)

    if gt_count == 0 and pred_count == 0:
        return 1.0
    if gt_count == 0 and pred_count > 0:
        return 0.0
    if gt_count > 0 and pred_count == 0:
        return 0.0

    if _linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian matching")

    f1_matrix = _build_f1_matrix(gt_instances, pred_instances)
    row_ind, col_ind = _linear_sum_assignment(-f1_matrix)
    if row_ind.size == 0:
        return 0.0

    matched = f1_matrix[row_ind, col_ind]
    if pred_count < gt_count:
        base = float(matched.sum() / gt_count)
    else:
        base = float(matched.mean())
    penalty = gt_count / max(pred_count, gt_count)
    return base * penalty


# -----------------------------
# Treino (segmentação)
# -----------------------------


@dataclass
class TrainStats:
    loss: float


@torch.no_grad()
def _batch_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= float(threshold)).float()
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return dice.mean()


def train_one_epoch(model, loader, criterion, optimizer, device: str, use_amp: bool = False) -> TrainStats:
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)

    return TrainStats(loss=total_loss / max(len(loader.dataset), 1))


def validate(model, loader, criterion, device: str) -> tuple[TrainStats, float]:
    model.eval()
    total_loss = 0.0
    dice_scores: list[float] = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += float(loss.item()) * images.size(0)
            dice_scores.append(float(_batch_dice(logits, masks)))

    mean_loss = total_loss / max(len(loader.dataset), 1)
    mean_dice = float(sum(dice_scores) / max(len(dice_scores), 1))
    return TrainStats(loss=mean_loss), mean_dice


# -----------------------------
# Losses (segmentação)
# -----------------------------


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class BCETverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, tversky_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.tversky_weight = float(tversky_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.tversky_weight * self.tversky(logits, targets)

# %%
# Célula 4 — Paths do dataset (Kaggle/local) + config

KAGGLE_COMP_DATASET = Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection")
if (KAGGLE_COMP_DATASET / "train_images").exists():
    DATA_ROOT = KAGGLE_COMP_DATASET
elif (PROJECT_ROOT / "data" / "train_images").exists():
    DATA_ROOT = PROJECT_ROOT / "data"
else:
    DATA_ROOT = PROJECT_ROOT / "data" / "recodai"

OUTPUT_ROOT = Path("/kaggle/working/outputs") if is_kaggle() else (PROJECT_ROOT / "outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline_fpn_convnext.json"
cfg: dict = {}
if CONFIG_PATH.exists():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

print("DATA_ROOT:", DATA_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("Config loaded:", bool(cfg))


def _find_checkpoint_root(default_dir: Path, *, expected_subdir: str) -> Path:
    """
    Procura por checkpoints em locais comuns (incluindo `/kaggle/input/*`) e retorna o primeiro diretório
    que contenha algum arquivo `.pt`.
    """
    candidates: list[Path] = [Path(default_dir)]

    # comum em repos locais: <repo>/outputs/<expected_subdir>
    candidates.append(PROJECT_ROOT / "outputs" / expected_subdir)
    candidates.append(PROJECT_ROOT / "weights" / expected_subdir)

    if is_kaggle():
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for ds in sorted(kaggle_input.glob("*")):
                candidates.append(ds / "outputs" / expected_subdir)
                candidates.append(ds / expected_subdir)
                candidates.append(ds / "weights" / expected_subdir)
                candidates.append(ds / "recodai_bundle" / "outputs" / expected_subdir)
                candidates.append(ds / "recodai_bundle" / expected_subdir)
                candidates.append(ds / "recodai_bundle" / "weights" / expected_subdir)

    for p in candidates:
        try:
            if p.exists() and any(p.glob("**/best.pt")):
                return p
        except Exception as exc:
            print(f"[CKPT] erro ao checar diretório: {p}")
            traceback.print_exc()
            continue
    return Path(default_dir)

# %%
# Célula 5 — Index (train/test) + contagens

train_samples = build_train_index(DATA_ROOT)
test_samples = build_test_index(DATA_ROOT)

print("Train samples:", len(train_samples))
print("Test samples:", len(test_samples))
print("Train authentic:", sum(1 for s in train_samples if s.is_authentic))
print("Train forged:", sum(1 for s in train_samples if s.is_authentic is False))

# %% [markdown]
# ## Análise dos Dados e Pré-processamento
#
# Antes de treinar modelos, é útil realizar uma exploração dos dados. Cada imagem possui um identificador `case_id`.
# No dataset de treino, existe um subconjunto de imagens **autênticas** (sem manipulação) e imagens **forjadas**
# (copy-move). Para as imagens forjadas, há uma máscara de segmentação indicando os pixels duplicados; para imagens
# autênticas, não há máscara (equivalente a "nenhum pixel forjado").
#
# **Observação importante (treino):** no snapshot do Kaggle, o mesmo `case_id` pode aparecer em **`authentic/` e
# `forged/`** (duas imagens diferentes). Para indexar sem colisões, use o caminho relativo (`rel_path`) ou uma chave
# composta (ex.: `f\"{label}/{case_id}\"`).
#
# O que precisamos construir:
#
# - **Segmentação:** pares `(imagem, máscara)` (aqui usamos a **união** das instâncias como máscara binária, e depois
#   recuperamos instâncias via componentes conexos na hora do `submission`).
# - **Classificação (opcional):** rótulo binário `y_cls` para decidir se é `authentic` (0) ou `forged` (1).
#
# Pré-processamento (baseline deste repo):
#
# - Leitura com PIL e conversão para **RGB**.
# - Conversão para `float32`, escala para `[0, 1]` e **normalização ImageNet**.
# - Treino *patch-based* (`PatchDataset`): amostra crops de tamanho `patch_size`, com *oversampling* de regiões positivas
#   em imagens forjadas (controlado por `positive_prob`/`min_pos_pixels`).
# - Inferência em imagem inteira via **tiling** (`tile_size`/`overlap`) para lidar com imagens grandes.
#
# ### Dimensionamento e formato (decisão do baseline)
#
# - **Canais:** padronizamos todas as imagens para **3 canais (RGB)**. Se a imagem for originalmente em escala de cinza,
#   duplicamos o canal (via `PIL.Image.convert("RGB")`), o que funciona bem para *backbones* pré-treinados em ImageNet.
# - **Tamanho no treino:** ao invés de fazer *downscale* agressivo da figura inteira (que pode apagar falsificações
#   pequenas), treinamos com **patches 512×512** (crop) e fazemos **padding** quando a imagem é menor. Isso fixa o shape
#   de entrada e mantém detalhes locais.
# - **Tamanho na inferência:** rodamos em **tiles** (ex.: `tile_size=1024`, `overlap=128`) para preservar resolução em
#   imagens grandes. Se o runtime ficar inviável, use `MAX_SIZE` para limitar o lado maior (trade-off controlado).
#
# ### Normalização (decisão do baseline)
#
# - **Escala:** convertemos para `float32` e, quando a imagem vem em `uint8`, reescalamos para **[0, 1]**.
# - **Padronização por canal:** aplicamos **média/desvio do ImageNet** (o padrão esperado por encoders pré-treinados).
# - **Sem equalização fixa:** não aplicamos equalização/histogram matching como pré-processamento determinístico para não
#   correr o risco de mascarar/alterar evidências sutis de copy-move. Em vez disso, lidamos com variações de contraste
#   via **data augmentation** (ex.: `RandomBrightnessContrast`, `RandomGamma`, `CLAHE`) e pela robustez do modelo.
#
# ### Divisão de dados (decisão do baseline: 5-fold CV + ensemble)
#
# Em *code competitions*, o conjunto de teste real é **oculto** e não existe um "val set oficial" fixo. Para
# desenvolvimento local, precisamos criar uma validação a partir do treino:
#
# - **Opção simples:** holdout (ex.: 80/20 estratificado).
# - **Opção de performance:** **K-fold cross-validation** (ex.: 5-fold), treinando 5 modelos e fazendo **ensemble** na
#   inferência. Isso melhora o uso do treino e tende a reduzir overfitting, mas custa ~5× mais tempo de treino.
#
# **Escolha aqui:** usamos **5 folds** e fazemos **ensemble** dos modelos finais.
# Para evitar vazamento, fazemos o split **agrupando por `case_id`** (quando existe par `authentic/` e `forged/` com o
# mesmo id, eles caem no mesmo fold).
#

# %%
# Célula 5b — Exemplo: carregando (imagem, máscara) e label binário
sample0 = train_samples[0]
image0 = load_image(sample0.image_path)
gt_instances0 = load_mask_instances(sample0.mask_path) if sample0.mask_path else []

if gt_instances0:
    union_mask0 = np.max(np.stack(gt_instances0, axis=0), axis=0).astype(np.uint8)
else:
    union_mask0 = np.zeros(image0.shape[:2], dtype=np.uint8)

y_cls0 = 0 if sample0.is_authentic else 1

print("case_id:", sample0.case_id)
print("label:", sample0.label, "| y_cls:", y_cls0)
print("image shape:", image0.shape, "dtype:", image0.dtype)
print("instances:", len(gt_instances0), "| union mask sum:", int(union_mask0.sum()))

# %%
# Célula 5c — EDA rápida (opcional): tamanhos e áreas de máscara
RUN_EDA = False  # deixe False no submit; True para explorar interativamente

if RUN_EDA:
    import pandas as pd
    from IPython.display import display
    from PIL import Image

    rows = []
    for s in train_samples:
        width = height = None
        mode = None
        with Image.open(s.image_path) as img:
            width, height = img.size
            mode = img.mode

        mask_instances = 0
        mask_area = 0
        mask_area_frac = 0.0
        if s.mask_path is not None:
            masks = np.load(s.mask_path)
            if masks.ndim == 2:
                masks = masks[None, ...]
            mask_instances = int(masks.shape[0])
            union = masks.max(axis=0)
            mask_area = int((union > 0).sum())
            mask_area_frac = mask_area / float(width * height)

        rows.append(
            {
                "case_id": s.case_id,
                "label": s.label,
                "rel_path": str(s.rel_path),
                "width": width,
                "height": height,
                "mode": mode,
                "mask_instances": mask_instances,
                "mask_area": mask_area,
                "mask_area_frac": mask_area_frac,
            }
        )

    df_train = pd.DataFrame(rows)
    display(df_train.head())
    display(df_train["label"].value_counts())
    display(df_train["mode"].value_counts())
    print("unique case_id:", int(df_train["case_id"].nunique()))
    print("duplicated case_id (train):", int(df_train.duplicated("case_id").sum()))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.scatter(df_train["width"], df_train["height"], s=3, alpha=0.25)
        plt.title("Train image sizes (width x height)")
        plt.xlabel("width")
        plt.ylabel("height")
        plt.show()

        plt.figure(figsize=(6, 4))
        df_train[df_train["label"] == "forged"]["mask_area_frac"].hist(bins=40)
        plt.title("Mask area fraction (forged)")
        plt.xlabel("mask_area_frac")
        plt.show()
    except Exception as exc:
        print("[PLOTS] plots indisponíveis (erro abaixo):")
        traceback.print_exc()

# %%
# Célula 6 — Split em folds (5-fold) com agrupamento por case_id


def _case_id_groups(samples) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, s in enumerate(samples):
        groups.setdefault(str(s.case_id), []).append(int(idx))
    return groups


def iter_case_id_folds(samples, n_splits: int, seed: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Gera folds garantindo que o mesmo case_id não apareça em treino e validação.
    A estratificação é feita no nível do case_id por "tipo de grupo" (par vs solo),
    o que ajuda a manter a proporção authentic/forged por fold no snapshot deste dataset.
    """
    groups = _case_id_groups(samples)
    case_ids = sorted(groups.keys())
    # 0 = par (authentic+forged), 1 = solo (apenas forged)
    y_group = np.array([0 if len(groups[cid]) >= 2 else 1 for cid in case_ids], dtype=int)

    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_g, val_g in splitter.split(np.zeros(len(case_ids)), y_group):
            train_idx: list[int] = []
            val_idx: list[int] = []
            for gi in train_g:
                train_idx.extend(groups[case_ids[int(gi)]])
            for gi in val_g:
                val_idx.extend(groups[case_ids[int(gi)]])
            yield np.array(sorted(train_idx), dtype=int), np.array(sorted(val_idx), dtype=int)
        return
    except Exception:
        print("[FOLDS] sklearn.model_selection.StratifiedKFold indisponível; usando split fallback (não-estratificado).")
        traceback.print_exc()

    rng = np.random.default_rng(seed)
    indices = np.arange(len(case_ids))
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(y_group):
        label_indices = indices[y_group == label]
        rng.shuffle(label_indices)
        for i, idx in enumerate(label_indices):
            folds[i % n_splits].append(int(idx))

    for fold_idx in range(n_splits):
        val_g = np.array(sorted(folds[fold_idx]), dtype=int)
        val_g_set = set(val_g.tolist())
        train_g = np.array([int(i) for i in indices if int(i) not in val_g_set], dtype=int)

        train_idx: list[int] = []
        val_idx: list[int] = []
        for gi in train_g:
            train_idx.extend(groups[case_ids[int(gi)]])
        for gi in val_g:
            val_idx.extend(groups[case_ids[int(gi)]])

        yield np.array(sorted(train_idx), dtype=int), np.array(sorted(val_idx), dtype=int)


N_FOLDS = int(cfg.get("folds", 5)) if cfg else 5
FOLD = 0

folds = list(iter_case_id_folds(train_samples, n_splits=N_FOLDS, seed=SEED))
train_idx, val_idx = folds[FOLD]

train_fold_samples = [train_samples[int(i)] for i in train_idx]
val_fold_samples = [train_samples[int(i)] for i in val_idx]

print(f"fold {FOLD}/{N_FOLDS}: train={len(train_fold_samples)} val={len(val_fold_samples)}")
print("val forged:", sum(1 for s in val_fold_samples if s.is_authentic is False))

# %% [markdown]
# ## Classificador (authentic vs forged) — opcional
#
# Usamos um classificador binário para:
#
# 1) **Gating**: pular a segmentação em imagens previstas como autênticas (economiza tempo), e
# 2) **Sinal adicional**: combinar a confiança do classificador com a segmentação.
#
# Decisão do baseline:
#
# - Backbone via **timm** (ex.: EfficientNet-B4 `tf_efficientnet_b4_ns`).
# - Saída binária com 1 neurônio (logits) + `BCEWithLogitsLoss`.
# - Split **sem vazamento por `case_id`**: reutilizamos os mesmos folds construídos acima.
#

# %%
# Célula 6b — Config + dataset/augs + treino do classificador (opcional)
import inspect

try:
    import albumentations as A
except Exception:
    print("[IMPORT ERROR] albumentations não importou (classificador); treino seguirá sem augmentations.")
    traceback.print_exc()
    A = None

try:
    import timm
except Exception:
    print("[IMPORT ERROR] timm não importou; classificador ficará indisponível.")
    traceback.print_exc()
    # Tenta fallback por código "vendor" via Kaggle Dataset (GitHub import).
    add_local_package_to_syspath("timm")
    try:
        import timm  # type: ignore[no-redef]
    except Exception:
        timm = None

try:
    from sklearn.metrics import f1_score as _sk_f1_score
    from sklearn.metrics import roc_auc_score as _sk_roc_auc_score
except Exception:
    _sk_f1_score = None
    _sk_roc_auc_score = None
    print("[IMPORT ERROR] sklearn.metrics não importou; AUC/F1 do classificador não serão calculados.")
    traceback.print_exc()


def _has_param(transform_cls, name: str) -> bool:
    return name in inspect.signature(transform_cls).parameters


def _random_resized_crop(size: int, scale=(0.7, 1.0), ratio=(0.85, 1.15), p: float = 1.0):
    if A is None:
        raise ImportError("albumentations is required for classifier augmentations")
    if _has_param(A.RandomResizedCrop, "size"):
        return A.RandomResizedCrop(size=(int(size), int(size)), scale=scale, ratio=ratio, p=p)
    return A.RandomResizedCrop(height=int(size), width=int(size), scale=scale, ratio=ratio, p=p)


def _gauss_noise(p: float = 1.0):
    if A is None:
        raise ImportError("albumentations is required for classifier augmentations")
    if _has_param(A.GaussNoise, "std_range"):
        return A.GaussNoise(std_range=(0.005, 0.02), p=p)
    return A.GaussNoise(var_limit=(5.0, 50.0), p=p)


def _image_compression(quality_range=(60, 100), p: float = 0.10):
    if A is None:
        raise ImportError("albumentations is required for classifier augmentations")
    if _has_param(A.ImageCompression, "quality_range"):
        return A.ImageCompression(quality_range=quality_range, p=p)
    return A.ImageCompression(quality_lower=int(quality_range[0]), quality_upper=int(quality_range[1]), p=p)


def build_cls_train_augment(image_size: int):
    if A is None:
        raise ImportError("albumentations is required for classifier augmentations")
    # Por padrão usamos augs "básicas" (flips/rotações leves) e evitamos
    # copy-move sintético no classificador para não viciar em artefatos.
    if CLS_AUG_MODE == "basic":
        return A.Compose(
            [
                A.LongestMaxSize(max_size=int(image_size), p=1.0),
                A.PadIfNeeded(
                    min_height=int(image_size),
                    min_width=int(image_size),
                    border_mode=0,
                    fill=0,
                    fill_mask=0,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.25),
            ]
        )
    return A.Compose(
        [
            # Augs um pouco mais fortes (ainda SEM falsificação sintética).
            A.LongestMaxSize(max_size=int(image_size), p=1.0),
            A.PadIfNeeded(
                min_height=int(image_size),
                min_width=int(image_size),
                border_mode=0,
                fill=0,
                fill_mask=0,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.25),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.CLAHE(clip_limit=(1.0, 3.0), p=1.0),
                ],
                p=0.35,
            ),
            A.OneOf([_gauss_noise(p=1.0), A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.15),
            _image_compression(quality_range=(60, 100), p=0.10),
        ]
    )


def build_cls_val_augment(image_size: int):
    if A is None:
        raise ImportError("albumentations is required for classifier augmentations")
    return A.Compose(
        [
            A.LongestMaxSize(max_size=int(image_size), p=1.0),
            A.PadIfNeeded(
                min_height=int(image_size),
                min_width=int(image_size),
                border_mode=0,
                fill=0,
                fill_mask=0,
                p=1.0,
            ),
        ]
    )


CLS_MODEL_NAME = "tf_efficientnet_b4_ns"
CLS_IMAGE_SIZE = 380
CLS_BATCH_SIZE = 16
CLS_EPOCHS = 15
CLS_LR = 2e-4
CLS_WEIGHT_DECAY = 1e-4
CLS_USE_AMP = DEVICE.startswith("cuda")
CLS_NUM_WORKERS = int(cfg.get("num_workers", 2)) if cfg else 2
CLS_AUG_MODE = "basic"  # "basic" (recomendado) ou "strong"
# Se `prob_forged` ficar abaixo deste valor, marcamos como `authentic` e pulamos a segmentação.
# Deixe baixo para minimizar falsos negativos (forged -> authentic).
CLS_SKIP_THRESHOLD = 0.10
CLS_EARLY_STOPPING = True
CLS_PATIENCE = 3
CLS_MIN_DELTA = 1e-4

CLS_SAVE_DIR = OUTPUT_ROOT / "models_cls"
CLS_LOAD_DIR = _find_checkpoint_root(CLS_SAVE_DIR, expected_subdir="models_cls")

print("classifier checkpoints (load):", CLS_LOAD_DIR)
print("classifier checkpoints (save):", CLS_SAVE_DIR)

RUN_CLS_TRAIN = False  # mude para True para treinar o classificador aqui
TRAIN_CLS_FOLDS = [FOLD]  # para CV, use: list(range(N_FOLDS))


def build_cls_model(model_name: str, pretrained: bool = True) -> torch.nn.Module:
    if timm is None:
        raise ImportError("timm is required for classifier model")
    try:
        m = timm.create_model(model_name, pretrained=bool(pretrained), num_classes=1, in_chans=3)
    except Exception as exc:
        if pretrained:
            print("[CLS] pretrained weights falharam; seguindo sem pesos. Erro abaixo:")
            traceback.print_exc()
            m = timm.create_model(model_name, pretrained=False, num_classes=1, in_chans=3)
        else:
            raise

    reset = getattr(m, "reset_classifier", None)
    if callable(reset):
        try:
            reset(num_classes=1)
        except TypeError:
            reset(1)
    return m


class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, samples, augment=None) -> None:
        self.samples = list(samples)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = load_image(s.image_path)
        if self.augment is not None:
            img = self.augment(image=img)["image"]
        img = normalize_image(img)
        x = torch.from_numpy(img).permute(2, 0, 1)
        y = torch.tensor([0.0 if s.is_authentic else 1.0], dtype=torch.float32)
        return x, y


def train_one_epoch_cls(model, loader, criterion, optimizer) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler(enabled=CLS_USE_AMP)

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).view(-1)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=CLS_USE_AMP):
            logits = model(x).view(-1)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).float()
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            total_loss += float(loss.item()) * int(y.numel())

    return {"loss": total_loss / float(max(total, 1)), "acc": float(correct) / float(max(total, 1))}


@torch.no_grad()
def validate_cls(model, loader, criterion) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    probs_all = []
    y_all = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).view(-1)
        logits = model(x).view(-1)
        loss = criterion(logits, y)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()

        correct += int((pred == y).sum().item())
        total += int(y.numel())
        total_loss += float(loss.item()) * int(y.numel())

        probs_all.append(prob.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    out = {"loss": total_loss / float(max(total, 1)), "acc": float(correct) / float(max(total, 1))}
    probs_np = np.concatenate(probs_all) if probs_all else np.array([], dtype=np.float32)
    y_np = np.concatenate(y_all) if y_all else np.array([], dtype=np.float32)
    if probs_np.size:
        thr = float(CLS_SKIP_THRESHOLD)
        y_pred = (probs_np >= thr).astype(np.uint8)
        y_true = y_np.astype(np.uint8)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        out["thr"] = thr
        out["tp"] = tp
        out["fn"] = fn
        out["fp"] = fp
        out["tn"] = tn
        out["recall_forged"] = float(tp) / float(max(tp + fn, 1))
        out["fpr_auth"] = float(fp) / float(max(fp + tn, 1))
        out["skip_rate_total"] = float(tn + fn) / float(max(tn + fp + tp + fn, 1))
        out["skip_rate_auth"] = float(tn) / float(max(tn + fp, 1))
        out["skip_rate_forged"] = float(fn) / float(max(fn + tp, 1))

    if _sk_roc_auc_score is not None and probs_np.size and np.unique(y_np).size > 1:
        out["auc"] = float(_sk_roc_auc_score(y_np, probs_np))
    if _sk_f1_score is not None and probs_np.size:
        out["f1"] = float(_sk_f1_score(y_np, probs_np >= 0.5))

    return out


if RUN_CLS_TRAIN:
    from torch.utils.data import DataLoader

    for fold_id in TRAIN_CLS_FOLDS:
        train_idx, val_idx = folds[int(fold_id)]
        cls_train_samples = [train_samples[int(i)] for i in train_idx]
        cls_val_samples = [train_samples[int(i)] for i in val_idx]

        try:
            cls_train_aug = build_cls_train_augment(CLS_IMAGE_SIZE)
            cls_val_aug = build_cls_val_augment(CLS_IMAGE_SIZE)
        except Exception as exc:
            print("[CLS] erro ao construir augmentations do classificador; treinando sem augs. Erro abaixo:")
            traceback.print_exc()
            cls_train_aug = None
            cls_val_aug = None

        ds_tr = ClsDataset(cls_train_samples, augment=cls_train_aug)
        ds_va = ClsDataset(cls_val_samples, augment=cls_val_aug)

        loader_tr = DataLoader(
            ds_tr,
            batch_size=CLS_BATCH_SIZE,
            shuffle=True,
            num_workers=CLS_NUM_WORKERS,
            pin_memory=DEVICE.startswith("cuda"),
        )
        loader_va = DataLoader(
            ds_va,
            batch_size=CLS_BATCH_SIZE,
            shuffle=False,
            num_workers=CLS_NUM_WORKERS,
            pin_memory=DEVICE.startswith("cuda"),
        )

        model_cls = build_cls_model(CLS_MODEL_NAME, pretrained=True).to(DEVICE)

        pos = sum(1 for s in cls_train_samples if s.is_authentic is False)
        neg = sum(1 for s in cls_train_samples if s.is_authentic)
        pos_weight = float(neg) / float(max(pos, 1))

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
        optimizer = torch.optim.AdamW(model_cls.parameters(), lr=CLS_LR, weight_decay=CLS_WEIGHT_DECAY)

        out_dir = CLS_SAVE_DIR / f"fold_{int(fold_id)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cls_ckpt = out_dir / "best.pt"

        best_score = -1e9
        bad_epochs = 0
        for epoch in range(1, CLS_EPOCHS + 1):
            tr = train_one_epoch_cls(model_cls, loader_tr, criterion, optimizer)
            va = validate_cls(model_cls, loader_va, criterion)
            extra = []
            if "auc" in va:
                extra.append(f"val_auc={va['auc']:.4f}")
            if "f1" in va:
                extra.append(f"val_f1={va['f1']:.4f}")
            if "recall_forged" in va:
                extra.append(f"val_recall_forged@{va.get('thr', CLS_SKIP_THRESHOLD):.2f}={va['recall_forged']:.4f}")
                extra.append(f"FN={int(va.get('fn', 0))}")
            print(
                f"[CLS] fold {int(fold_id)} epoch {epoch:02d}/{CLS_EPOCHS} "
                f"tr_loss={tr['loss']:.4f} tr_acc={tr['acc']:.4f} "
                f"va_loss={va['loss']:.4f} va_acc={va['acc']:.4f} {' '.join(extra)}".strip()
            )

            score = float(va["auc"]) if "auc" in va else (-float(va["loss"]))
            if score > best_score + CLS_MIN_DELTA:
                best_score = score
                bad_epochs = 0
                torch.save(
                    {
                        "model_state": model_cls.state_dict(),
                        "epoch": epoch,
                        "val_auc": va.get("auc"),
                        "val_loss": va["loss"],
                        "config": {
                            "model_name": CLS_MODEL_NAME,
                            "image_size": CLS_IMAGE_SIZE,
                            "aug_mode": CLS_AUG_MODE,
                        },
                    },
                    cls_ckpt,
                )
                print("[CLS] saved:", cls_ckpt)
            else:
                bad_epochs += 1

            if CLS_EARLY_STOPPING and bad_epochs >= CLS_PATIENCE:
                print(f"[CLS] early stopping: sem melhora por {CLS_PATIENCE} épocas")
                break

        del model_cls, optimizer
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

# %%
# Célula 6c — (Opcional) Tuning do threshold para favorecer recall (evitar falsos negativos)
RUN_CLS_THRESHOLD_TUNING = False
CLS_TARGET_RECALL = 0.995
CLS_SKIP_THRESHOLD_GRID = np.linspace(0.01, 0.80, 80)


def _confusion_from_probs(y_true: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    y_true = y_true.astype(np.uint8)
    y_pred = (probs >= float(thr)).astype(np.uint8)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    recall = float(tp) / float(max(tp + fn, 1))
    fpr = float(fp) / float(max(fp + tn, 1))
    return {"thr": float(thr), "tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": recall, "fpr": fpr}


@torch.no_grad()
def _collect_probs_cls(model, samples, image_size: int):
    import torch.nn.functional as F

    probs: list[float] = []
    y_true: list[int] = []
    keys: list[str] = []

    try:
        aug = build_cls_val_augment(image_size) if A is not None else None
    except Exception:
        print("[CLS] erro ao construir augmentation de validação; seguindo sem augs. Erro abaixo:")
        traceback.print_exc()
        aug = None

    for s in samples:
        img = load_image(s.image_path)
        if aug is not None:
            img = aug(image=img)["image"]
        img = normalize_image(img)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        if image_size and x.shape[-2:] != (image_size, image_size):
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        logit = model(x).view(-1)
        prob = float(torch.sigmoid(logit)[0].item())
        probs.append(prob)
        y_true.append(0 if s.is_authentic else 1)
        keys.append(str(s.rel_path))

    return np.asarray(y_true, dtype=np.uint8), np.asarray(probs, dtype=np.float32), keys


if RUN_CLS_THRESHOLD_TUNING:
    # Usa o fold atual por padrão (melhor: OOF com todos folds, se você treinou todos).
    fold_id = int(FOLD)
    cls_ckpt_path = (CLS_LOAD_DIR / f"fold_{fold_id}") / "best.pt"
    if not cls_ckpt_path.exists():
        print("[CLS] checkpoint não encontrado:", cls_ckpt_path)
    else:
        state, cfg_cls = _load_checkpoint(cls_ckpt_path)
        model_name = cfg_cls.get("model_name", CLS_MODEL_NAME)
        image_size = int(cfg_cls.get("image_size", CLS_IMAGE_SIZE))
        model_cls = build_cls_model(model_name, pretrained=False).to(DEVICE)
        model_cls.load_state_dict(state)
        model_cls.eval()

        y_true, probs, keys = _collect_probs_cls(model_cls, val_fold_samples, image_size=image_size)

        rows = [_confusion_from_probs(y_true, probs, thr) for thr in CLS_SKIP_THRESHOLD_GRID]
        feasible = [r for r in rows if r["recall"] >= float(CLS_TARGET_RECALL)]
        if feasible:
            best = max(feasible, key=lambda r: r["thr"])  # maior thr mantendo recall => menos falsos positivos
        else:
            best = max(rows, key=lambda r: (r["recall"], r["thr"]))

        CLS_SKIP_THRESHOLD = float(best["thr"])
        print("[CLS] selected CLS_SKIP_THRESHOLD:", CLS_SKIP_THRESHOLD)
        print("[CLS] stats:", best)

        fn_keys = [k for k, yt, p in zip(keys, y_true.tolist(), probs.tolist()) if yt == 1 and p < CLS_SKIP_THRESHOLD]
        print("[CLS] forged false negatives at threshold:", len(fn_keys))
        if fn_keys:
            print("[CLS] FN examples (rel_path):", fn_keys[:20])

# %%
# Célula 7 — Config de treino (patch-based)
from torch.utils.data import DataLoader, WeightedRandomSampler

PATCH_SIZE = int(cfg.get("patch_size", 512)) if cfg else 512
BATCH_SIZE = int(cfg.get("batch_size", 8)) if cfg else 8
NUM_WORKERS = int(cfg.get("num_workers", 2)) if cfg else 2

POSITIVE_PROB = float(cfg.get("positive_prob", 0.7)) if cfg else 0.7
MIN_POS_PIXELS = int(cfg.get("min_pos_pixels", 32)) if cfg else 32
MAX_TRIES = int(cfg.get("max_tries", 10)) if cfg else 10
POS_SAMPLE_WEIGHT = float(cfg.get("pos_sample_weight", 2.0)) if cfg else 2.0

# %% [markdown]
# ## Data Augmentation (Aumento de Dados)
#
# Este baseline usa **Albumentations** para aplicar aumentos **coerentes** entre `image` e `mask` (para geometria),
# e aumentos **apenas na imagem** (para ruído/cor/blur).
#
# Geometria (image + mask):
#
# - **Flips** horizontal/vertical
# - **Rotação 90°** aleatória e **pequenas rotações** (Affine)
# - **Escala/zoom e translação** (Affine + RandomResizedCrop)
#
# Robustez fotométrica (apenas image):
#
# - **Brilho/contraste**, **gamma** e **CLAHE** (leve)
# - **Ruído gaussiano** e **blur** (gauss/motion)
# - **Compressão** (artefatos tipo JPEG) e **cutout**
#
# Copy-move sintético (image + mask):
#
# - Para patches com máscara vazia (amostra autêntica), aplicamos um **copy-move on-the-fly**:
#   copiamos uma região e colamos em outra posição no mesmo patch, marcando **origem e destino** na máscara.
#   Opcionalmente aplicamos pequena rotação/escala no patch colado.
#

# %%
# Célula 7b — Config do augmentation (inclui copy-move sintético)
COPY_MOVE_PROB = float(cfg.get("copy_move_prob", 0.25)) if cfg else 0.25
COPY_MOVE_MIN_AREA_FRAC = float(cfg.get("copy_move_min_area_frac", 0.05)) if cfg else 0.05
COPY_MOVE_MAX_AREA_FRAC = float(cfg.get("copy_move_max_area_frac", 0.20)) if cfg else 0.20
COPY_MOVE_ROTATION_LIMIT = float(cfg.get("copy_move_rotation_limit", 15.0)) if cfg else 15.0

scale_range = cfg.get("copy_move_scale_range", [0.9, 1.1]) if cfg else [0.9, 1.1]
if isinstance(scale_range, (list, tuple)) and len(scale_range) == 2:
    COPY_MOVE_SCALE_RANGE = (float(scale_range[0]), float(scale_range[1]))
else:
    COPY_MOVE_SCALE_RANGE = (0.9, 1.1)

try:
    train_aug = get_train_augment(
        patch_size=PATCH_SIZE,
        copy_move_prob=COPY_MOVE_PROB,
        copy_move_min_area_frac=COPY_MOVE_MIN_AREA_FRAC,
        copy_move_max_area_frac=COPY_MOVE_MAX_AREA_FRAC,
        copy_move_rotation_limit=COPY_MOVE_ROTATION_LIMIT,
        copy_move_scale_range=COPY_MOVE_SCALE_RANGE,
    )
    val_aug = get_val_augment()
except ImportError as exc:
    print("[SEG] albumentations indisponível; desativando augmentations de treino. Erro abaixo:")
    traceback.print_exc()
    train_aug = None
    val_aug = None

def make_loaders(train_samples_fold, val_samples_fold, *, train_aug, val_aug, batch_size: int, patch_size: int):
    train_ds = PatchDataset(
        train_samples_fold,
        patch_size=patch_size,
        train=True,
        augment=train_aug,
        positive_prob=POSITIVE_PROB,
        min_pos_pixels=MIN_POS_PIXELS,
        max_tries=MAX_TRIES,
        seed=SEED,
    )
    val_ds = PatchDataset(
        val_samples_fold,
        patch_size=patch_size,
        train=False,
        augment=val_aug,
        seed=SEED,
    )

    weights = [POS_SAMPLE_WEIGHT if s.is_authentic is False else 1.0 for s in train_samples_fold]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_samples_fold), replacement=True)

    train_loader_fold = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
    )
    val_loader_fold = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
    )
    return train_loader_fold, val_loader_fold

# %% [markdown]
# ### Implementação via `segmentation_models_pytorch` (SMP)
#
# Neste pipeline usamos o **SMP** para instanciar arquiteturas SOTA rapidamente (U-Net++, DeepLabV3+, SegFormer).
#
# **Importante (logits vs sigmoid):** aqui mantemos `activation=None` nos modelos do SMP, e tratamos a saída como
# **logits**:
#
# - Treino: usamos losses com `BCEWithLogitsLoss` (`BCEDiceLoss` / `BCETverskyLoss`), que esperam logits.
# - Inferência: `predict_image()` aplica `torch.sigmoid(logits)` para gerar probabilidades.
#
# Se você configurar `activation="sigmoid"` no SMP, precisa ajustar a loss (sem logits) e remover o `sigmoid` da
# inferência para não aplicar duas vezes.
#
# Sobre encoders:
#
# - `efficientnet-b7`, `se_resnet101` e `mit_b*` estão disponíveis no SMP.
# - `timm-resnest101e` não está disponível como encoder no SMP deste ambiente; por isso usamos **ResNet101+SE**
#   (`se_resnet101`) como substituto no DeepLabV3+.
#

# %%
# Célula 8 — Modelos de segmentação (ensemble)
import torch.nn as nn


def build_fallback_deeplab(pretrained: bool = True) -> nn.Module:
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    base = deeplabv3_resnet50(weights=weights)
    in_ch = int(base.classifier[-1].in_channels)
    base.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)
    base.aux_classifier = None

    class Wrapper(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.model(x)
            if isinstance(out, dict):
                return out["out"]
            return out

    return Wrapper(base)


try:
    import segmentation_models_pytorch as smp
except Exception:
    print("[IMPORT ERROR] segmentation_models_pytorch não importou; não é possível montar os modelos do ensemble.")
    traceback.print_exc()
    # Fallback: tenta importar via código "vendor" em um Kaggle Dataset (ex.: repo importado do GitHub).
    add_local_package_to_syspath("segmentation_models_pytorch")
    add_local_package_to_syspath("timm")  # smp pode depender de timm em alguns encoders
    try:
        import segmentation_models_pytorch as smp  # type: ignore[no-redef]
        print("[IMPORT] segmentation_models_pytorch importado via fallback local.")
    except Exception:
        print("[IMPORT ERROR] falha também no fallback local; não há como seguir sem SMP.")
        traceback.print_exc()
        raise


SEG_MODEL_SPECS = [
    # Encoder-decoder CNN (detalhes finos) + dropout no decoder.
    {
        "id": "unetpp_effb7",
        "arch": "unetpp",
        "encoder_name": "efficientnet-b7",
        "encoder_weights": "imagenet",
        "decoder_attention_type": "scse",
        "decoder_dropout": 0.20,
        "batch_size": 1,
    },
    # DeepLabV3+ (ASPP / contexto multi-escala) para complementar o U-Net++.
    # ResNeSt não está disponível como encoder neste setup; usamos ResNet101+SE como alternativa.
    {
        "id": "deeplabv3p_se_r101",
        "arch": "deeplabv3p",
        "encoder_name": "se_resnet101",
        "encoder_weights": "imagenet",
        "encoder_output_stride": 16,
        "decoder_channels": 256,
        "decoder_atrous_rates": [12, 24, 36],
        "decoder_aspp_separable": False,
        "decoder_aspp_dropout": 0.10,
        "decoder_dropout": 0.10,
        "batch_size": 1,
    },
    # Transformer-style (autoatenção) para complementar as CNNs.
    # Nota: B5 é bem mais custoso; B3 costuma ser um bom trade-off.
    {
        "id": "segformer_mitb3",
        "arch": "segformer",
        "encoder_name": "mit_b3",
        "encoder_weights": "imagenet",
        "decoder_segmentation_channels": 256,
        "decoder_dropout": 0.10,
        "batch_size": 1,
    },
]

SEG_SAVE_DIR = OUTPUT_ROOT / "models_seg"
SEG_LOAD_DIR = _find_checkpoint_root(SEG_SAVE_DIR, expected_subdir="models_seg")

print("segmentation checkpoints (load):", SEG_LOAD_DIR)
print("segmentation checkpoints (save):", SEG_SAVE_DIR)


def _wrap_segmentation_head_dropout(model: nn.Module, p: float) -> nn.Module:
    p = float(p)
    if p <= 0:
        return model
    if hasattr(model, "segmentation_head"):
        model.segmentation_head = nn.Sequential(nn.Dropout2d(p=p), model.segmentation_head)
    return model


def _inject_unetpp_decoder_dropout(model: nn.Module, p: float) -> nn.Module:
    p = float(p)
    if p <= 0:
        return model
    decoder = getattr(model, "decoder", None)
    blocks = getattr(decoder, "blocks", None)
    if blocks is None or not hasattr(blocks, "items"):
        return model

    for _, block in blocks.items():
        for attr in ("conv1", "conv2"):
            layer = getattr(block, attr, None)
            if layer is None:
                continue
            if isinstance(layer, nn.Sequential) and len(layer) > 0 and isinstance(layer[-1], nn.Dropout2d):
                continue
            setattr(block, attr, nn.Sequential(layer, nn.Dropout2d(p=p)))
    return model


def _safe_build_smp(builder, **kwargs):
    if smp is None:
        raise ImportError("segmentation_models_pytorch is required for SMP models")
    try:
        return builder(**kwargs)
    except Exception as exc:
        if kwargs.get("encoder_weights"):
            print(
                f"[SEG] encoder_weights falharam ({kwargs.get('encoder_name')}); seguindo sem pesos. Erro abaixo:",
            )
            traceback.print_exc()
            kwargs["encoder_weights"] = None
            return builder(**kwargs)
        raise


def build_seg_model(spec: dict, pretrained: bool = True) -> nn.Module:
    arch = str(spec.get("arch"))
    encoder_name = str(spec.get("encoder_name"))
    encoder_weights = spec.get("encoder_weights", "imagenet") if pretrained else None
    if encoder_weights == "":
        encoder_weights = None

    if arch == "unetpp":
        model = _safe_build_smp(
            smp.UnetPlusPlus,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_attention_type=spec.get("decoder_attention_type", "scse"),
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch == "segformer":
        model = _safe_build_smp(
            smp.Segformer,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_segmentation_channels=int(spec.get("decoder_segmentation_channels", 256)),
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch == "deeplabv3p":
        model = _safe_build_smp(
            smp.DeepLabV3Plus,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_output_stride=int(spec.get("encoder_output_stride", 16)),
            decoder_channels=int(spec.get("decoder_channels", 256)),
            decoder_atrous_rates=tuple(spec.get("decoder_atrous_rates", (12, 24, 36))),
            decoder_aspp_separable=bool(spec.get("decoder_aspp_separable", False)),
            decoder_aspp_dropout=float(spec.get("decoder_aspp_dropout", 0.0)),
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch == "deeplabv3":
        model = build_fallback_deeplab(pretrained=bool(pretrained))
    else:
        raise ValueError(f"Unknown seg arch: {arch}")

    p = float(spec.get("decoder_dropout", 0.0))
    if arch == "unetpp":
        model = _inject_unetpp_decoder_dropout(model, p)
    else:
        model = _wrap_segmentation_head_dropout(model, p)
    return model


print("segmentation model builders ready")

# %%
# Célula 9 — Loss, otimizador e loop de treino (opcional)

LOSS_NAME = (cfg.get("loss", "bce_dice") if cfg else "bce_dice").lower()
LR = float(cfg.get("learning_rate", 1e-4)) if cfg else 1e-4
WEIGHT_DECAY = float(cfg.get("weight_decay", 1e-4)) if cfg else 1e-4
EPOCHS = int(cfg.get("epochs", 30)) if cfg else 30
USE_AMP = bool(cfg.get("use_amp", True)) if cfg else True
USE_AMP = USE_AMP and DEVICE.startswith("cuda")

RUN_TRAIN = False  # mude para True para treinar aqui
TRAIN_FOLDS = [FOLD]  # para CV, use: list(range(N_FOLDS))
TRAIN_SEG_MODEL_IDS = [s["id"] for s in SEG_MODEL_SPECS]  # edite para treinar só um dos modelos

SEG_LR_ENCODER = float(cfg.get("learning_rate_encoder", LR)) if cfg else LR
SEG_LR_DECODER = float(cfg.get("learning_rate_decoder", 1e-3)) if cfg else 1e-3
SEG_FREEZE_EPOCHS = int(cfg.get("freeze_encoder_epochs", 3)) if cfg else 3

SEG_SCHEDULER = str(cfg.get("seg_scheduler", "cosine")) if cfg else "cosine"  # "cosine" | "plateau" | "none"
SEG_SCHEDULER = SEG_SCHEDULER.lower().strip()
SEG_PLATEAU_FACTOR = float(cfg.get("seg_plateau_factor", 0.5)) if cfg else 0.5
SEG_PLATEAU_PATIENCE = int(cfg.get("seg_plateau_patience", 2)) if cfg else 2

SEG_EARLY_STOPPING = bool(cfg.get("seg_early_stopping", True)) if cfg else True
SEG_PATIENCE = int(cfg.get("seg_patience", 5)) if cfg else 5
SEG_MIN_DELTA = float(cfg.get("seg_min_delta", 1e-4)) if cfg else 1e-4

# Treino em estágios (opcional): começar com patches menores e depois refinar no tamanho final.
SEG_STAGED_TRAINING = False
SEG_TRAIN_STAGES = [{"patch_size": PATCH_SIZE, "epochs": EPOCHS}]
# Exemplo (mais rápido no começo, mais detalhado no fim):
# SEG_STAGED_TRAINING = True
# SEG_TRAIN_STAGES = [{"patch_size": 256, "epochs": 5}, {"patch_size": 512, "epochs": 25}]

SEG_BCE_WEIGHT = float(cfg.get("bce_weight", 1.0)) if cfg else 1.0
SEG_DICE_WEIGHT = float(cfg.get("dice_weight", 1.0)) if cfg else 1.0
SEG_DICE_SMOOTH = float(cfg.get("dice_smooth", 1.0)) if cfg else 1.0
SEG_BCE_POS_WEIGHT = cfg.get("bce_pos_weight", None) if cfg else None  # opcional (float)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class WeightedBCEDiceLoss(nn.Module):
    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
        bce_pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = DiceLoss(smooth=float(dice_smooth))
        self.bce_pos_weight = float(bce_pos_weight) if bce_pos_weight is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        if self.bce_pos_weight is None:
            bce = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            pos_weight = torch.tensor([self.bce_pos_weight], device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


def _split_encoder_params(model: nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return [], list(model.parameters())
    enc_params = list(encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]
    return enc_params, other_params


def _set_trainable(params: list[torch.nn.Parameter], trainable: bool) -> None:
    for p in params:
        p.requires_grad = bool(trainable)


def _build_optimizer(model: nn.Module) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    enc_params, other_params = _split_encoder_params(model)
    param_groups = []
    if enc_params:
        param_groups.append({"params": enc_params, "lr": SEG_LR_ENCODER})
    if other_params:
        param_groups.append({"params": other_params, "lr": SEG_LR_DECODER})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    return optimizer, enc_params, other_params


def _build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int):
    if SEG_SCHEDULER == "none":
        return None
    if SEG_SCHEDULER == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(SEG_PLATEAU_FACTOR),
            patience=int(SEG_PLATEAU_PATIENCE),
            verbose=True,
        )
    if SEG_SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(max(total_epochs, 1)))
    raise ValueError(f"Unknown SEG_SCHEDULER: {SEG_SCHEDULER}")


def _build_seg_augs(patch_size: int):
    try:
        train_aug = get_train_augment(
            patch_size=int(patch_size),
            copy_move_prob=COPY_MOVE_PROB,
            copy_move_min_area_frac=COPY_MOVE_MIN_AREA_FRAC,
            copy_move_max_area_frac=COPY_MOVE_MAX_AREA_FRAC,
            copy_move_rotation_limit=COPY_MOVE_ROTATION_LIMIT,
            copy_move_scale_range=COPY_MOVE_SCALE_RANGE,
        )
        val_aug = get_val_augment()
    except Exception as exc:
        print("[SEG] erro ao construir augmentations; treinando sem augs. Erro abaixo:")
        traceback.print_exc()
        train_aug = None
        val_aug = None
    return train_aug, val_aug


if RUN_TRAIN:
    for spec in SEG_MODEL_SPECS:
        if spec["id"] not in set(TRAIN_SEG_MODEL_IDS):
            continue
        for fold_id in TRAIN_FOLDS:
            train_idx, val_idx = folds[int(fold_id)]
            train_samples_fold = [train_samples[int(i)] for i in train_idx]
            val_samples_fold = [train_samples[int(i)] for i in val_idx]

            fold_dir = SEG_SAVE_DIR / str(spec["id"]) / f"fold_{int(fold_id)}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = fold_dir / "best.pt"

            model = build_seg_model(spec, pretrained=True).to(DEVICE)
            if LOSS_NAME == "bce_tversky":
                criterion = BCETverskyLoss(
                    alpha=float(cfg.get("tversky_alpha", 0.7)) if cfg else 0.7,
                    beta=float(cfg.get("tversky_beta", 0.3)) if cfg else 0.3,
                    tversky_weight=float(cfg.get("tversky_weight", 1.0)) if cfg else 1.0,
                )
            else:
                pos_weight = float(SEG_BCE_POS_WEIGHT) if isinstance(SEG_BCE_POS_WEIGHT, (int, float)) else None
                criterion = WeightedBCEDiceLoss(
                    bce_weight=SEG_BCE_WEIGHT,
                    dice_weight=SEG_DICE_WEIGHT,
                    dice_smooth=SEG_DICE_SMOOTH,
                    bce_pos_weight=pos_weight,
                )

            optimizer, enc_params, _ = _build_optimizer(model)

            # Fase 1: (opcional) congelar encoder por algumas épocas.
            if enc_params and SEG_FREEZE_EPOCHS > 0:
                _set_trainable(enc_params, False)
                print(f"[SEG:{spec['id']}] freeze encoder for {SEG_FREEZE_EPOCHS} epoch(s)")

            best_dice = -1.0
            best_loss = float("inf")
            bad_epochs = 0
            global_epoch = 0

            stages = SEG_TRAIN_STAGES if SEG_STAGED_TRAINING else [{"patch_size": PATCH_SIZE, "epochs": EPOCHS}]
            total_epochs = int(sum(int(s["epochs"]) for s in stages))
            scheduler = _build_scheduler(optimizer, total_epochs=total_epochs)

            for stage in stages:
                patch_size = int(stage["patch_size"])
                stage_epochs = int(stage["epochs"])
                stage_train_aug, stage_val_aug = _build_seg_augs(patch_size)

                batch_size = int(spec.get("batch_size", BATCH_SIZE))
                train_loader_fold, val_loader_fold = make_loaders(
                    train_samples_fold,
                    val_samples_fold,
                    train_aug=stage_train_aug,
                    val_aug=stage_val_aug,
                    batch_size=batch_size,
                    patch_size=patch_size,
                )

                for _ in range(stage_epochs):
                    global_epoch += 1

                    if enc_params and SEG_FREEZE_EPOCHS > 0 and global_epoch == (SEG_FREEZE_EPOCHS + 1):
                        _set_trainable(enc_params, True)
                        print(f"[SEG:{spec['id']}] unfreeze encoder at epoch {global_epoch}")

                    train_stats = train_one_epoch(
                        model,
                        train_loader_fold,
                        criterion,
                        optimizer,
                        DEVICE,
                        use_amp=USE_AMP,
                    )
                    val_stats, val_dice = validate(model, val_loader_fold, criterion, DEVICE)

                    lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]
                    print(
                        f"[SEG:{spec['id']}] fold {int(fold_id)} epoch {global_epoch:02d}/{total_epochs} "
                        f"patch={patch_size} lr={lrs} "
                        f"train_loss={train_stats.loss:.4f} val_loss={val_stats.loss:.4f} val_dice={val_dice:.4f}"
                    )

                    if scheduler is not None:
                        if SEG_SCHEDULER == "plateau":
                            scheduler.step(val_stats.loss)
                        else:
                            scheduler.step()

                    val_dice_f = float(val_dice)
                    val_loss_f = float(val_stats.loss)

                    improved_dice = val_dice_f > best_dice + SEG_MIN_DELTA
                    improved_tie = (abs(val_dice_f - best_dice) <= SEG_MIN_DELTA) and (
                        val_loss_f < best_loss - SEG_MIN_DELTA
                    )

                    if improved_dice or improved_tie:
                        if improved_dice:
                            best_dice = val_dice_f
                        best_loss = val_loss_f
                        bad_epochs = 0
                        torch.save(
                            {
                                "model_state": model.state_dict(),
                                "epoch": global_epoch,
                                "val_loss": val_loss_f,
                                "val_dice": val_dice_f,
                                "patch_size": patch_size,
                                "spec": spec,
                                "config": cfg,
                            },
                            ckpt_path,
                        )
                    else:
                        bad_epochs += 1

                    if SEG_EARLY_STOPPING and bad_epochs >= SEG_PATIENCE:
                        print(f"[SEG:{spec['id']}] early stopping: sem melhora por {SEG_PATIENCE} épocas")
                        break

                if SEG_EARLY_STOPPING and bad_epochs >= SEG_PATIENCE:
                    break

            print("[SEG] saved:", ckpt_path)

            del model, optimizer
            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()

print("segmentation checkpoints dir (save):", SEG_SAVE_DIR)

# %%
# Célula 10 — Carregar checkpoint (necessário para inferência/submissão)

def _load_checkpoint(path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("config", {})
    return ckpt, {}


def _seg_ckpt_path(model_id: str, fold: int) -> Path:
    return (SEG_LOAD_DIR / model_id / f"fold_{int(fold)}") / "best.pt"


SEG_USE_ENSEMBLE = True
SEG_INFER_FOLDS = list(range(N_FOLDS)) if SEG_USE_ENSEMBLE else [FOLD]
SEG_INFER_MODEL_IDS = [s["id"] for s in SEG_MODEL_SPECS]

models_by_id: dict[str, list[nn.Module]] = {}
loaded_models: list[str] = []
missing_models: list[str] = []
loaded_folds_set: set[int] = set()

for spec in SEG_MODEL_SPECS:
    model_id = str(spec["id"])
    if model_id not in set(SEG_INFER_MODEL_IDS):
        continue
    models_by_id.setdefault(model_id, [])
    for fold_id in SEG_INFER_FOLDS:
        fold_ckpt = _seg_ckpt_path(model_id, int(fold_id))
        tag = f"{model_id}/fold_{int(fold_id)}"
        if not fold_ckpt.exists():
            missing_models.append(tag)
            continue

        m = build_seg_model(spec, pretrained=False)
        state, _ = _load_checkpoint(fold_ckpt)
        m.load_state_dict(state)
        m.to(DEVICE)
        m.eval()
        models_by_id[model_id].append(m)
        loaded_models.append(tag)
        loaded_folds_set.add(int(fold_id))

loaded_folds = sorted(loaded_folds_set)
print("loaded seg models:", loaded_models)
if missing_models:
    print("missing seg models:", missing_models[:20], ("..." if len(missing_models) > 20 else ""))
if not loaded_models:
    print(f"[SEG] nenhum checkpoint encontrado em: {SEG_LOAD_DIR}")
    print("[SEG] opções:")
    print("- Treinar agora: defina `RUN_TRAIN=True` (célula 9) e rode o treino (vai salvar em `SEG_SAVE_DIR`).")
    print("- Usar pesos prontos: coloque `outputs/models_seg/<model_id>/fold_*/best.pt` em um dataset do Kaggle;")
    print("  o notebook tenta localizar automaticamente em `/kaggle/input/*/outputs/models_seg`.")

# %%
# Célula 10b — (Opcional) Carregar classificador para gating na inferência
CLS_GATE = True
CLS_SKIP_THRESHOLD_INFER = float(CLS_SKIP_THRESHOLD)
CLS_USE_ENSEMBLE = False
CLS_INFER_FOLDS = loaded_folds if CLS_USE_ENSEMBLE else [FOLD]

cls_models: list[torch.nn.Module] = []
cls_loaded_folds: list[int] = []
CLS_INFER_IMAGE_SIZE = CLS_IMAGE_SIZE

if CLS_GATE and timm is None:
    print("[CLS] timm indisponível; gating desativado.")
    CLS_GATE = False

if CLS_GATE:
    for fold_id in CLS_INFER_FOLDS:
        cls_ckpt_path = (CLS_LOAD_DIR / f"fold_{int(fold_id)}") / "best.pt"
        if not cls_ckpt_path.exists():
            continue

        state, cfg_cls = _load_checkpoint(cls_ckpt_path)
        model_name = cfg_cls.get("model_name", CLS_MODEL_NAME)
        CLS_INFER_IMAGE_SIZE = int(cfg_cls.get("image_size", CLS_INFER_IMAGE_SIZE))

        m_cls = build_cls_model(model_name, pretrained=False)
        m_cls.load_state_dict(state)
        m_cls.to(DEVICE)
        m_cls.eval()
        cls_models.append(m_cls)
        cls_loaded_folds.append(int(fold_id))

    print("[CLS] loaded folds:", cls_loaded_folds)
    if not cls_models:
        print(f"[CLS] nenhum checkpoint encontrado em {CLS_LOAD_DIR}/fold_*/best.pt; gating desativado.")
        CLS_GATE = False


@torch.no_grad()
def predict_prob_forged(image: np.ndarray) -> float:
    import torch.nn.functional as F

    if not cls_models:
        raise RuntimeError("Classifier models not loaded")

    img = normalize_image(image)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    if CLS_INFER_IMAGE_SIZE and x.shape[-2:] != (CLS_INFER_IMAGE_SIZE, CLS_INFER_IMAGE_SIZE):
        x = F.interpolate(x, size=(CLS_INFER_IMAGE_SIZE, CLS_INFER_IMAGE_SIZE), mode="bilinear", align_corners=False)

    probs = []
    for m in cls_models:
        logits = m(x).view(-1)
        probs.append(float(torch.sigmoid(logits)[0].item()))
    return float(np.mean(probs))

# %%
# Célula 11 — Pós-processamento (componentes conexos)
_cc_backend = "scipy" if _cc_label is not None else ("opencv" if cv2 is not None else "none")


def extract_components_safe(mask: np.ndarray, min_area: int = 0) -> list[np.ndarray]:
    return extract_components(mask, min_area=min_area)


print("connected components backend:", _cc_backend)

# %%
# Célula 12 — Validação em imagem inteira (oF1) + tuning simples de threshold (opcional)

RUN_VAL_FULL = False  # mude para True para validar com oF1 em imagem inteira

TILE_SIZE = 1024
OVERLAP = 128
MAX_SIZE = 0  # se quiser, defina ex.: 2048 para reduzir custo

THRESHOLD = 0.5
MIN_AREA = 32
VAL_LIMIT = 200  # limite para não ficar gigante


SEG_USE_TTA = True
# Flips simples costumam dar bom ganho/custo no Kaggle.
# Obs: usamos `.copy()` ao flipar a imagem para evitar strides negativos (torch.from_numpy não aceita).
SEG_TTA_MODES = ("none", "hflip", "vflip")

# Pesos do ensemble por arquitetura (média por-fold dentro de cada `model_id`, depois média ponderada entre ids).
# Comece com pesos iguais; ajuste via validação se quiser.
SEG_MODEL_WEIGHTS = {str(s["id"]): 1.0 for s in SEG_MODEL_SPECS}
# Exemplo:
# SEG_MODEL_WEIGHTS.update({"unetpp_effb7": 0.4, "deeplabv3p_se_r101": 0.4, "segformer_mitb3": 0.2})

# Pós-processamento (binário)
PP_FILL_SMALL_HOLES = True
PP_MAX_HOLE_AREA = 64  # só preenche "buracos" bem pequenos (em pixels)
PP_MORPH_CLOSE_KERNEL = 0  # 0 desativa; ex.: 3 para suavizar/fechar micro falhas
PP_MORPH_OPEN_KERNEL = 0  # 0 desativa; cuidado: pode apagar regiões pequenas reais
PP_MORPH_ITERS = 1


def _tta_apply_image(image: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode)
    if mode == "none":
        return image
    if mode == "hflip":
        return image[:, ::-1, :].copy()
    if mode == "vflip":
        return image[::-1, :, :].copy()
    if mode in {"hvflip", "vhflip"}:
        return image[::-1, ::-1, :].copy()
    raise ValueError(f"Unknown TTA mode: {mode}")


def _tta_invert_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode)
    if mode == "none":
        return mask
    if mode == "hflip":
        return mask[:, ::-1]
    if mode == "vflip":
        return mask[::-1, :]
    if mode in {"hvflip", "vhflip"}:
        return mask[::-1, ::-1]
    raise ValueError(f"Unknown TTA mode: {mode}")


def _predict_prob_single_model(model: nn.Module, image: np.ndarray) -> np.ndarray:
    if not SEG_USE_TTA or not SEG_TTA_MODES:
        return predict_image(model, image, DEVICE, tile_size=TILE_SIZE, overlap=OVERLAP, max_size=MAX_SIZE)

    prob_sum: np.ndarray | None = None
    for mode in SEG_TTA_MODES:
        img_t = _tta_apply_image(image, mode)
        prob = predict_image(model, img_t, DEVICE, tile_size=TILE_SIZE, overlap=OVERLAP, max_size=MAX_SIZE)
        prob = _tta_invert_mask(prob, mode)
        if prob_sum is None:
            prob_sum = prob.astype(np.float32, copy=False)
        else:
            prob_sum += prob.astype(np.float32, copy=False)
    return prob_sum / np.float32(len(SEG_TTA_MODES))


def _fill_small_holes_cv2(mask: np.ndarray, max_area: int) -> np.ndarray:
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if max_area <= 0 or m.max() == 0:
        return m

    inv = (1 - m).astype(np.uint8)  # 1 = fundo/buracos
    if inv.max() == 0:
        return m

    # Flood-fill no "fundo externo" para sobrar apenas buracos internos.
    inv_pad = np.pad(inv, 1, mode="constant", constant_values=1)
    flood = inv_pad.copy()
    ff_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=0)

    holes = (flood == 1).astype(np.uint8)[1:-1, 1:-1]
    if holes.max() == 0:
        return m

    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    for idx in range(1, int(n)):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= int(max_area):
            m[labels == idx] = 1
    return m


def _morphology_cv2(mask: np.ndarray, close_kernel: int, open_kernel: int, iters: int) -> np.ndarray:
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    iters = int(max(iters, 1))

    if int(close_kernel) and int(close_kernel) > 1:
        k = int(close_kernel)
        kernel = np.ones((k, k), dtype=np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=iters)

    if int(open_kernel) and int(open_kernel) > 1:
        k = int(open_kernel)
        kernel = np.ones((k, k), dtype=np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=iters)

    return (m > 0).astype(np.uint8)


def postprocess_binary_mask(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.max() == 0:
        return m

    try:
        if PP_FILL_SMALL_HOLES:
            m = _fill_small_holes_cv2(m, max_area=int(PP_MAX_HOLE_AREA))
        if (int(PP_MORPH_CLOSE_KERNEL) and int(PP_MORPH_CLOSE_KERNEL) > 1) or (
            int(PP_MORPH_OPEN_KERNEL) and int(PP_MORPH_OPEN_KERNEL) > 1
        ):
            m = _morphology_cv2(
                m,
                close_kernel=int(PP_MORPH_CLOSE_KERNEL),
                open_kernel=int(PP_MORPH_OPEN_KERNEL),
                iters=int(PP_MORPH_ITERS),
            )
    except Exception as exc:
        print("[PP] erro no pós-processamento; seguindo sem. Erro abaixo:")
        traceback.print_exc()
    return (m > 0).astype(np.uint8)


def predict_prob_ensemble(image: np.ndarray) -> np.ndarray:
    if not models_by_id or not any(models_by_id.values()):
        raise RuntimeError(
            f"Sem modelos carregados (nenhum checkpoint encontrado em {SEG_LOAD_DIR}/<model_id>/fold_*/best.pt)."
        )

    total: np.ndarray | None = None
    weight_sum = 0.0

    for model_id, models in models_by_id.items():
        if not models:
            continue

        # Média dentro do model_id (ex.: folds)
        prob_id_sum: np.ndarray | None = None
        for m in models:
            prob = _predict_prob_single_model(m, image)
            if prob_id_sum is None:
                prob_id_sum = prob.astype(np.float32, copy=False)
            else:
                prob_id_sum += prob.astype(np.float32, copy=False)
        prob_id = prob_id_sum / np.float32(len(models))

        w = np.float32(SEG_MODEL_WEIGHTS.get(str(model_id), 1.0))
        if float(w) <= 0.0:
            continue

        if total is None:
            total = prob_id * w
        else:
            total += prob_id * w
        weight_sum += float(w)

    if total is None or weight_sum <= 0:
        raise RuntimeError("Ensemble inválido: nenhum modelo com peso > 0 foi usado.")

    return total / np.float32(max(weight_sum, 1e-8))


def predict_instances_for_image(image: np.ndarray, threshold: float, min_area: int) -> list[np.ndarray]:
    prob = predict_prob_ensemble(image)
    bin_mask = binarize(prob, threshold=threshold)
    bin_mask = postprocess_binary_mask(bin_mask)
    return extract_components_safe(bin_mask, min_area=min_area)


def predict_instances_for_sample(sample, threshold: float, min_area: int) -> list[np.ndarray]:
    image = load_image(sample.image_path)
    return predict_instances_for_image(image, threshold=threshold, min_area=min_area)


def mean_of1(samples, threshold: float, min_area: int, limit: int = 0) -> float | None:
    scores: list[float] = []
    for i, sample in enumerate(samples):
        if limit and i >= limit:
            break
        gt_instances = load_mask_instances(sample.mask_path) if sample.mask_path else []
        pred_instances = predict_instances_for_sample(sample, threshold=threshold, min_area=min_area)
        try:
            scores.append(float(score_image(gt_instances, pred_instances)))
        except Exception as exc:
            print("[METRIC] erro ao calcular métrica (provável falta de SciPy). Erro abaixo:")
            traceback.print_exc()
            return None
    if not scores:
        return None
    return float(np.mean(scores))


if RUN_VAL_FULL and any(models_by_id.values()):
    score = mean_of1(val_fold_samples, threshold=THRESHOLD, min_area=MIN_AREA, limit=VAL_LIMIT)
    print("val mean oF1:", score)

    # grid simples (faixa curta; ajuste conforme custo)
    thresholds = [0.3, 0.4, 0.5, 0.6]
    min_areas = [0, 16, 32, 64, 128]
    best = (None, None, -1.0)
    for t in thresholds:
        for a in min_areas:
            s = mean_of1(val_fold_samples, threshold=t, min_area=a, limit=VAL_LIMIT)
            if s is None:
                break
            print(f"t={t:.2f} a={a:4d} -> {s:.5f}")
            if s > best[2]:
                best = (t, a, s)
    print("best:", best)

# %%
# Célula 13 — Gerar submission.csv (test)

SUBMISSION_PATH = Path("/kaggle/working/submission.csv") if is_kaggle() else (OUTPUT_ROOT / "submission.csv")

RUN_SUBMISSION = True  # deixe True no Kaggle submit

if RUN_SUBMISSION:
    if not any(models_by_id.values()):
        print(f"Sem checkpoint(s) para inferência. Treine e salve em {SEG_SAVE_DIR}/<model_id>/fold_*/best.pt.")
    else:
        SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SUBMISSION_PATH.open("w", newline="") as f:
            # Formato oficial do `annotation` nesta competição:
            # - "authentic" OU
            # - 1+ instâncias em RLE (coluna-major/F-order, 1-based), cada uma serializada via `json.dumps([...])`,
            #   concatenadas por ';'. `csv.DictWriter` cuida das aspas automaticamente (há vírgulas/colchetes).
            writer = csv.DictWriter(f, fieldnames=["case_id", "annotation"])
            writer.writeheader()
            for sample in test_samples:
                image = load_image(sample.image_path)
                if CLS_GATE:
                    prob_forged = predict_prob_forged(image)
                    if prob_forged < CLS_SKIP_THRESHOLD_INFER:
                        writer.writerow({"case_id": sample.case_id, "annotation": "authentic"})
                        continue

                instances = predict_instances_for_image(image, threshold=THRESHOLD, min_area=MIN_AREA)
                writer.writerow({"case_id": sample.case_id, "annotation": encode_instances(instances)})

        print("wrote:", SUBMISSION_PATH)

# %%
# Célula 14 — Preview do CSV
try:
    import pandas as pd

    from IPython.display import display

    if not SUBMISSION_PATH.exists():
        print("submission ainda não foi gerada:", SUBMISSION_PATH)
    else:
        df_sub = pd.read_csv(SUBMISSION_PATH)
        display(df_sub.head())
        print("rows:", len(df_sub))
except Exception as exc:
    print("[PREVIEW] erro ao mostrar preview do CSV. Erro abaixo:")
    traceback.print_exc()
