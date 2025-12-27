# %% [markdown]
# # Fase 1 — Setup offline + checagem (Kaggle)
#
# Este notebook existe para **rodar com internet OFF** no Kaggle e deixar claro:
# - quais pacotes estão disponíveis,
# - se existe um **bundle offline** (`wheels/*.whl`) anexado,
# - e se o dataset da competição está montado corretamente.
#
# **Sem “código do projeto”**: tudo aqui é auto-contido no notebook.
#
# ---

# %%
# Célula 1 — Sanidade Kaggle (lembrete)
print("Kaggle submission constraints (lembrete):")
print("- Submissions via Notebook")
print("- Runtime <= 4h (CPU/GPU)")
print("- Internet: OFF no submit")
print("- Output: submission.csv ou submission.parquet")

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
#
# Se você anexar um Dataset que contenha `wheels/*.whl`, esta célula instala os pacotes **offline** via pip.
# Estruturas suportadas:
# - `/kaggle/input/<dataset>/wheels/*.whl`
# - `/kaggle/input/<dataset>/recodai_bundle/wheels/*.whl`
#
# Se nada for encontrado, nada é instalado (e os imports abaixo vão mostrar o erro explicitamente).
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

    Útil quando você importa um repositório GitHub como Dataset do Kaggle contendo libs "vendorizadas".
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
# Célula 3 — Checagem de dependências (mostra tudo, não esconde erro)


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

print("\nDica:")
print("- Se faltar `timm` / `segmentation_models_pytorch`, anexe um Dataset com `wheels/*.whl` e rode a Célula 2b.")
print("- Se uma lib for puro-Python e não tiver wheel, você pode vendorizá-la em um Dataset GitHub e usar `add_local_package_to_syspath()`.")

# %%
# Célula 3b — Import do projeto (src/forgeryseg)
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

# %%
# Célula 4 — Detectar dataset (Kaggle/local) e imprimir contagens


def find_dataset_root() -> Path:
    # Kaggle: padrão da competição
    if is_kaggle():
        base = Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection")
        if base.exists():
            return base
        # fallback: procurar algo que tenha a estrutura do dataset
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for ds in sorted(kaggle_input.glob("*")):
                if (ds / "train_images").exists() and (ds / "test_images").exists():
                    return ds

    # Local: respeita layout do repo (data/)
    base = Path("data").resolve()
    if (base / "train_images").exists() and (base / "test_images").exists():
        return base

    raise FileNotFoundError(
        "Não encontrei o dataset. No Kaggle, anexe o dataset da competição. Localmente, espere `data/train_images` etc."
    )


DATA_ROOT = find_dataset_root()
TRAIN_IMAGES = DATA_ROOT / "train_images"
TRAIN_MASKS = DATA_ROOT / "train_masks"
TEST_IMAGES = DATA_ROOT / "test_images"

print("DATA_ROOT:", DATA_ROOT)
print("train_images/authentic:", len(list((TRAIN_IMAGES / "authentic").glob("*.png"))))
print("train_images/forged:", len(list((TRAIN_IMAGES / "forged").glob("*.png"))))
print("train_masks:", len(list(TRAIN_MASKS.glob("*.npy"))))
print("test_images:", len(list(TEST_IMAGES.glob('*.png'))))
