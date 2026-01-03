# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.14.2
# ---

# %% [markdown]
# # Fase 01 — Pré-treinamento (Kaggle)
#
# Objetivo: **treinar (ou fine-tunar)** os modelos (principalmente **segmentação**),
# salvar os checkpoints em `/kaggle/working/outputs` e gerar um `.zip` para você
# **baixar** e anexar depois no notebook de submissão (internet OFF).
#
# Recomendação (workflow Kaggle):
#
# 1) Rode este notebook com **Internet ON** (**obrigatório**) para treinar/baixar pesos pretrained.
# 2) Baixe `outputs_pretrain.zip` gerado em `/kaggle/working/`.
# 3) Crie um **Kaggle Dataset** contendo a pasta `outputs/` (com `models_seg/` e `models_cls/`).
# 4) No notebook de submissão (internet OFF), anexe esse Dataset (ele será detectado automaticamente).
#
# Variáveis de ambiente úteis (opcionais):
#
# - `FORGERYSEG_DATA_ROOT`: path do dataset (`.../recodai`).
# - `FORGERYSEG_REPO_ROOT`: path do repo (ex.: `/kaggle/input/<ds>/recodai_bundle`).
# - `FORGERYSEG_N_FOLDS`: default `5`.
# - `FORGERYSEG_FOLD`: `0` (um fold) ou `-1` (todos).
# - `FORGERYSEG_PROFILE`: `quick|sweep|full` (controla budget/épocas padrão).
# - `FORGERYSEG_SEG_LEVEL`: `base|plus|max` (quantidade de variações de segmentação).
# - `FORGERYSEG_SEG_FILTER`: filtra nomes (ex.: `convnext,segformer`).
# - `FORGERYSEG_CLS_ACTIVE`: escolhe 1 classificador (evita sobrescrita em `models_cls/`).
#
# Este notebook assume internet ligada sempre; não há modo offline aqui.

# %%
import json
import os
import subprocess
import sys
from pathlib import Path


# %%
# Helpers de ambiente


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    if value == "":
        return str(default)
    return str(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name, "").strip()
    if value == "":
        return bool(default)
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = env_str(name, "").strip()
    if value == "":
        return int(default)
    return int(value)


def env_path(name: str) -> Path | None:
    value = env_str(name, "").strip()
    if not value:
        return None
    return Path(value)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    cmd_str = " ".join(str(c) for c in cmd)
    print("[cmd]", cmd_str)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    tail: list[str] = []
    tail_limit = 200
    for line in proc.stdout:
        print(line, end="")
        tail.append(line)
        if len(tail) > tail_limit:
            tail = tail[-tail_limit:]

    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output="".join(tail))


def find_repo_root() -> Path | None:
    explicit = env_path("FORGERYSEG_REPO_ROOT")
    if explicit is not None:
        return explicit if explicit.exists() else None

    here = Path(".").resolve()
    candidates = [here] + list(here.parents)
    for cand in candidates:
        if (cand / "src" / "forgeryseg" / "__init__.py").exists() and (cand / "scripts").exists():
            return cand

    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                for base in (ds, ds / "recodai_bundle"):
                    if (base / "src" / "forgeryseg" / "__init__.py").exists():
                        return base
    return None


def find_data_root() -> Path | None:
    explicit = env_path("FORGERYSEG_DATA_ROOT")
    if explicit is not None:
        return explicit if (explicit / "train_images").exists() else None

    candidates = [
        Path("data/recodai"),
        Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection/recodai"),
        Path("/kaggle/input/recodai-luc-scientific-image-forgery-detection"),
    ]
    for cand in candidates:
        if (cand / "train_images").exists():
            return cand

    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                if (ds / "train_images").exists():
                    return ds
                if (ds / "recodai" / "train_images").exists():
                    return ds / "recodai"
    return None


def find_requirements_file(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "requirements.txt",
        repo_root / "recodai_bundle" / "requirements.txt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def find_wheels_root(repo_root: Path) -> Path | None:
    explicit = env_path("FORGERYSEG_WHEELS_ROOT")
    if explicit is not None:
        return explicit if explicit.exists() else None

    candidates = [
        repo_root / "recodai_bundle" / "wheels",
        repo_root / "wheels",
    ]
    for cand in candidates:
        if cand.exists() and any(cand.glob("*.whl")):
            return cand

    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                for cand in (ds / "wheels", ds / "recodai_bundle" / "wheels"):
                    if cand.exists() and any(cand.glob("*.whl")):
                        return cand
    return None


def _missing_modules(mod_names: list[str]) -> list[str]:
    missing: list[str] = []
    for name in mod_names:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    return missing


def maybe_install_from_wheels(wheels_root: Path) -> None:
    module_to_pip = {
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "timm": "timm",
        "albumentations": "albumentations",
        "huggingface_hub": "huggingface-hub",
        "safetensors": "safetensors",
        "tqdm": "tqdm",
        "sklearn": "scikit-learn",
    }
    wanted_modules = list(module_to_pip.keys())
    missing = _missing_modules(wanted_modules)
    if not missing:
        print("[wheels] ok (nada a instalar).")
        return

    packages = [module_to_pip[m] for m in missing if m in module_to_pip]
    if not packages:
        return

    print("[wheels] faltando:", ", ".join(missing))
    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels_root),
            *packages,
        ]
    )


# %%
# Setup base (repo/data/outputs)

REPO_ROOT = find_repo_root() or Path(".").resolve()
if not (REPO_ROOT / "src" / "forgeryseg" / "__init__.py").exists():
    raise FileNotFoundError(
        "Não encontrei o código do repo (src/forgeryseg). "
        "No Kaggle, anexe um Dataset com `recodai_bundle/` e/ou defina `FORGERYSEG_REPO_ROOT`."
        f"\nTentativa: {REPO_ROOT}"
    )

DATA_ROOT = find_data_root()
if DATA_ROOT is None or not (DATA_ROOT / "train_images").exists():
    raise FileNotFoundError(
        "Dataset não encontrado. Anexe o dataset da competição e/ou defina `FORGERYSEG_DATA_ROOT`."
        f"\nTentativa: {DATA_ROOT}"
    )

OUTPUTS_ROOT = Path("/kaggle/working/outputs") if is_kaggle() else (REPO_ROOT / "outputs")
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

print("REPO_ROOT:", REPO_ROOT)
print("DATA_ROOT:", DATA_ROOT)
print("OUTPUTS_ROOT:", OUTPUTS_ROOT)


# %%
# Instalação (opcional)
#
# Para treino no Kaggle com Internet ON, o ambiente costuma já ter torch/torchvision.
# Se faltar algo (ex.: segmentation_models_pytorch), instalamos automaticamente via pip.
#
# Nota (Kaggle): o ambiente costuma vir com TensorFlow instalado. Se `protobuf>=5`,
# alguns imports podem emitir erros do tipo:
# `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`.
# Para evitar isso, fixamos `protobuf<5` aqui.
try:
    import google.protobuf  # type: ignore

    pb_ver = str(getattr(google.protobuf, "__version__", "0.0.0"))
    pb_major = int(pb_ver.split(".", maxsplit=1)[0])
except Exception:
    pb_ver = "unknown"
    pb_major = 0

if pb_major >= 5:
    print(f"[pip] protobuf {pb_ver} detectado; ajustando para protobuf<5 (compat Kaggle).")
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

required = [
    "torch",
    "numpy",
    "cv2",
    "albumentations",
    "timm",
    "segmentation_models_pytorch",
    "transformers",
    "sklearn",
]
missing = _missing_modules(required)
if missing:
    module_to_pip = {
        "albumentations": "albumentations",
        "cv2": "opencv-python",
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "timm": "timm",
        "transformers": "transformers",
        "sklearn": "scikit-learn",
    }
    pkgs = [module_to_pip[m] for m in missing if m in module_to_pip]
    # Mantém TensorFlow/Kaggle compatível ao resolver dependências.
    pkgs.append("protobuf<5")
    if pkgs:
        print("[pip] faltando:", ", ".join(missing))
        run_cmd([sys.executable, "-m", "pip", "install", "-q", *pkgs])

    missing = _missing_modules(required)
    if missing:
        raise ImportError(
            "Dependências faltando (mesmo após pip install): "
            + ", ".join(missing)
            + "\nGaranta que o notebook está com Internet=ON no Kaggle e rode novamente."
        )


# %%
# Laboratório de experimentos (edite esta célula)
#
# Objetivo: rodar um "sweep" local (Kaggle) de vários modelos/configs, gerar OOF e
# empacotar checkpoints/caches para uso offline na Fase 00.

from dataclasses import dataclass
from typing import Any, Iterable

import shutil


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def write_config(base_cfg: Path, out_path: Path, overrides: dict[str, Any]) -> Path:
    cfg = read_json(base_cfg)
    cfg.update(overrides)
    return write_json(out_path, cfg)


def _cfg_model_id(cfg_path: Path) -> str:
    cfg = read_json(cfg_path)
    return str(cfg.get("model_id", cfg_path.stem))


def _seg_ckpt_path(outputs_root: Path, model_id: str, fold: int) -> Path:
    return outputs_root / "models_seg" / model_id / f"fold_{fold}" / "best.pt"


def _cls_ckpt_path(outputs_root: Path, fold: int) -> Path:
    return outputs_root / "models_cls" / f"fold_{fold}" / "best.pt"


def _folds_to_run(n_folds: int, fold: int) -> list[int]:
    return [int(fold)] if int(fold) >= 0 else list(range(int(n_folds)))


def _all_exist(paths: Iterable[Path]) -> bool:
    return all(p.exists() for p in paths)


@dataclass(frozen=True)
class SegExperiment:
    name: str
    base_config: Path
    overrides: dict[str, Any]
    include_supplemental: bool = False

    def model_id(self) -> str:
        if "model_id" in self.overrides:
            return str(self.overrides["model_id"])
        return _cfg_model_id(self.base_config)


@dataclass(frozen=True)
class ClsExperiment:
    name: str
    config: Path
    overrides: dict[str, Any]
    include_supplemental: bool = False

    def model_id(self) -> str:
        if "model_id" in self.overrides:
            return str(self.overrides["model_id"])
        cfg = read_json(self.config)
        return str(cfg.get("model_id", self.config.stem))


@dataclass(frozen=True)
class DinoHeadExperiment:
    name: str
    dino_path: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    decoder_dropout: float
    patience: int = 3


def _strip_tu_prefix(encoder_name: str) -> str:
    text = str(encoder_name).strip()
    return text[3:] if text.startswith("tu-") else text


# %%
# Import do projeto (para garantir que o repo está visível)
src_root = REPO_ROOT / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from forgeryseg.offline import configure_cache_dirs

# Cache root (recomendado no Kaggle para empacotar pesos)
CACHE_ROOT = env_path("FORGERYSEG_CACHE_ROOT")
if CACHE_ROOT is None and is_kaggle():
    CACHE_ROOT = Path("/kaggle/working/weights_cache")

if CACHE_ROOT is not None:
    configure_cache_dirs(CACHE_ROOT)
    print("[CACHE] using", CACHE_ROOT)


# %%
# Perfil / budget (edite aqui)
PROFILE = env_str("FORGERYSEG_PROFILE", "quick").strip().lower()  # quick | sweep | full
N_FOLDS = env_int("FORGERYSEG_N_FOLDS", 5)
FOLD = env_int("FORGERYSEG_FOLD", 0)  # -1 = todos
DRY_RUN = env_bool("FORGERYSEG_DRY_RUN", default=False)
SKIP_EXISTING = env_bool("FORGERYSEG_SKIP_EXISTING", default=True)
LIMIT = env_int("FORGERYSEG_LIMIT", 0)  # limita OOF/score (debug)

RUN_WARMUP_CACHE = env_bool("FORGERYSEG_WARMUP_CACHE", default=False)
RUN_SEG_EXPERIMENTS = env_bool("FORGERYSEG_RUN_SEG_EXPERIMENTS", default=True)
RUN_CLS_EXPERIMENTS = env_bool("FORGERYSEG_RUN_CLS_EXPERIMENTS", default=False)
RUN_DINO_HEAD_EXPERIMENTS = env_bool("FORGERYSEG_RUN_DINO_HEAD_EXPERIMENTS", default=False)

RUN_OOF_SEG = env_bool("FORGERYSEG_RUN_OOF_SEG", default=True)
RUN_OOF_DINO = env_bool("FORGERYSEG_RUN_OOF_DINO", default=False)
RUN_TUNE_THRESHOLDS = env_bool("FORGERYSEG_RUN_TUNE_THRESHOLDS", default=False)
RUN_OPTIMIZE_ENSEMBLE = env_bool("FORGERYSEG_RUN_OPTIMIZE_ENSEMBLE", default=False)

# Ajustes automáticos por perfil (p/ acelerar o sweep)
SEG_EPOCHS_OVERRIDE = env_int("FORGERYSEG_SEG_EPOCHS_OVERRIDE", 0)
CLS_EPOCHS_OVERRIDE = env_int("FORGERYSEG_CLS_EPOCHS_OVERRIDE", 0)
if PROFILE in {"quick", "sweep"}:
    SEG_EPOCHS_OVERRIDE = int(SEG_EPOCHS_OVERRIDE or 5)
    CLS_EPOCHS_OVERRIDE = int(CLS_EPOCHS_OVERRIDE or 3)

OOF_TTA = env_str("FORGERYSEG_OOF_TTA", "none").strip() if PROFILE in {"quick", "sweep"} else env_str("FORGERYSEG_OOF_TTA", "none,hflip,vflip").strip()
OOF_TILE_SIZE = env_int("FORGERYSEG_OOF_TILE_SIZE", 1024)
OOF_OVERLAP = env_int("FORGERYSEG_OOF_OVERLAP", 128)

print("PROFILE:", PROFILE)
print("N_FOLDS:", N_FOLDS)
print("FOLD:", FOLD)
print("DRY_RUN:", DRY_RUN)
print("SKIP_EXISTING:", SKIP_EXISTING)
print("LIMIT:", LIMIT)
print("RUN_WARMUP_CACHE:", RUN_WARMUP_CACHE)
print("RUN_SEG_EXPERIMENTS:", RUN_SEG_EXPERIMENTS)
print("RUN_CLS_EXPERIMENTS:", RUN_CLS_EXPERIMENTS)
print("RUN_DINO_HEAD_EXPERIMENTS:", RUN_DINO_HEAD_EXPERIMENTS)
print("RUN_OOF_SEG:", RUN_OOF_SEG)
print("RUN_OOF_DINO:", RUN_OOF_DINO)
print("RUN_TUNE_THRESHOLDS:", RUN_TUNE_THRESHOLDS)
print("RUN_OPTIMIZE_ENSEMBLE:", RUN_OPTIMIZE_ENSEMBLE)
print("SEG_EPOCHS_OVERRIDE:", SEG_EPOCHS_OVERRIDE)
print("CLS_EPOCHS_OVERRIDE:", CLS_EPOCHS_OVERRIDE)
print("OOF_TTA:", OOF_TTA)
print("OOF_TILE_SIZE:", OOF_TILE_SIZE)
print("OOF_OVERLAP:", OOF_OVERLAP)


# %%
# Definição de experimentos (máximo útil, mas nada roda sem os RUN_* acima)
#
# Importante:
# - Segmentação: cada `model_id` cria uma pasta própria em `outputs/models_seg/<model_id>/...` (ok treinar vários).
# - Classificação: por padrão o treino escreve em `outputs/models_cls/fold_*/...` (uma “família” por vez).
#   Neste notebook você seleciona **1** classificador ativo para evitar sobrescrita.

SEG_CONFIGS = [
    REPO_ROOT / "configs" / "seg_unetpp_tu_convnext_small.json",
    REPO_ROOT / "configs" / "seg_unetpp_tu_swin_tiny.json",
    REPO_ROOT / "configs" / "seg_deeplabv3p_tu_resnest101e.json",
    REPO_ROOT / "configs" / "seg_segformer_mit_b2.json",
    REPO_ROOT / "configs" / "seg_dinov2_base.json",
    REPO_ROOT / "configs" / "seg_dinov2_base_640.json",
    REPO_ROOT / "configs" / "seg_hero_effnet_b7.json",
]
for p in SEG_CONFIGS:
    if not p.exists():
        raise FileNotFoundError(p)

CLS_CONFIGS = [
    REPO_ROOT / "configs" / "cls_effnet_b4.json",
    REPO_ROOT / "configs" / "cls_convnext_small_encoder.json",
    REPO_ROOT / "configs" / "cls_swin_tiny_encoder.json",
    REPO_ROOT / "configs" / "cls_dinov2_base.json",
]
for p in CLS_CONFIGS:
    if not p.exists():
        raise FileNotFoundError(p)

SEG_LEVEL = env_str("FORGERYSEG_SEG_LEVEL", "base").strip().lower()  # base | plus | max
SEG_FILTER = env_str("FORGERYSEG_SEG_FILTER", "").strip()  # opcional: "convnext,segformer"
CLS_ACTIVE = env_str("FORGERYSEG_CLS_ACTIVE", "cls_effnet_b4").strip()  # escolha 1

seg_experiments_all: list[SegExperiment] = []
for base_cfg in SEG_CONFIGS:
    base = read_json(base_cfg)
    mid = str(base.get("model_id", base_cfg.stem))

    # Base
    seg_experiments_all.append(SegExperiment(name=mid, base_config=base_cfg, overrides={}))

    # + supplemental (pode ajudar recall)
    seg_experiments_all.append(
        SegExperiment(
            name=f"{mid}_supp",
            base_config=base_cfg,
            overrides={"model_id": f"{mid}_supp", "include_supplemental": True},
            include_supplemental=True,
        )
    )

    # + FFT channels (4ch) (experimento agressivo: pode melhorar copy-move)
    already_has_extra_channels = bool(base.get("use_freq_channels", False)) or int(base.get("in_channels", 3)) != 3
    if not already_has_extra_channels:
        seg_experiments_all.append(
            SegExperiment(
                name=f"{mid}_fft",
                base_config=base_cfg,
                overrides={"model_id": f"{mid}_fft", "use_freq_channels": True},
            )
        )

    # + BCE+Dice (às vezes melhora componentes pequenas)
    if str(base.get("backend", "smp")).lower() in {"smp", ""}:
        seg_experiments_all.append(
            SegExperiment(
                name=f"{mid}_bce_dice",
                base_config=base_cfg,
                overrides={"model_id": f"{mid}_bce_dice", "loss": "bce_dice", "dice_weight": 1.0},
            )
        )

# Força HF configs a baixarem pesos (internet ON)
seg_experiments_all = [
    SegExperiment(
        name=e.name,
        base_config=e.base_config,
        overrides={
            **e.overrides,
            **({"local_files_only": False} if str(read_json(e.base_config).get("backend", "")).lower() in {"dinov2", "hf"} else {}),
        },
        include_supplemental=e.include_supplemental,
    )
    for e in seg_experiments_all
]

def _select_seg_experiments(experiments: list[SegExperiment], level: str, flt: str) -> list[SegExperiment]:
    level = str(level).strip().lower()
    if level not in {"base", "plus", "max"}:
        raise ValueError("FORGERYSEG_SEG_LEVEL deve ser: base | plus | max")

    if level == "max":
        selected = list(experiments)
    elif level == "plus":
        selected = [e for e in experiments if not (e.name.endswith("_fft") or e.name.endswith("_bce_dice"))]
    else:  # base
        selected = [e for e in experiments if not (e.name.endswith("_supp") or e.name.endswith("_fft") or e.name.endswith("_bce_dice"))]

    if flt.strip():
        tokens = [t.strip().lower() for t in flt.split(",") if t.strip()]
        selected = [e for e in selected if any(tok in e.name.lower() for tok in tokens)]

    # remove duplicates mantendo ordem
    seen = set()
    out = []
    for e in selected:
        if e.name in seen:
            continue
        seen.add(e.name)
        out.append(e)
    return out


seg_experiments = _select_seg_experiments(seg_experiments_all, SEG_LEVEL, SEG_FILTER)

cls_experiments_all: list[ClsExperiment] = []
for cfg_path in CLS_CONFIGS:
    base = read_json(cfg_path)
    mid = str(base.get("model_id", cfg_path.stem))
    cls_experiments_all.append(ClsExperiment(name=mid, config=cfg_path, overrides={}))
    cls_experiments_all.append(
        ClsExperiment(
            name=f"{mid}_supp",
            config=cfg_path,
            overrides={"model_id": f"{mid}_supp", "include_supplemental": True},
            include_supplemental=True,
        )
    )

# Força HF configs a baixarem pesos (internet ON)
cls_experiments_all = [
    ClsExperiment(
        name=e.name,
        config=e.config,
        overrides={
            **e.overrides,
            **({"local_files_only": False} if str(read_json(e.config).get("backend", "")).lower() in {"dinov2", "hf"} else {}),
        },
        include_supplemental=e.include_supplemental,
    )
    for e in cls_experiments_all
]

cls_experiments = [e for e in cls_experiments_all if e.name == CLS_ACTIVE]
if RUN_CLS_EXPERIMENTS and not cls_experiments:
    raise ValueError(f"CLS_ACTIVE inválido: {CLS_ACTIVE}. Opções: {[e.name for e in cls_experiments_all]}")

print(f"Seg experiments (all): {len(seg_experiments_all)} | active={len(seg_experiments)} | level={SEG_LEVEL} filter={SEG_FILTER!r}")
print(f"Cls experiments (all): {len(cls_experiments_all)} | active={len(cls_experiments)} | CLS_ACTIVE={CLS_ACTIVE!r}")
folds_to_run = _folds_to_run(N_FOLDS, FOLD)
if RUN_SEG_EXPERIMENTS:
    seg_runs = int(len(seg_experiments) * len(folds_to_run))
    print(f"[plan] SEG: {len(seg_experiments)} experimento(s) x {len(folds_to_run)} fold(s) = {seg_runs} treino(s)")
    if not DRY_RUN and seg_runs >= 6:
        print("[WARN] Muitos treinos de segmentação selecionados; considere usar FORGERYSEG_SEG_FILTER='hero' (ou outro filtro).")
if RUN_CLS_EXPERIMENTS:
    cls_runs = int(len(cls_experiments) * len(folds_to_run))
    print(f"[plan] CLS: {len(cls_experiments)} experimento(s) x {len(folds_to_run)} fold(s) = {cls_runs} treino(s)")


# %%
# Experimentos DINO head (opcional) — útil como baseline leve / ablação
DINO_HEAD_LEVEL = env_str("FORGERYSEG_DINO_HEAD_LEVEL", "base").strip().lower()  # base | max
DINO_HEAD_FILTER = env_str("FORGERYSEG_DINO_HEAD_FILTER", "").strip()
DINO_PATH = env_str("FORGERYSEG_DINO_PATH", "facebook/dinov2-base").strip()

dino_head_epochs = env_int("FORGERYSEG_DINO_HEAD_EPOCHS", 0)
if int(dino_head_epochs) <= 0:
    dino_head_epochs = 5 if PROFILE in {"quick", "sweep"} else 10

dino_head_experiments_all: list[DinoHeadExperiment] = [
    DinoHeadExperiment(
        name="dino_head_512",
        dino_path=DINO_PATH,
        image_size=512,
        batch_size=4,
        epochs=int(dino_head_epochs),
        lr=3e-4,
        weight_decay=1e-2,
        decoder_dropout=0.0,
        patience=3,
    ),
    DinoHeadExperiment(
        name="dino_head_384",
        dino_path=DINO_PATH,
        image_size=384,
        batch_size=8,
        epochs=int(dino_head_epochs),
        lr=3e-4,
        weight_decay=1e-2,
        decoder_dropout=0.0,
        patience=3,
    ),
    DinoHeadExperiment(
        name="dino_head_640",
        dino_path=DINO_PATH,
        image_size=640,
        batch_size=2,
        epochs=int(dino_head_epochs),
        lr=3e-4,
        weight_decay=1e-2,
        decoder_dropout=0.0,
        patience=3,
    ),
    DinoHeadExperiment(
        name="dino_head_512_do10",
        dino_path=DINO_PATH,
        image_size=512,
        batch_size=4,
        epochs=int(dino_head_epochs),
        lr=3e-4,
        weight_decay=1e-2,
        decoder_dropout=0.10,
        patience=3,
    ),
]

if DINO_HEAD_LEVEL == "base":
    dino_head_experiments = [e for e in dino_head_experiments_all if e.name in {"dino_head_512", "dino_head_512_do10"}]
elif DINO_HEAD_LEVEL == "max":
    dino_head_experiments = list(dino_head_experiments_all)
else:
    raise ValueError("FORGERYSEG_DINO_HEAD_LEVEL deve ser: base | max")

if DINO_HEAD_FILTER.strip():
    tokens = [t.strip().lower() for t in DINO_HEAD_FILTER.split(",") if t.strip()]
    dino_head_experiments = [e for e in dino_head_experiments if any(tok in e.name.lower() for tok in tokens)]

print(f"DINO head experiments (all): {len(dino_head_experiments_all)} | active={len(dino_head_experiments)} | level={DINO_HEAD_LEVEL} filter={DINO_HEAD_FILTER!r}")


# %%
# Warmup de cache (opcional): baixa pesos pretrained para não depender de downloads depois.
# Isso ajuda a empacotar `weights_cache/` para uso offline no Kaggle.
if RUN_WARMUP_CACHE:
    if CACHE_ROOT is None:
        raise RuntimeError("Defina FORGERYSEG_CACHE_ROOT ou rode no Kaggle (default /kaggle/working/weights_cache).")

    # Timm weights (classificadores e encoders `tu-*` do SMP)
    timm_models: set[str] = set()
    for cfg_path in CLS_CONFIGS:
        cfg = read_json(cfg_path)
        backend = str(cfg.get("backend", "timm")).lower()
        if backend in {"timm", "timm_encoder"}:
            timm_models.add(str(cfg.get("model_name", "")))
    for cfg_path in SEG_CONFIGS:
        cfg = read_json(cfg_path)
        encoder = str(cfg.get("encoder_name", ""))
        if encoder.startswith("tu-"):
            timm_models.add(_strip_tu_prefix(encoder))

    timm_models = {m for m in timm_models if m}
    if timm_models:
        dl_script = REPO_ROOT / "scripts" / "download_timm_weights.py"
        if dl_script.exists():
            run_cmd(
                [
                    sys.executable,
                    str(dl_script),
                    "--models",
                    ",".join(sorted(timm_models)),
                    "--out-dir",
                    str(CACHE_ROOT / "timm_state_dict"),
                ]
            )
        else:
            print("[warmup] scripts/download_timm_weights.py não encontrado (pulando).")

    # HuggingFace weights (DINOv2)
    try:
        from transformers import AutoModel

        hf_ids = set()
        for cfg_path in SEG_CONFIGS + CLS_CONFIGS:
            cfg = read_json(cfg_path)
            if str(cfg.get("backend", "")).lower() in {"dinov2", "hf"}:
                hf_ids.add(str(cfg.get("hf_model_id", "")))
        hf_ids = {x for x in hf_ids if x}
        for mid in sorted(hf_ids):
            print("[hf] baixando:", mid)
            AutoModel.from_pretrained(mid)
    except Exception as exc:
        print("[warmup] transformers não disponível ou falhou:", exc)


# %%
# Runner: treino (seg/cls) + OOF + tuning

def _maybe_override_epochs(cfg: dict[str, Any], epochs_override: int) -> dict[str, Any]:
    if int(epochs_override) <= 0:
        return cfg
    out = dict(cfg)
    out["epochs"] = int(epochs_override)
    return out


def _train_seg(exp: SegExperiment) -> None:
    train_seg_script = REPO_ROOT / "scripts" / "train_seg_smp_cv.py"
    if not train_seg_script.exists():
        raise FileNotFoundError(train_seg_script)

    model_id = exp.model_id()
    out_cfg = OUTPUTS_ROOT / "configs" / f"seg_{exp.name}.json"
    cfg = read_json(exp.base_config)
    cfg.update(exp.overrides)
    cfg = _maybe_override_epochs(cfg, SEG_EPOCHS_OVERRIDE)
    out_cfg = write_json(out_cfg, cfg)

    folds_to_run = _folds_to_run(N_FOLDS, FOLD)
    if SKIP_EXISTING and _all_exist(_seg_ckpt_path(OUTPUTS_ROOT, model_id, f) for f in folds_to_run):
        print(f"[SEG] {model_id}: ckpt já existe (pulando).")
        return

    cmd = [
        sys.executable,
        str(train_seg_script),
        "--config",
        str(out_cfg),
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(OUTPUTS_ROOT),
        "--folds",
        str(N_FOLDS),
    ]
    if int(FOLD) >= 0:
        cmd += ["--fold", str(FOLD)]
    if exp.include_supplemental or bool(cfg.get("include_supplemental", False)):
        cmd += ["--include-supplemental"]
    if SKIP_EXISTING:
        cmd += ["--skip-existing"]
    if CACHE_ROOT is not None:
        cmd += ["--cache-root", str(CACHE_ROOT)]

    if DRY_RUN:
        print("[dry-run]", " ".join(map(str, cmd)))
        return
    run_cmd(cmd)


def _train_cls(exp: ClsExperiment) -> None:
    train_cls_script = REPO_ROOT / "scripts" / "train_cls_cv.py"
    if not train_cls_script.exists():
        raise FileNotFoundError(train_cls_script)

    out_cfg = OUTPUTS_ROOT / "configs" / f"cls_{exp.name}.json"
    cfg = read_json(exp.config)
    cfg.update(exp.overrides)
    cfg = _maybe_override_epochs(cfg, CLS_EPOCHS_OVERRIDE)
    out_cfg = write_json(out_cfg, cfg)

    folds_to_run = _folds_to_run(N_FOLDS, FOLD)
    if SKIP_EXISTING and _all_exist(_cls_ckpt_path(OUTPUTS_ROOT, f) for f in folds_to_run):
        try:
            import torch

            expected_id = exp.model_id()
            same = True
            for fold_id in folds_to_run:
                ckpt = torch.load(_cls_ckpt_path(OUTPUTS_ROOT, fold_id), map_location="cpu")
                ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
                if str(ckpt_cfg.get("model_id", "")) != str(expected_id):
                    same = False
                    break
            if same:
                print(f"[CLS] ckpt já existe para {expected_id} (pulando).")
                return
        except Exception:
            print("[CLS] ckpt existe, mas não consegui validar model_id; retreinando (para evitar sobrescrita errada).")

    cmd = [
        sys.executable,
        str(train_cls_script),
        "--config",
        str(out_cfg),
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(OUTPUTS_ROOT),
        "--folds",
        str(N_FOLDS),
    ]
    if int(FOLD) >= 0:
        cmd += ["--fold", str(FOLD)]
    if exp.include_supplemental or bool(cfg.get("include_supplemental", False)):
        cmd += ["--include-supplemental"]
    if SKIP_EXISTING:
        cmd += ["--skip-existing"]
    if CACHE_ROOT is not None:
        cmd += ["--cache-root", str(CACHE_ROOT)]

    if DRY_RUN:
        print("[dry-run]", " ".join(map(str, cmd)))
        return
    run_cmd(cmd)


def _oof_seg(model_id: str) -> None:
    script = REPO_ROOT / "scripts" / "predict_seg_oof.py"
    if not script.exists():
        raise FileNotFoundError(script)

    cmd = [
        sys.executable,
        str(script),
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(OUTPUTS_ROOT),
        "--model-id",
        str(model_id),
        "--folds",
        str(N_FOLDS),
        "--tta",
        str(OOF_TTA),
        "--tile-size",
        str(OOF_TILE_SIZE),
        "--overlap",
        str(OOF_OVERLAP),
    ]
    if int(FOLD) >= 0:
        cmd += ["--fold", str(FOLD)]
    if int(LIMIT) > 0:
        cmd += ["--limit", str(LIMIT)]

    if DRY_RUN:
        print("[dry-run]", " ".join(map(str, cmd)))
        return
    run_cmd(cmd)


def _tune_postprocess(model_id: str) -> None:
    script = REPO_ROOT / "scripts" / "tune_thresholds.py"
    if not script.exists():
        raise FileNotFoundError(script)

    preds_root = OUTPUTS_ROOT / "oof" / model_id
    out_cfg = OUTPUTS_ROOT / "configs" / f"postproc_{model_id}.json"

    # Grid pequeno por padrão (aumente se quiser)
    cmd = [
        sys.executable,
        str(script),
        "--data-root",
        str(DATA_ROOT),
        "--preds-root",
        str(preds_root),
        "--folds",
        str(N_FOLDS),
        "--adaptive-threshold",
        "--threshold-factors",
        "0.2,0.3,0.4",
        "--min-areas",
        "0,30,64,128",
        "--min-area-percents",
        "0.0002,0.0005,0.001",
        "--min-confidences",
        "0.30,0.33,0.36,0.40",
        "--out-config",
        str(out_cfg),
    ]
    if int(LIMIT) > 0:
        cmd += ["--limit", str(LIMIT)]

    if DRY_RUN:
        print("[dry-run]", " ".join(map(str, cmd)))
        return
    run_cmd(cmd)


def _train_dino_head(exp: DinoHeadExperiment) -> None:
    script = REPO_ROOT / "scripts" / "train_dino_head.py"
    if not script.exists():
        raise FileNotFoundError(script)

    out_root = OUTPUTS_ROOT / "models_dino" / exp.name
    folds_to_run = _folds_to_run(N_FOLDS, FOLD)
    for fold_id in folds_to_run:
        ckpt_path = out_root / f"fold_{fold_id}" / "best.pt"
        if SKIP_EXISTING and ckpt_path.exists():
            print(f"[DINO] {exp.name} fold={fold_id}: ckpt já existe (pulando).")
            continue

        cmd = [
            sys.executable,
            str(script),
            "--data-root",
            str(DATA_ROOT),
            "--output-dir",
            str(out_root),
            "--folds",
            str(N_FOLDS),
            "--fold",
            str(fold_id),
            "--dino-path",
            str(exp.dino_path),
            "--image-size",
            str(int(exp.image_size)),
            "--batch-size",
            str(int(exp.batch_size)),
            "--epochs",
            str(int(exp.epochs)),
            "--lr",
            str(float(exp.lr)),
            "--weight-decay",
            str(float(exp.weight_decay)),
            "--decoder-dropout",
            str(float(exp.decoder_dropout)),
            "--patience",
            str(int(exp.patience)),
        ]
        if CACHE_ROOT is not None:
            cmd += ["--cache-dir", str(CACHE_ROOT / "hf")]

        if DRY_RUN:
            print("[dry-run]", " ".join(map(str, cmd)))
            continue
        run_cmd(cmd)


def _oof_dino(exp: DinoHeadExperiment) -> None:
    script = REPO_ROOT / "scripts" / "predict_dino_oof.py"
    if not script.exists():
        raise FileNotFoundError(script)

    preds_root = OUTPUTS_ROOT / "oof_dino" / exp.name
    head_dir = OUTPUTS_ROOT / "models_dino" / exp.name
    run_dir = OUTPUTS_ROOT / "runs_dino" / exp.name

    cmd = [
        sys.executable,
        str(script),
        "--data-root",
        str(DATA_ROOT),
        "--preds-root",
        str(preds_root),
        "--folds",
        str(N_FOLDS),
        "--head-ckpt-dir",
        str(head_dir),
        "--tta",
        str(OOF_TTA if PROFILE in {"quick", "sweep"} else "none,hflip,vflip,rot90,rot180,rot270"),
        "--run-dir",
        str(run_dir),
    ]
    if int(FOLD) >= 0:
        cmd += ["--fold", str(FOLD)]
    if int(LIMIT) > 0:
        cmd += ["--limit", str(LIMIT)]
    if CACHE_ROOT is not None:
        cmd += ["--cache-dir", str(CACHE_ROOT / "hf")]

    if DRY_RUN:
        print("[dry-run]", " ".join(map(str, cmd)))
        return
    run_cmd(cmd)


# %%
# 1) Treino de segmentação (sweep)
if RUN_SEG_EXPERIMENTS:
    for exp in seg_experiments:
        _train_seg(exp)
else:
    print("[SEG] RUN_SEG_EXPERIMENTS=False (pulando).")


# %%
# 2) Treino de classificação (opcional, para gate na submissão)
if RUN_CLS_EXPERIMENTS:
    for exp in cls_experiments:
        _train_cls(exp)
else:
    print("[CLS] RUN_CLS_EXPERIMENTS=False (pulando).")


# %%
# 2b) Treino DINO head (opcional)
if RUN_DINO_HEAD_EXPERIMENTS:
    for exp in dino_head_experiments:
        _train_dino_head(exp)
else:
    print("[DINO] RUN_DINO_HEAD_EXPERIMENTS=False (pulando).")


# %%
# 2c) OOF + score (DINO head)
if RUN_OOF_DINO:
    for exp in dino_head_experiments:
        _oof_dino(exp)
else:
    print("[DINO OOF] RUN_OOF_DINO=False (pulando).")


# %%
# 3) OOF + score rápido (segmentação)
if RUN_OOF_SEG:
    for exp in seg_experiments:
        _oof_seg(exp.model_id())
else:
    print("[OOF] RUN_OOF_SEG=False (pulando).")


# %%
# 4) Tuning de pós-processamento (segmentação)
if RUN_TUNE_THRESHOLDS:
    for exp in seg_experiments:
        _tune_postprocess(exp.model_id())
else:
    print("[TUNE] RUN_TUNE_THRESHOLDS=False (pulando).")


# %%
# 5) Otimizar pesos do ensemble (proxy) (requer OOF de múltiplos modelos)
if RUN_OPTIMIZE_ENSEMBLE:
    script = REPO_ROOT / "scripts" / "optimize_ensemble.py"
    if not script.exists():
        raise FileNotFoundError(script)

    # Edite aqui se quiser restringir o ensemble a poucos modelos base (mais realista)
    ensemble_model_ids = sorted({e.model_id() for e in seg_experiments})

    out_path = OUTPUTS_ROOT / "ensemble_weights.json"
    cmd = [
        sys.executable,
        str(script),
        "--data-root",
        str(DATA_ROOT),
        "--oof-dir",
        str(OUTPUTS_ROOT / "oof"),
        "--models",
        ",".join(ensemble_model_ids),
        "--out",
        str(out_path),
    ]
    if DRY_RUN:
        print("[dry-run]", " ".join(map(str, cmd)))
    else:
        run_cmd(cmd)
        print("[ENSEMBLE] pesos ->", out_path)
else:
    print("[ENSEMBLE] RUN_OPTIMIZE_ENSEMBLE=False (pulando).")


# %%
# Sumário rápido dos checkpoints de segmentação (dice@0.5 da validação)
def summarize_seg_checkpoints(outputs_root: Path) -> None:
    import torch

    base = outputs_root / "models_seg"
    if not base.exists():
        print("[summary] outputs/models_seg não existe.")
        return
    rows = []
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        scores = []
        for ckpt_path in sorted(model_dir.glob("fold_*/best.pt")):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                score = ckpt.get("score", None) if isinstance(ckpt, dict) else None
                if score is not None:
                    scores.append(float(score))
            except Exception:
                continue
        if scores:
            rows.append((model_dir.name, float(sum(scores) / len(scores)), len(scores)))
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    print("Top seg ckpts (mean dice):")
    for name, mean_dice, n in rows[:15]:
        print(f"- {name}: mean_dice={mean_dice:.4f} folds={n}")


summarize_seg_checkpoints(OUTPUTS_ROOT)


# %%
# Empacotar artefatos para download
#
# Saídas (Kaggle):
# - `/kaggle/working/outputs_pretrain.zip` (checkpoints + logs + configs gerados)
# - `/kaggle/working/weights_cache_pretrain.zip` (opcional; útil para submissão offline sem downloads)

zip_base = Path("/kaggle/working/outputs_pretrain") if is_kaggle() else (OUTPUTS_ROOT.parent / "outputs_pretrain")
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(OUTPUTS_ROOT))
print("ZIP outputs:", zip_path)
print("Conteúdo (top-level):", [p.name for p in OUTPUTS_ROOT.iterdir()])

if CACHE_ROOT is not None and CACHE_ROOT.exists():
    cache_zip_base = Path("/kaggle/working/weights_cache_pretrain") if is_kaggle() else (CACHE_ROOT.parent / "weights_cache_pretrain")
    cache_zip = shutil.make_archive(str(cache_zip_base), "zip", root_dir=str(CACHE_ROOT))
    print("ZIP weights_cache:", cache_zip)
else:
    print("[zip] CACHE_ROOT não definido/existe (pulando weights_cache zip).")
