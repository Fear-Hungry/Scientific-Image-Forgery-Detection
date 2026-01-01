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
# # Pipeline completo — Recod.ai/LUC (Kaggle)
#
# Neste notebook, iremos desenvolver um **pipeline completo** para o desafio
# **Recod.ai/LUC — Scientific Image Forgery Detection** do Kaggle. O objetivo do
# desafio é detectar e segmentar regiões manipuladas (*copy-move forgeries*) em
# imagens científicas biomédicas.
#
# O problema envolve duas tarefas principais:
#
# 1) **Classificação binária** — identificar se a imagem contém fraude
#    (*authentic* vs. *forged*)
# 2) **Segmentação** — indicar os **pixels exatos** onde há fraude
#
# Este pipeline cobre tudo **do treinamento** dos modelos **até a geração do
# `submission.csv`**, incluindo boas práticas de participantes de alto desempenho:
# **TTA** (Test Time Augmentation) e **pós-processamento morfológico**.
#
# Fluxo do notebook:
#
# - **Setup** (paths, cache offline, validações)
# - **Treino opcional** do classificador
# - **Treino opcional** do segmentador
# - **Inferência + pós-processamento** (TTA + morfologia)
# - **Geração do `submission.csv`**

# %% [markdown]
# ## 1. Modo Kaggle offline (internet OFF) + setup
#
# Para rodar **offline** no Kaggle (modo submissão), este notebook foi pensado
# para funcionar com:
#
# - **Dataset da competição** montado em `/kaggle/input/...` (padrão).
# - **Este repositório** empacotado como Kaggle Dataset (recomendado).
# - *(Opcional)* **wheels** locais em `.../recodai_bundle/wheels/` para instalar
#   dependências que não vêm no ambiente do Kaggle.
# - *(Opcional)* **cache de pesos** (timm/torch hub e/ou HuggingFace) para usar
#   modelos/pretrained sem downloads.
#
# Variáveis de ambiente úteis (todas opcionais):
#
# - `FORGERYSEG_DATA_ROOT`: força o path do dataset (ex.: `/kaggle/input/.../recodai`).
# - `FORGERYSEG_REPO_ROOT`: força o path do repo (ex.: `/kaggle/input/<ds>/recodai_bundle`).
# - `FORGERYSEG_WHEELS_ROOT`: força o path dos wheels (ex.: `/kaggle/input/<ds>/recodai_bundle/wheels`).
# - `FORGERYSEG_CACHE_ROOT`: força o path do cache (ex.: `/kaggle/input/<ds>/recodai_bundle/weights_cache`).
# - `FORGERYSEG_ALLOW_DOWNLOAD=1`: libera downloads (NÃO use na submissão offline).
#
# Dica: para “empacotar” tudo em um Dataset para Kaggle, use `recodai_bundle/`
# (há pastas `wheels/` e `weights_cache/` prontas).
#
# ---
#
# ## 2. Instalação de Pacotes (opcional)
#
# Nesta seção, instalamos as bibliotecas necessárias e configuramos parâmetros
# globais (diretórios de dados e *device*). Garantimos uso de GPU quando disponível.
#
# Bibliotecas usadas:
#
# - **PyTorch**: framework de deep learning.
# - **Albumentations**: aumentos de dados (imagem + máscara).
# - **segmentation_models_pytorch (SMP)**: U-Net, U-Net++, DeepLabV3+, etc.
# - **Transformers (HuggingFace)**: modelos como DINOv2 e SegFormer.
# - **OpenCV**: operações morfológicas no pós-processamento.
# - **numpy/pandas**: manipulação de dados.
#
# Instalação (se necessário no Kaggle):
#
# ```
# !pip install -q segmentation-models-pytorch==0.3.0 albumentations==1.3.0 transformers==4.34.0 timm==0.9.2
# ```

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
    value = os.environ.get(name, "")
    if value == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def find_wheels_root() -> Path | None:
    explicit = env_path("FORGERYSEG_WHEELS_ROOT")
    if explicit is not None:
        return explicit if explicit.exists() else None

    local_candidates = [Path("recodai_bundle") / "wheels", Path("wheels")]
    for cand in local_candidates:
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


def find_cache_root() -> Path | None:
    explicit = env_path("FORGERYSEG_CACHE_ROOT")
    if explicit is not None:
        return explicit if explicit.exists() else None

    local_candidates = [Path("recodai_bundle") / "weights_cache", Path("weights_cache")]
    for cand in local_candidates:
        if cand.exists():
            return cand

    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                for cand in (ds / "weights_cache", ds / "recodai_bundle" / "weights_cache"):
                    if cand.exists():
                        return cand
    return None


def find_requirements_file() -> Path | None:
    candidates: list[Path] = []
    here = Path(".").resolve()
    for cand in [here] + list(here.parents):
        candidates.append(cand / "requirements.txt")
        candidates.append(cand / "recodai_bundle" / "requirements.txt")
    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                candidates.append(ds / "requirements.txt")
                candidates.append(ds / "recodai_bundle" / "requirements.txt")
    for cand in candidates:
        if cand.exists():
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


def maybe_install_from_wheels(wheels_root: Path | None) -> None:
    """Try to install only what is missing, using local wheels (offline-safe)."""
    if wheels_root is None:
        return

    module_to_pip = {
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "huggingface_hub": "huggingface-hub",
        "safetensors": "safetensors",
        "tqdm": "tqdm",
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
    print("[wheels] instalando via --no-index/--find-links:", wheels_root)
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


def has_any_ckpt(dir_name: str, pattern: str, outputs_root: Path) -> bool:
    if is_kaggle():
        ki = Path("/kaggle/input")
        if ki.exists():
            for ds in sorted(ki.glob("*")):
                for base in (ds, ds / "recodai_bundle"):
                    cand = base / "outputs" / dir_name
                    if cand.exists() and any(cand.glob(pattern)):
                        return True
    cand = outputs_root / dir_name
    return cand.exists() and any(cand.glob(pattern))


def write_config(base_cfg: Path, out_path: Path, overrides: dict) -> Path:
    with base_cfg.open("r") as f:
        cfg = json.load(f)
    cfg.update(overrides)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    return out_path


# %%
# Config base do notebook (edite via env vars, se quiser)
KAGGLE = is_kaggle()
ALLOW_DOWNLOAD = env_bool("FORGERYSEG_ALLOW_DOWNLOAD", default=not KAGGLE)
OFFLINE = bool(KAGGLE and not ALLOW_DOWNLOAD)

# Instaladores (opcionais)
RUN_PIP_INSTALL = env_bool("FORGERYSEG_PIP_INSTALL", default=ALLOW_DOWNLOAD)
INSTALL_WHEELS = env_bool("FORGERYSEG_INSTALL_WHEELS", default=KAGGLE and not ALLOW_DOWNLOAD)

WHEELS_ROOT = find_wheels_root() if INSTALL_WHEELS else None
if INSTALL_WHEELS:
    if WHEELS_ROOT is None:
        print("[wheels] nenhum wheel root encontrado (ok se já tiver as libs no ambiente).")
    else:
        maybe_install_from_wheels(WHEELS_ROOT)
else:
    print("INSTALL_WHEELS=False (pulando).")

if RUN_PIP_INSTALL:
    # Atenção: isso usa internet (a menos que você esteja apontando um index local).
    req_path = find_requirements_file()
    if req_path is None:
        raise FileNotFoundError(
            "RUN_PIP_INSTALL=True, mas não encontrei `requirements.txt`. "
            "Defina `FORGERYSEG_REPO_ROOT`/mude o cwd, ou desative `FORGERYSEG_PIP_INSTALL`."
        )
    run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
else:
    print("RUN_PIP_INSTALL=False (pulando).")

try:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)
except Exception as exc:
    raise RuntimeError("PyTorch não disponível. No Kaggle ele já vem instalado; localmente instale `torch`.") from exc

# Validação rápida de deps críticas (falha cedo, com mensagem clara)
required_modules = [
    "numpy",
    "cv2",
    "timm",
    "segmentation_models_pytorch",
]
missing = _missing_modules(required_modules)
if missing:
    msg = (
        "Dependências Python faltando: "
        + ", ".join(missing)
        + "\n- No Kaggle offline: anexe um Dataset com wheels em `.../recodai_bundle/wheels/` e rode com "
        + "`FORGERYSEG_INSTALL_WHEELS=1`."
        + "\n- Alternativa: habilite internet e use `FORGERYSEG_PIP_INSTALL=1` (não serve para submissão offline)."
    )
    raise ImportError(msg)

# %%
# Paths dos dados da competição (Kaggle)
DATA_ROOT = env_path("FORGERYSEG_DATA_ROOT") or find_data_root() or Path("data/recodai")

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        "DATA_ROOT não existe. Defina `FORGERYSEG_DATA_ROOT` ou anexe o dataset do Kaggle."
        f"\nTentativa: {DATA_ROOT}"
    )

SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"
if SAMPLE_SUB_PATH.exists():
    try:
        import pandas as pd

        sample_df = pd.read_csv(SAMPLE_SUB_PATH)
        print("Colunas do sample submission:", sample_df.columns.tolist())
        print(sample_df.head(3))
    except Exception:
        print("[warn] pandas não disponível para preview do sample_submission.csv (ok).")
else:
    print("sample_submission.csv não encontrado (ok fora do Kaggle).")

# %%
# Fase 0 — Sanidade Kaggle (lembrete)
print("Kaggle constraints (lembrete):")
print("- Runtime <= 4h (CPU/GPU)")
print("- Internet: OFF no submit")
print("- Outputs: /kaggle/working/outputs (checkpoints)")


# %%
# Fase 0 — Config (edite esta célula)
REPO_ROOT = env_path("FORGERYSEG_REPO_ROOT") or find_repo_root() or Path(".").resolve()
if not (REPO_ROOT / "src" / "forgeryseg" / "__init__.py").exists():
    raise FileNotFoundError(
        "Não encontrei o código do repo (src/forgeryseg). "
        "No Kaggle, anexe um Dataset com `recodai_bundle/` e/ou defina `FORGERYSEG_REPO_ROOT`."
        f"\nTentativa: {REPO_ROOT}"
    )

# Onde salvar checkpoints e logs
OUTPUTS_ROOT = Path("/kaggle/working/outputs") if is_kaggle() else REPO_ROOT / "outputs"
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

# Cache offline (opcional). Ex.: /kaggle/input/<dataset>/weights_cache
CACHE_ROOT = find_cache_root()

# Configs base (pode trocar por outras em configs/)
CLS_CONFIG = REPO_ROOT / "configs" / "cls_effnet_b4.json"
SEG_CONFIG = REPO_ROOT / "configs" / "seg_unetpp_tu_convnext_small.json"
INFER_CONFIG = REPO_ROOT / "configs" / "infer_ensemble.json"

for cfg_path in (CLS_CONFIG, SEG_CONFIG, INFER_CONFIG):
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config não encontrado: {cfg_path}")

# CV / folds (FOLD=-1 para treinar todos)
N_FOLDS = int(os.environ.get("FORGERYSEG_N_FOLDS", "5"))
FOLD = int(os.environ.get("FORGERYSEG_FOLD", "0"))

# Heurística: se já existem checkpoints, não treina (a menos que você force via env)
HAS_CLS_CKPT = has_any_ckpt("models_cls", "fold_*/best.pt", OUTPUTS_ROOT)
HAS_SEG_CKPT = has_any_ckpt("models_seg", "*/*/best.pt", OUTPUTS_ROOT)

RUN_TRAIN_CLS = env_bool("FORGERYSEG_RUN_TRAIN_CLS", default=not HAS_CLS_CKPT)
RUN_TRAIN_SEG = env_bool("FORGERYSEG_RUN_TRAIN_SEG", default=not HAS_SEG_CKPT)
RUN_SUBMISSION = env_bool("FORGERYSEG_RUN_SUBMISSION", default=True)

print("REPO_ROOT:", REPO_ROOT)
print("DATA_ROOT:", DATA_ROOT)
print("OUTPUTS_ROOT:", OUTPUTS_ROOT)
print("CACHE_ROOT:", CACHE_ROOT)
print("HAS_CLS_CKPT:", HAS_CLS_CKPT)
print("HAS_SEG_CKPT:", HAS_SEG_CKPT)
print("RUN_TRAIN_CLS:", RUN_TRAIN_CLS)
print("RUN_TRAIN_SEG:", RUN_TRAIN_SEG)
print("RUN_SUBMISSION:", RUN_SUBMISSION)
print("ALLOW_DOWNLOAD:", ALLOW_DOWNLOAD)
print("RUN_PIP_INSTALL:", RUN_PIP_INSTALL)
print("INSTALL_WHEELS:", INSTALL_WHEELS)
print("WHEELS_ROOT:", WHEELS_ROOT)
print("N_FOLDS:", N_FOLDS)
print("FOLD:", FOLD)

# %%
# Fase 0 — Import do projeto + cache offline
if (REPO_ROOT / "src" / "forgeryseg" / "__init__.py").exists() and str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from forgeryseg.dataset import build_supplemental_index, build_test_index, build_train_index
from forgeryseg.offline import configure_cache_dirs

if CACHE_ROOT is not None:
    configure_cache_dirs(CACHE_ROOT)
    print("[CACHE] using", CACHE_ROOT)

OFFLINE_NO_DOWNLOAD = bool(is_kaggle() and not ALLOW_DOWNLOAD)
if OFFLINE_NO_DOWNLOAD:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("[OFFLINE] downloads disabled (Kaggle submission-safe).")

# %%
# Fase 0 — Sanidade rápida do dataset
if not (DATA_ROOT / "train_images").exists():
    raise FileNotFoundError(
        f"Dataset não encontrado. Verifique DATA_ROOT: {DATA_ROOT} (esperado: .../train_images)."
    )

train_samples = build_train_index(DATA_ROOT, strict=False)
supp_samples = build_supplemental_index(DATA_ROOT, strict=False)
test_samples = build_test_index(DATA_ROOT)

n_auth = sum(1 for s in train_samples if s.is_authentic)
n_forged = sum(1 for s in train_samples if s.is_authentic is False)

print("train/authentic:", n_auth)
print("train/forged:", n_forged)
print("supplemental:", len(supp_samples))
print("test:", len(test_samples))

# %% [markdown]
# ## Fase 1 — Treino do classificador (opcional)
#
# O classificador é usado como **gate** para evitar segmentar imagens
# claramente autênticas. O threshold ótimo (F1) é salvo no checkpoint e
# pode ser usado automaticamente na inferência (`cls_skip_threshold=auto`).
#
# Dica: se você **não** treinar o classificador, desative o gate na inferência
# (`--cls-skip-threshold 0.0`).

# %%
# Fase 1 — Treino do classificador (via script)
cls_config_path = CLS_CONFIG
if OFFLINE_NO_DOWNLOAD:
    cls_config_path = write_config(
        CLS_CONFIG,
        OUTPUTS_ROOT / "configs" / "cls_offline.json",
        {"pretrained": False},
    )

if RUN_TRAIN_CLS:
    train_cls_script = REPO_ROOT / "scripts" / "train_cls_cv.py"
    if not train_cls_script.exists():
        raise FileNotFoundError(f"Não encontrei {train_cls_script}.")

    cmd = [
        sys.executable,
        str(train_cls_script),
        "--config",
        str(cls_config_path),
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(OUTPUTS_ROOT),
        "--folds",
        str(N_FOLDS),
    ]
    if FOLD >= 0:
        cmd += ["--fold", str(FOLD)]
    if CACHE_ROOT is not None:
        cmd += ["--cache-root", str(CACHE_ROOT)]
    run_cmd(cmd)
else:
    print("[CLS] RUN_TRAIN_CLS=False (pulando).")

# %% [markdown]
# ## Fase 2 — Treino do segmentador (opcional)
#
# Aqui treinamos o modelo de segmentação que gera as máscaras de fraude.
# Você pode testar diferentes arquiteturas em `configs/` e combinar tudo
# na inferência com *ensemble* e **TTA**.
#
# Observação: configs com *DINOv2/HF* exigem cache local e `local_files_only=true`
# no Kaggle (internet OFF).

# %%
# Fase 2 — Treino do segmentador (via script)
seg_config_path = SEG_CONFIG
if OFFLINE_NO_DOWNLOAD:
    seg_config_path = write_config(
        SEG_CONFIG,
        OUTPUTS_ROOT / "configs" / "seg_offline.json",
        {"encoder_weights": "", "pretrained": False},
    )

if RUN_TRAIN_SEG:
    train_seg_script = REPO_ROOT / "scripts" / "train_seg_smp_cv.py"
    if not train_seg_script.exists():
        raise FileNotFoundError(f"Não encontrei {train_seg_script}.")

    cmd = [
        sys.executable,
        str(train_seg_script),
        "--config",
        str(seg_config_path),
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(OUTPUTS_ROOT),
        "--folds",
        str(N_FOLDS),
    ]
    if FOLD >= 0:
        cmd += ["--fold", str(FOLD)]
    if CACHE_ROOT is not None:
        cmd += ["--cache-root", str(CACHE_ROOT)]
    run_cmd(cmd)
else:
    print("[SEG] RUN_TRAIN_SEG=False (pulando).")

# %% [markdown]
# ## Fase 3 — Inferência + pós-processamento (TTA + morfologia)
#
# A inferência usa `scripts/submit_ensemble.py` e o arquivo
# `configs/infer_ensemble.json`, que já inclui:
#
# - **TTA**: `none`, `hflip`, `vflip`
# - **Pós-processamento morfológico**: *closing*, *opening*, *fill holes*
# - **Filtro por área/confiança** e threshold adaptativo
#
# Para ajustar, edite `configs/infer_ensemble.json` ou passe flags no CLI.

# %%
# Fase 3 — Gerar submission.csv
infer_config_path = INFER_CONFIG
if OFFLINE_NO_DOWNLOAD and env_bool("FORGERYSEG_DISABLE_HF_MODELS", default=True):
    # Remove modelos que dependem de HuggingFace (ex.: DINOv2) quando não queremos
    # depender de cache HF no Kaggle offline.
    try:
        with INFER_CONFIG.open("r") as f:
            infer_cfg = json.load(f)
        models = infer_cfg.get("models", [])
        filtered = []
        dropped = []
        for m in models:
            mid = str(m.get("model_id", ""))
            if "dinov2" in mid.lower():
                dropped.append(mid)
                continue
            filtered.append(m)
        if filtered and dropped:
            total_w = sum(float(m.get("weight", 1.0)) for m in filtered) or 1.0
            for m in filtered:
                m["weight"] = float(m.get("weight", 1.0)) / total_w
            infer_cfg["models"] = filtered
            infer_config_path = write_config(INFER_CONFIG, OUTPUTS_ROOT / "configs" / "infer_offline.json", infer_cfg)
            print("[INFER] OFFLINE: removendo modelos HF:", ", ".join(dropped))
    except Exception:
        pass

submission_path = Path("/kaggle/working/submission.csv") if is_kaggle() else OUTPUTS_ROOT / "submission.csv"
models_seg_dir = OUTPUTS_ROOT / "models_seg"
models_cls_dir = OUTPUTS_ROOT / "models_cls"

# Recalcula presença de checkpoints após treino
has_cls_now = has_any_ckpt("models_cls", "fold_*/best.pt", OUTPUTS_ROOT)

if RUN_SUBMISSION:
    submit_script = REPO_ROOT / "scripts" / "submit_ensemble.py"
    if not submit_script.exists():
        raise FileNotFoundError(f"Não encontrei {submit_script}.")

    cmd = [
        sys.executable,
        str(submit_script),
        "--data-root",
        str(DATA_ROOT),
        "--out-csv",
        str(submission_path),
        "--config",
        str(infer_config_path),
    ]
    if models_seg_dir.exists():
        cmd += ["--models-dir", str(models_seg_dir)]
    if models_cls_dir.exists():
        cmd += ["--cls-models-dir", str(models_cls_dir)]
    if not has_cls_now:
        cmd += ["--cls-skip-threshold", "0.0"]
        print("[CLS] sem checkpoints -> gate desativado (cls_skip_threshold=0.0).")

    run_cmd(cmd)
    print("submission.csv ->", submission_path)
else:
    print("[SUBMISSION] RUN_SUBMISSION=False (pulando).")
