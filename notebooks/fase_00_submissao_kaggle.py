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
# # Fase 00 — Submissão (Kaggle)
#
# Notebook **enxuto** para gerar `submission.csv` no Kaggle (preferencialmente com **internet OFF**).
#
# Ele assume que você já tem checkpoints em `outputs/models_seg` (e opcionalmente `outputs/models_cls`),
# normalmente gerados no **Fase 01** (pré-treinamento) e anexados como Kaggle Dataset.
#
# Saída:
# - Kaggle: `/kaggle/working/submission.csv`
# - Local: `outputs/submission.csv`

# %%
import json
import os
import subprocess
import sys
from pathlib import Path


# %%
# Helpers de ambiente (mínimos)


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    return str(default) if value == "" else str(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name, "").strip().lower()
    if value == "":
        return bool(default)
    return value in {"1", "true", "yes", "y", "on"}


def env_path(name: str) -> Path | None:
    value = env_str(name, "").strip()
    return Path(value) if value else None


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
    for cand in [here] + list(here.parents):
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
    if explicit is not None and (explicit / "train_images").exists():
        return explicit

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


def _missing_modules(mod_names: list[str]) -> list[str]:
    missing: list[str] = []
    for name in mod_names:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    return missing


def maybe_install_from_wheels(wheels_root: Path | None) -> None:
    """Instala apenas o que estiver faltando, via wheels locais (offline-safe)."""
    if wheels_root is None:
        return

    module_to_pip = {
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "timm": "timm",
        "albumentations": "albumentations",
        "huggingface_hub": "huggingface-hub",
        "safetensors": "safetensors",
        "tqdm": "tqdm",
    }
    missing = _missing_modules(list(module_to_pip.keys()))
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


def write_config(base_cfg: Path, out_path: Path, overrides: dict) -> Path:
    with base_cfg.open("r") as f:
        cfg = json.load(f)
    cfg.update(overrides)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    return out_path


# %%
# Setup Kaggle offline + dependências

KAGGLE = is_kaggle()
ALLOW_DOWNLOAD = env_bool("FORGERYSEG_ALLOW_DOWNLOAD", default=not KAGGLE)
OFFLINE = bool(KAGGLE and not ALLOW_DOWNLOAD)

INSTALL_WHEELS = env_bool("FORGERYSEG_INSTALL_WHEELS", default=KAGGLE and OFFLINE)
WHEELS_ROOT = find_wheels_root() if INSTALL_WHEELS else None
if INSTALL_WHEELS:
    if WHEELS_ROOT is None:
        print("[wheels] nenhum wheel root encontrado (ok se já tiver as libs no ambiente).")
    else:
        maybe_install_from_wheels(WHEELS_ROOT)

# Checagem mínima de deps críticas (falha cedo, com mensagem clara)
required_modules = ["torch", "numpy", "cv2", "timm", "segmentation_models_pytorch"]
missing = _missing_modules(required_modules)
if missing:
    raise ImportError(
        "Dependências Python faltando: "
        + ", ".join(missing)
        + "\n- No Kaggle offline: anexe um Dataset com wheels em `.../recodai_bundle/wheels/` e rode com "
        + "`FORGERYSEG_INSTALL_WHEELS=1`."
    )

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)


# %%
# Paths (repo, dados, cache)

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

# Onde escrever artefatos (sempre gravável no Kaggle)
WORK_DIR = Path("/kaggle/working") if KAGGLE else REPO_ROOT
OUTPUTS_ROOT = (WORK_DIR / "outputs").resolve()
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

print("REPO_ROOT:", REPO_ROOT)
print("DATA_ROOT:", DATA_ROOT)
print("OUTPUTS_ROOT:", OUTPUTS_ROOT)

# Cache offline (opcional). Ex.: /kaggle/input/<dataset>/weights_cache
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from forgeryseg.offline import configure_cache_dirs

CACHE_ROOT = find_cache_root()
if CACHE_ROOT is not None:
    configure_cache_dirs(CACHE_ROOT)
    print("[CACHE] using", CACHE_ROOT)

if OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("[OFFLINE] downloads disabled (Kaggle submission-safe).")


# %%
# Config de inferência (edite se quiser)

INFER_CONFIG = REPO_ROOT / "configs" / "infer_ensemble.json"
if not INFER_CONFIG.exists():
    raise FileNotFoundError(f"Config não encontrado: {INFER_CONFIG}")

# Se você não empacotou cache HF, é melhor remover modelos que dependem do HuggingFace no Kaggle offline.
DISABLE_HF_MODELS = env_bool("FORGERYSEG_DISABLE_HF_MODELS", default=True)

infer_config_path = INFER_CONFIG
if OFFLINE and DISABLE_HF_MODELS:
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
    except Exception as exc:
        print("[warn] não consegui filtrar modelos HF:", exc)


# %%
# Gerar submission.csv

submit_script = REPO_ROOT / "scripts" / "submit_ensemble.py"
if not submit_script.exists():
    raise FileNotFoundError(f"Não encontrei {submit_script}.")

submission_path = Path("/kaggle/working/submission.csv") if KAGGLE else (OUTPUTS_ROOT / "submission.csv")

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

run_cmd(cmd)
print("submission.csv ->", submission_path)
