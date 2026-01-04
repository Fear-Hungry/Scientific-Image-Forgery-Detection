from __future__ import annotations

# %% [markdown]
# # Recod.ai/LUC — Submission (Kaggle)
#
# Gera `submission.csv` no formato da competição a partir de 1+ configs em `configs/*.json`,
# com suporte a:
#
# - Segmentação (DINOv2) + pós-processamento + TTA (via config)
# - `fft_gate` (opcional) para revisar casos `authentic`
# - `dinov2_freq_fusion` (opcional) via `model_type`
# - Ensemble de múltiplas submissões (opcional)
#
# **No Kaggle**:
# 1. Anexe o dataset da competição.
# 2. (Opcional) Anexe um dataset com seus checkpoints em `outputs/models/*.pth`.
# 3. Rode todas as células; o arquivo final fica em `/kaggle/working/submission.csv`.
#
# Observação: por regra do repo, a lógica nasce aqui (`.py`) e é espelhada no `.ipynb`.

# %%
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def _find_code_root() -> Path:
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "src" / "forgeryseg").exists():
            return p

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if not d.is_dir():
                continue
            if (d / "src" / "forgeryseg").exists():
                return d
            # common: dataset root contains a single folder with the repo inside
            try:
                for child in d.iterdir():
                    if child.is_dir() and (child / "src" / "forgeryseg").exists():
                        return child
            except PermissionError:
                continue

    raise FileNotFoundError(
        "Não encontrei o código (src/forgeryseg). "
        "No Kaggle: anexe um Dataset contendo este repo (com pastas src/ e configs/)."
    )


CODE_ROOT = _find_code_root()
SRC = CODE_ROOT / "src"
CONFIG_ROOT = CODE_ROOT / "configs"
print(f"code_root={CODE_ROOT}")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forgeryseg.checkpoint import load_flexible_state_dict
from forgeryseg.dataset import list_cases
from forgeryseg.ensemble import ensemble_annotations, rank_weights_by_score
from forgeryseg.frequency import FFTParams, fft_tensor
from forgeryseg.inference import default_tta, load_rgb, predict_prob_map
from forgeryseg.models.dinov2_decoder import DinoV2EncoderSpec, DinoV2SegmentationModel
from forgeryseg.models.dinov2_freq_fusion import DinoV2FreqFusionSegmentationModel, FreqFusionSpec
from forgeryseg.models.fft_classifier import FFTClassifier
from forgeryseg.postprocess import PostprocessParams, postprocess_prob
from forgeryseg.rle import masks_to_annotation

# %%
# -------------------------
# Config (edite aqui)
# -------------------------

DATA_ROOT: Path | None = None  # None => auto-detect (Kaggle -> local)
SPLIT = "test"  # "test" no Kaggle (train/supplemental só para debug)
LIMIT = 0  # 0 = sem limite
SKIP_MISSING_CONFIGS = True  # se faltar config/ckpt, pula ao invés de quebrar

# 1+ configs para gerar submissões individuais
CONFIG_PATHS = [
    CONFIG_ROOT / "dino_v3_518_r69_fft_gate.json",
    # CONFIG_ROOT / "dino_v2_518_basev1.json",
    # CONFIG_ROOT / "dino_v1_718_u52.json",
    # CONFIG_ROOT / "dino_v3_518_r69_freq_fusion.json",
]

# ensemble (opcional) se CONFIG_PATHS tiver 2+
DO_ENSEMBLE = len(CONFIG_PATHS) > 1
ENSEMBLE_METHOD = "weighted"  # weighted | majority | union | intersection
ENSEMBLE_THRESHOLD = 0.5  # só para method="weighted"

# Se quiser pesos fixos, preencha WEIGHTS (mesmo tamanho de CONFIG_PATHS).
# Caso contrário, se SCORES for fornecido, os pesos vêm de rank_weights_by_score(scores).
WEIGHTS: list[float] | None = None
SCORES: list[float] | None = None

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
FINAL_OUT = OUT_DIR / "submission.csv"

# %%


def _find_recodai_root() -> Path:
    if DATA_ROOT is not None:
        return Path(DATA_ROOT)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if not d.is_dir():
                continue
            if (d / "recodai" / "sample_submission.csv").exists():
                return d / "recodai"
            if (d / "sample_submission.csv").exists() and (d / "test_images").exists():
                return d

    local = Path("data/recodai")
    if local.exists():
        return local
    local2 = CODE_ROOT / "data" / "recodai"
    if local2.exists():
        return local2

    raise FileNotFoundError(
        "Não encontrei o data root. Defina DATA_ROOT manualmente "
        "(ex.: /kaggle/input/<dataset>/recodai ou data/recodai)."
    )


def _resolve_maybe_in_kaggle_input(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if p.is_absolute():
        return p
    cand_local = CODE_ROOT / p
    if cand_local.exists():
        return cand_local

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if not d.is_dir():
                continue
            cand = d / p
            if cand.exists():
                return cand
    return p


def _warn_state_dict(missing: list[str], unexpected: list[str]) -> None:
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def _load_segmentation_model(cfg: dict, *, device: torch.device) -> torch.nn.Module:
    enc_cfg = cfg.get("encoder", {})
    encoder = DinoV2EncoderSpec(
        model_name=enc_cfg.get("model_name", "vit_base_patch14_dinov2"),
        checkpoint_path=enc_cfg.get("checkpoint_path"),
    )

    model_type = str(cfg.get("model_type", "dinov2"))
    if model_type == "dinov2_freq_fusion":
        freq = FreqFusionSpec(**cfg.get("freq_fusion", {}))
        model = DinoV2FreqFusionSegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.get("decoder_hidden_channels", 256)),
            decoder_dropout=float(cfg.get("decoder_dropout", 0.0)),
            freeze_encoder=bool(cfg.get("freeze_encoder", True)),
            freq=freq,
        )
    else:
        model = DinoV2SegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.get("decoder_hidden_channels", 256)),
            decoder_dropout=float(cfg.get("decoder_dropout", 0.0)),
            freeze_encoder=bool(cfg.get("freeze_encoder", True)),
        )

    ckpt = cfg.get("checkpoint")
    if ckpt:
        ckpt_path = _resolve_maybe_in_kaggle_input(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt} (tentado: {ckpt_path})")
        missing, unexpected = load_flexible_state_dict(model, ckpt_path)
        _warn_state_dict(missing, unexpected)

    return model.to(device).eval()


def _load_fft_gate(cfg: dict, *, device: torch.device) -> tuple[FFTClassifier, FFTParams, float] | None:
    fft_gate = cfg.get("fft_gate")
    if not isinstance(fft_gate, dict) or not fft_gate.get("enabled", True):
        return None

    fft_ckpt = fft_gate.get("checkpoint")
    if not fft_ckpt:
        raise ValueError("fft_gate.enabled is true but fft_gate.checkpoint is missing")
    fft_ckpt_path = _resolve_maybe_in_kaggle_input(fft_ckpt)
    if not fft_ckpt_path.exists():
        raise FileNotFoundError(f"FFT checkpoint não encontrado: {fft_ckpt} (tentado: {fft_ckpt_path})")

    fft_params = FFTParams(**fft_gate.get("fft", {}))
    fft_threshold = float(fft_gate.get("threshold", 0.5))
    fft_model = FFTClassifier(
        backbone=fft_gate.get("backbone", "resnet18"),
        in_chans=1,
        dropout=float(fft_gate.get("dropout", 0.0)),
    )
    missing, unexpected = load_flexible_state_dict(fft_model, fft_ckpt_path)
    _warn_state_dict(missing, unexpected)
    return fft_model.to(device).eval(), fft_params, fft_threshold


@torch.no_grad()
def predict_submission_for_config(
    *,
    config_path: Path,
    data_root: Path,
    split: str,
    out_path: Path,
    device: torch.device,
    limit: int = 0,
) -> None:
    cfg = json.loads(config_path.read_text())
    input_size = int(cfg["input_size"])
    post = PostprocessParams(**cfg.get("postprocess", {}))

    tta_cfg = cfg.get("tta", {})
    tta_transforms, tta_weights = default_tta(
        zoom_scale=float(tta_cfg.get("zoom_scale", 0.9)),
        weights=tuple(tta_cfg.get("weights", [0.5, 0.25, 0.25])),
    )

    model = _load_segmentation_model(cfg, device=device)
    fft_gate = _load_fft_gate(cfg, device=device)

    cases = list_cases(data_root, split, include_authentic=True, include_forged=True)
    if split == "test":
        # use sample_submission.csv como fonte de case_id (ordem/coverage)
        sample_path = data_root / "sample_submission.csv"
        sample = pd.read_csv(sample_path)
        sample["case_id"] = sample["case_id"].astype(str)
        case_ids = sample["case_id"].tolist()
    else:
        case_ids = [c.case_id for c in cases]

    case_by_id = {c.case_id: c for c in cases}
    if split == "test":
        missing = [cid for cid in case_ids if cid not in case_by_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)} case_id(s) do sample_submission não foram encontrados em {split}_images "
                f"(ex.: {missing[:5]}). Verifique o DATA_ROOT."
            )
        ordered = [case_by_id[cid] for cid in case_ids]
    else:
        ordered = cases
    if limit and limit > 0:
        ordered = ordered[: int(limit)]

    rows: list[dict[str, str]] = []
    n_fft_overrides = 0
    for case in tqdm(ordered, desc=f"Predict ({config_path.name})"):
        image = load_rgb(case.image_path)

        prob = predict_prob_map(
            model,
            image,
            input_size=input_size,
            device=device,
            tta_transforms=tta_transforms,
            tta_weights=tta_weights,
        )
        instances = postprocess_prob(prob, post)
        ann = masks_to_annotation(instances)

        if ann == "authentic" and fft_gate is not None:
            fft_model, fft_params, fft_threshold = fft_gate
            x_fft = fft_tensor(image, fft_params).unsqueeze(0).to(device)
            p_forged = float(torch.sigmoid(fft_model(x_fft))[0].detach().cpu().item())

            if p_forged >= float(fft_threshold):
                relaxed_post = dataclasses.replace(post, authentic_area_max=None, authentic_conf_max=None)
                instances_relaxed = postprocess_prob(prob, relaxed_post)
                ann_relaxed = masks_to_annotation(instances_relaxed)
                if ann_relaxed != "authentic":
                    ann = ann_relaxed
                    n_fft_overrides += 1

        rows.append({"case_id": case.case_id, "annotation": ann})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    n_auth = int((out_df["annotation"] == "authentic").sum())
    print(f"Wrote {out_path} ({n_auth}/{len(out_df)} authentic)")
    if fft_gate is not None:
        print(f"fft_gate overrides: {n_fft_overrides}")


def ensemble_submissions_from_csvs(
    *,
    sub_paths: list[Path],
    data_root: Path,
    split: str,
    out_path: Path,
    method: str = "weighted",
    weights: list[float] | None = None,
    scores: list[float] | None = None,
    threshold: float = 0.5,
) -> None:
    tables: list[dict[str, str]] = []
    for p in sub_paths:
        df = pd.read_csv(p)
        if "case_id" not in df.columns or "annotation" not in df.columns:
            raise ValueError(f"{p} precisa ter colunas case_id,annotation")
        tables.append(dict(zip(df["case_id"].astype(str), df["annotation"], strict=True)))

    if method == "weighted":
        if weights is None:
            if scores is None:
                weights = [1.0 / len(sub_paths)] * len(sub_paths)
            else:
                weights = rank_weights_by_score(scores)
        if len(weights) != len(sub_paths):
            raise ValueError("weights precisa ter o mesmo tamanho de sub_paths")
        print(f"ensemble weights={weights}")

    cases = list_cases(data_root, split, include_authentic=True, include_forged=True)
    if split == "test":
        # respeitar order do sample_submission
        sample = pd.read_csv(data_root / "sample_submission.csv")
        sample["case_id"] = sample["case_id"].astype(str)
        case_by_id = {c.case_id: c for c in cases}
        case_ids = sample["case_id"].tolist()
        missing = [cid for cid in case_ids if cid not in case_by_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)} case_id(s) do sample_submission não foram encontrados em {split}_images "
                f"(ex.: {missing[:5]}). Verifique o DATA_ROOT."
            )
        cases = [case_by_id[cid] for cid in case_ids]

    import cv2

    rows: list[dict[str, str]] = []
    for case in tqdm(cases, desc="Ensemble"):
        h, w = cv2.imread(str(case.image_path), cv2.IMREAD_UNCHANGED).shape[:2]
        anns = [t.get(case.case_id, "authentic") for t in tables]
        ann_out = ensemble_annotations(
            anns,
            shape=(h, w),
            method=method,  # type: ignore[arg-type]
            weights=weights if method == "weighted" else None,
            threshold=float(threshold),
        )
        rows.append({"case_id": case.case_id, "annotation": ann_out})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    n_auth = int((out_df["annotation"] == "authentic").sum())
    print(f"Wrote {out_path} ({n_auth}/{len(out_df)} authentic)")


# %%
# -------------------------
# Run
# -------------------------

data_root = _find_recodai_root()
print(f"data_root={data_root}")
print(f"configs={[p.as_posix() for p in CONFIG_PATHS]}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True

sub_paths: list[Path] = []
for cfg_path in CONFIG_PATHS:
    if not cfg_path.exists():
        msg = f"[warn] Config não encontrado: {cfg_path}"
        if SKIP_MISSING_CONFIGS:
            print(msg)
            continue
        raise FileNotFoundError(msg)

    cfg = json.loads(cfg_path.read_text())
    name = str(cfg.get("name", cfg_path.stem))
    out_path = OUT_DIR / f"submission_{name}.csv"
    try:
        predict_submission_for_config(
            config_path=cfg_path,
            data_root=data_root,
            split=SPLIT,
            out_path=out_path,
            device=device,
            limit=LIMIT,
        )
        sub_paths.append(out_path)
    except FileNotFoundError as e:
        if SKIP_MISSING_CONFIGS:
            print(f"[warn] {e} (pulando {cfg_path.name})")
            continue
        raise

if not sub_paths:
    raise RuntimeError("Nenhuma submissão foi gerada (verifique configs/checkpoints).")

if DO_ENSEMBLE and len(sub_paths) > 1:
    ensemble_submissions_from_csvs(
        sub_paths=sub_paths,
        data_root=data_root,
        split=SPLIT,
        out_path=FINAL_OUT,
        method=str(ENSEMBLE_METHOD),
        weights=WEIGHTS,
        scores=SCORES,
        threshold=float(ENSEMBLE_THRESHOLD),
    )
else:
    # 1 config => apenas renomeia como submission.csv
    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    if sub_paths:
        Path(sub_paths[0]).replace(FINAL_OUT)
    print(f"Wrote {FINAL_OUT}")
