from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .checkpoint import load_flexible_state_dict
from .config import FFTGateConfig, SegmentationModelConfig, load_segmentation_config
from .dataset import list_cases
from .frequency import FFTParams, fft_tensor
from .inference import TilingParams, default_tta, load_rgb, predict_prob_map, predict_prob_map_tiled
from .models.dinov2_decoder import DinoV2EncoderSpec, DinoV2SegmentationModel
from .models.dinov2_freq_fusion import DinoV2FreqFusionSegmentationModel, FreqFusionSpec
from .models.fft_classifier import FFTClassifier
from .paths import resolve_existing_path
from .postprocess import PostprocessParams, postprocess_prob
from .rle import masks_to_annotation
from .typing import Pathish, Split


def _warn_state_dict(missing: list[str], unexpected: list[str]) -> None:
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


@dataclass(frozen=True)
class SubmissionStats:
    out_path: Path
    n_rows: int
    n_authentic: int
    fft_gate_overrides: int = 0


def list_ordered_cases(data_root: Path, split: Split) -> list:
    """
    List cases for a split.

    For `split="test"`, if `sample_submission.csv` exists, it is used to define the expected
    `case_id` set and ordering.
    """
    cases = list_cases(data_root, split, include_authentic=True, include_forged=True)
    if split != "test":
        return cases

    sample_path = data_root / "sample_submission.csv"
    if not sample_path.exists():
        return cases

    sample = pd.read_csv(sample_path)
    if "case_id" not in sample.columns:
        return cases
    case_ids = sample["case_id"].astype(str).tolist()
    case_by_id = {c.case_id: c for c in cases}
    missing = [cid for cid in case_ids if cid not in case_by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} case_id(s) do sample_submission não foram encontrados em {split}_images "
            f"(ex.: {missing[:5]}). Verifique o data_root."
        )
    return [case_by_id[cid] for cid in case_ids]


def _load_segmentation_model(
    cfg: SegmentationModelConfig,
    *,
    device: torch.device,
    path_roots: list[Path] | None = None,
) -> torch.nn.Module:
    ckpt = cfg.checkpoint
    enc_cfg = cfg.encoder
    enc_ckpt = enc_cfg.checkpoint_path
    if enc_ckpt:
        enc_ckpt = str(resolve_existing_path(str(enc_ckpt), roots=path_roots, search_kaggle_input=True))
    encoder = DinoV2EncoderSpec(
        model_name=str(enc_cfg.model_name),
        checkpoint_path=enc_ckpt,
        pretrained=bool(enc_cfg.pretrained) and not bool(ckpt),
    )

    model_type = str(cfg.type)
    if model_type == "dinov2_freq_fusion":
        freq = FreqFusionSpec(**cfg.freq_fusion)
        model = DinoV2FreqFusionSegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.decoder_hidden_channels),
            decoder_dropout=float(cfg.decoder_dropout),
            freeze_encoder=bool(cfg.freeze_encoder),
            freq=freq,
        )
    else:
        model = DinoV2SegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.decoder_hidden_channels),
            decoder_dropout=float(cfg.decoder_dropout),
            freeze_encoder=bool(cfg.freeze_encoder),
        )

    if ckpt:
        ckpt_path = resolve_existing_path(str(ckpt), roots=path_roots, search_kaggle_input=True)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt} (tentado: {ckpt_path})")
        missing, unexpected = load_flexible_state_dict(model, ckpt_path)
        _warn_state_dict(missing, unexpected)

    return model.to(device).eval()


def _load_fft_gate(
    fft_gate: FFTGateConfig | None,
    *,
    device: torch.device,
    path_roots: list[Path] | None = None,
) -> tuple[FFTClassifier, FFTParams, float] | None:
    if fft_gate is None or not bool(fft_gate.enabled):
        return None

    fft_ckpt = fft_gate.checkpoint
    if not fft_ckpt:
        raise ValueError("fft_gate.enabled is true but fft_gate.checkpoint is missing")
    fft_ckpt_path = resolve_existing_path(str(fft_ckpt), roots=path_roots, search_kaggle_input=True)
    if not fft_ckpt_path.exists():
        raise FileNotFoundError(f"FFT checkpoint não encontrado: {fft_ckpt} (tentado: {fft_ckpt_path})")

    fft_percentiles = tuple(float(x) for x in fft_gate.fft.normalize_percentiles)
    if len(fft_percentiles) != 2:
        raise ValueError("fft_gate.fft.normalize_percentiles must have 2 values")
    fft_params = FFTParams(
        mode=fft_gate.fft.mode,  # type: ignore[arg-type]
        input_size=int(fft_gate.fft.input_size),
        hp_radius_fraction=float(fft_gate.fft.hp_radius_fraction),
        normalize_percentiles=fft_percentiles,  # type: ignore[arg-type]
    )
    fft_threshold = float(fft_gate.threshold)
    fft_model = FFTClassifier(
        backbone=str(fft_gate.backbone),
        in_chans=1,
        dropout=float(fft_gate.dropout),
    )
    missing, unexpected = load_flexible_state_dict(fft_model, fft_ckpt_path)
    _warn_state_dict(missing, unexpected)
    return fft_model.to(device).eval(), fft_params, fft_threshold


def _load_tiling(cfg: object) -> TilingParams | None:
    if not isinstance(cfg, dict):
        return None
    tile_size = int(cfg.get("tile_size", 0))
    if tile_size <= 0:
        return None
    return TilingParams(
        tile_size=tile_size,
        overlap=int(cfg.get("overlap", 0)),
        batch_size=int(cfg.get("batch_size", 4)),
    )


def write_submission_csv(
    *,
    config_path: Pathish,
    data_root: Pathish,
    split: Split = "test",
    out_path: Pathish,
    device: torch.device,
    limit: int = 0,
    overrides: list[str] | None = None,
    path_roots: list[Path] | None = None,
) -> SubmissionStats:
    config_path = Path(config_path)
    data_root = Path(data_root)
    out_path = Path(out_path)

    cfg = load_segmentation_config(config_path, overrides=overrides)
    input_size = int(cfg.model.input_size)
    post = PostprocessParams(**dataclasses.asdict(cfg.inference.postprocess))

    tta_transforms, tta_weights = default_tta(
        zoom_scale=float(cfg.inference.tta.zoom_scale),
        weights=tuple(cfg.inference.tta.weights),
    )
    tiling = _load_tiling(dataclasses.asdict(cfg.inference.tiling) if cfg.inference.tiling is not None else None)

    # ensure config-relative paths work out of the box
    if path_roots is None:
        path_roots = [config_path.parent, Path.cwd()]

    model = _load_segmentation_model(cfg.model, device=device, path_roots=path_roots)
    fft_gate = _load_fft_gate(cfg.inference.fft_gate, device=device, path_roots=path_roots)

    ordered = list_ordered_cases(data_root, split)
    if limit and limit > 0:
        ordered = ordered[: int(limit)]

    rows: list[dict[str, str]] = []
    n_fft_overrides = 0
    for case in tqdm(ordered, desc=f"Predict ({config_path.name})"):
        image = load_rgb(case.image_path)

        if tiling is None:
            prob = predict_prob_map(
                model,
                image,
                input_size=input_size,
                device=device,
                tta_transforms=tta_transforms,
                tta_weights=tta_weights,
            )
        else:
            prob = predict_prob_map_tiled(
                model,
                image,
                input_size=input_size,
                device=device,
                tiling=tiling,
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

    return SubmissionStats(
        out_path=out_path,
        n_rows=int(len(out_df)),
        n_authentic=n_auth,
        fft_gate_overrides=int(n_fft_overrides),
    )
