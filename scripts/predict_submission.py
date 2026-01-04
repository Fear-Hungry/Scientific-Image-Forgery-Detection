from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from forgeryseg.checkpoint import load_flexible_state_dict
from forgeryseg.dataset import list_cases
from forgeryseg.frequency import FFTParams, fft_tensor
from forgeryseg.inference import default_tta, load_rgb, predict_prob_map
from forgeryseg.models.fft_classifier import FFTClassifier
from forgeryseg.models.dinov2_decoder import DinoV2EncoderSpec, DinoV2SegmentationModel
from forgeryseg.models.dinov2_freq_fusion import DinoV2FreqFusionSegmentationModel, FreqFusionSpec
from forgeryseg.postprocess import PostprocessParams, postprocess_prob
from forgeryseg.rle import masks_to_annotation


def _warn_state_dict(missing: list[str], unexpected: list[str]) -> None:
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "test", "supplemental"], default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    input_size = int(cfg["input_size"])

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
        missing, unexpected = load_flexible_state_dict(model, Path(ckpt))
        _warn_state_dict(missing, unexpected)

    fft_gate = cfg.get("fft_gate")
    fft_model: FFTClassifier | None = None
    fft_params: FFTParams | None = None
    fft_threshold = 0.5
    if isinstance(fft_gate, dict) and fft_gate.get("enabled", True):
        fft_ckpt = fft_gate.get("checkpoint")
        if not fft_ckpt:
            raise ValueError("fft_gate.enabled is true but fft_gate.checkpoint is missing")

        fft_params = FFTParams(**fft_gate.get("fft", {}))
        fft_threshold = float(fft_gate.get("threshold", 0.5))
        fft_model = FFTClassifier(
            backbone=fft_gate.get("backbone", "resnet18"),
            in_chans=1,
            dropout=float(fft_gate.get("dropout", 0.0)),
        )
        missing, unexpected = load_flexible_state_dict(fft_model, Path(fft_ckpt))
        _warn_state_dict(missing, unexpected)

    tta_cfg = cfg.get("tta", {})
    tta_transforms, tta_weights = default_tta(
        zoom_scale=float(tta_cfg.get("zoom_scale", 0.9)),
        weights=tuple(tta_cfg.get("weights", [0.5, 0.25, 0.25])),
    )
    post = PostprocessParams(**cfg.get("postprocess", {}))
    device = torch.device(args.device)
    if fft_model is not None:
        fft_model = fft_model.to(device)
        fft_model.eval()

    cases = list_cases(args.data_root, args.split, include_authentic=True, include_forged=True)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    rows: list[dict[str, str]] = []
    n_fft_overrides = 0
    for case in tqdm(cases, desc="Predict"):
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

        if ann == "authentic" and fft_model is not None and fft_params is not None:
            x_fft = fft_tensor(image, fft_params).unsqueeze(0).to(device)
            with torch.no_grad():
                p_forged = torch.sigmoid(fft_model(x_fft))[0].detach().cpu().item()

            if float(p_forged) >= float(fft_threshold):
                relaxed_post = dataclasses.replace(post, authentic_area_max=None, authentic_conf_max=None)
                instances_relaxed = postprocess_prob(prob, relaxed_post)
                ann_relaxed = masks_to_annotation(instances_relaxed)
                if ann_relaxed != "authentic":
                    ann = ann_relaxed
                    n_fft_overrides += 1

        rows.append({"case_id": case.case_id, "annotation": ann})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    n_auth = sum(1 for r in rows if r["annotation"] == "authentic")
    print(f"Wrote {args.out} ({n_auth}/{len(rows)} authentic)")
    if fft_model is not None:
        print(f"fft_gate overrides: {n_fft_overrides}")


if __name__ == "__main__":
    main()
