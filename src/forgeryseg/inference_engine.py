from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .checkpoint import load_flexible_state_dict, warn_state_dict
from .config import SegmentationModelConfig, load_segmentation_config
from .fft_gate import FFTGate
from .image import LetterboxMeta, letterbox_reflect, unletterbox
from .inference import TilingParams, build_tta, load_rgb, predict_prob_map_tiled
from .models.dinov2_decoder import DinoV2EncoderSpec, DinoV2SegmentationModel
from .models.dinov2_freq_fusion import DinoV2FreqFusionSegmentationModel, FreqFusionSpec
from .models.dinov2_multiscale import DinoV2MultiScaleSegmentationModel, MultiScaleSpec
from .paths import resolve_existing_path
from .postprocess import PostprocessParams, postprocess_prob
from .rle import masks_to_annotation
from .tta import IdentityTTA, TTATransform, predict_with_tta
from .typing import Pathish


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
        enc_ckpt = str(resolve_existing_path(enc_ckpt, roots=path_roots, search_kaggle_input=True))

    encoder = DinoV2EncoderSpec(
        model_name=str(enc_cfg.model_name),
        checkpoint_path=enc_ckpt,
        pretrained=bool(enc_cfg.pretrained) and not bool(ckpt),
    )

    if cfg.type == "dinov2_freq_fusion":
        freq = FreqFusionSpec(**cfg.freq_fusion)
        model: torch.nn.Module = DinoV2FreqFusionSegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.decoder_hidden_channels),
            decoder_dropout=float(cfg.decoder_dropout),
            freeze_encoder=bool(cfg.freeze_encoder),
            freq=freq,
        )
    elif cfg.type == "dinov2_multiscale":
        multiscale = MultiScaleSpec(**cfg.multiscale)
        model = DinoV2MultiScaleSegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.decoder_hidden_channels),
            decoder_dropout=float(cfg.decoder_dropout),
            freeze_encoder=bool(cfg.freeze_encoder),
            multiscale=multiscale,
        )
    else:
        model = DinoV2SegmentationModel(
            encoder,
            decoder_hidden_channels=int(cfg.decoder_hidden_channels),
            decoder_dropout=float(cfg.decoder_dropout),
            freeze_encoder=bool(cfg.freeze_encoder),
        )

    if ckpt:
        ckpt_rel = Path(ckpt)
        ckpt_path = resolve_existing_path(ckpt_rel, roots=path_roots, search_kaggle_input=True)

        def _pick_best_by_val_of1(paths: list[Path]) -> Path | None:
            best_p: Path | None = None
            best_s = float("-inf")
            best_mtime = float("-inf")
            for p in paths:
                if not p.exists():
                    continue
                try:
                    d = torch.load(p, map_location="cpu")
                    v = d.get("val_of1", None)
                    s = float(v) if isinstance(v, (int, float)) else float("-inf")
                except Exception:
                    s = float("-inf")
                try:
                    mtime = float(p.stat().st_mtime)
                except Exception:
                    mtime = float("-inf")

                if (s > best_s) or (s == best_s and mtime > best_mtime):
                    best_p = p
                    best_s = s
                    best_mtime = mtime
            return best_p

        if not ckpt_path.exists():
            # Fallback 1: try "<stem>_last.pth"
            ckpt_last_rel = ckpt_rel.with_name(f"{ckpt_rel.stem}_last{ckpt_rel.suffix}")
            ckpt_last_path = resolve_existing_path(ckpt_last_rel, roots=path_roots, search_kaggle_input=True)
            if ckpt_last_path.exists():
                print(f"[warn] Checkpoint não encontrado: {ckpt_rel} (usando fallback: {ckpt_last_path})")
                ckpt_path = ckpt_last_path
            else:
                # Fallback 2: if base ckpt missing, try best fold checkpoint (e.g., <stem>_fold{i}.pth)
                if "_fold" not in ckpt_rel.stem:
                    pattern = f"{ckpt_rel.stem}_fold*{ckpt_rel.suffix}"
                    candidates: list[Path] = []
                    parent_rel = ckpt_rel.parent

                    if path_roots is not None:
                        for root in path_roots:
                            base = Path(root) / parent_rel
                            if base.exists():
                                candidates.extend(base.glob(pattern))

                    kaggle_input = Path("/kaggle/input")
                    if kaggle_input.exists():
                        for d in kaggle_input.iterdir():
                            if not d.is_dir():
                                continue
                            base = d / parent_rel
                            if base.exists():
                                candidates.extend(base.glob(pattern))
                            try:
                                for child in d.iterdir():
                                    if not child.is_dir():
                                        continue
                                    base2 = child / parent_rel
                                    if base2.exists():
                                        candidates.extend(base2.glob(pattern))
                            except PermissionError:
                                continue

                    best = _pick_best_by_val_of1(candidates)
                    if best is not None and best.exists():
                        print(f"[warn] Checkpoint não encontrado: {ckpt_rel} (usando melhor fold: {best})")
                        ckpt_path = best

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt} (tentado: {ckpt_path})")

        missing, unexpected = load_flexible_state_dict(model, ckpt_path)
        warn_state_dict(missing, unexpected)

    return model.to(device).eval()


@dataclass(frozen=True)
class InferenceEngine:
    model: torch.nn.Module
    device: torch.device
    input_size: int
    postprocess: PostprocessParams
    tta_transforms: list[TTATransform]
    tta_weights: list[float]
    tiling: TilingParams | None = None
    batch_size: int = 1
    fft_gate: FFTGate | None = None
    amp: bool = False

    @classmethod
    def from_config(
        cls,
        *,
        config_path: Pathish,
        device: torch.device,
        overrides: list[str] | None = None,
        path_roots: list[Path] | None = None,
        amp: bool = False,
    ) -> InferenceEngine:
        config_path = Path(config_path)
        cfg = load_segmentation_config(config_path, overrides=overrides)

        if path_roots is None:
            path_roots = [config_path.parent, Path.cwd()]

        model = _load_segmentation_model(cfg.model, device=device, path_roots=path_roots)

        post = PostprocessParams(**dataclasses.asdict(cfg.inference.postprocess))
        tta_transforms, tta_weights = build_tta(
            modes=cfg.inference.tta.modes,
            zoom_scale=float(cfg.inference.tta.zoom_scale),
            zoom_in_scale=float(cfg.inference.tta.zoom_in_scale),
            weights=list(cfg.inference.tta.weights),
        )

        tiling = None
        if cfg.inference.tiling is not None and int(cfg.inference.tiling.tile_size) > 0:
            tiling = TilingParams(
                tile_size=int(cfg.inference.tiling.tile_size),
                overlap=int(cfg.inference.tiling.overlap),
                batch_size=int(cfg.inference.tiling.batch_size),
            )

        fft_gate = (
            FFTGate.from_config(cfg.inference.fft_gate, device=device, path_roots=path_roots)
            if cfg.inference.fft_gate
            else None
        )

        return cls(
            model=model,
            device=device,
            input_size=int(cfg.model.input_size),
            postprocess=post,
            tta_transforms=tta_transforms,
            tta_weights=tta_weights,
            tiling=tiling,
            batch_size=int(max(1, cfg.inference.batch_size)),
            fft_gate=fft_gate,
            amp=bool(amp),
        )

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Predict a probability map (H, W) in [0,1] for a single RGB image.
        """
        if self.tiling is not None:
            return predict_prob_map_tiled(
                self.model,
                image_rgb,
                input_size=int(self.input_size),
                device=self.device,
                tiling=self.tiling,
                tta_transforms=self.tta_transforms,
                tta_weights=self.tta_weights,
            )
        return self._predict_prob_maps_batched([image_rgb])[0]

    def predict_path(self, image_path: Pathish) -> np.ndarray:
        return self.predict(load_rgb(image_path))

    def predict_batch(self, image_paths: Sequence[Pathish]) -> list[list[np.ndarray]]:
        """
        Predict instance masks for multiple images (postprocess + optional fft_gate).

        Returns one list of instance masks per input image.
        """
        paths = [Path(p) for p in image_paths]
        out: list[list[np.ndarray]] = []
        if self.tiling is not None:
            for p in paths:
                image = load_rgb(p)
                instances, _ = self.predict_instances(image)
                out.append(instances)
            return out

        bs = int(max(1, self.batch_size))
        for i in range(0, len(paths), bs):
            chunk_paths = paths[i : i + bs]
            images = [load_rgb(p) for p in chunk_paths]
            probs = self._predict_prob_maps_batched(images)
            for image, prob in zip(images, probs, strict=True):
                instances, _ = self.predict_instances(image, prob_map=prob)
                out.append(instances)
        return out

    def predict_batch_with_overrides(self, image_paths: Sequence[Pathish]) -> list[tuple[list[np.ndarray], bool]]:
        """
        Same as predict_batch, but also returns a flag indicating whether fft_gate overrode the result.
        """
        paths = [Path(p) for p in image_paths]
        out: list[tuple[list[np.ndarray], bool]] = []
        if self.tiling is not None:
            for p in paths:
                image = load_rgb(p)
                out.append(self.predict_instances(image))
            return out

        bs = int(max(1, self.batch_size))
        for i in range(0, len(paths), bs):
            chunk_paths = paths[i : i + bs]
            images = [load_rgb(p) for p in chunk_paths]
            probs = self._predict_prob_maps_batched(images)
            for image, prob in zip(images, probs, strict=True):
                out.append(self.predict_instances(image, prob_map=prob))
        return out

    def predict_annotations(self, image_paths: Sequence[Pathish]) -> tuple[list[str], int]:
        pairs = self.predict_batch_with_overrides(image_paths)
        anns = [masks_to_annotation(instances) for instances, _ in pairs]
        n_overrides = sum(1 for _, overridden in pairs if overridden)
        return anns, int(n_overrides)

    def predict_annotation(self, image_rgb: np.ndarray) -> tuple[str, bool]:
        instances, overridden = self.predict_instances(image_rgb)
        return masks_to_annotation(instances), overridden

    def predict_annotation_path(self, image_path: Pathish) -> tuple[str, bool]:
        return self.predict_annotation(load_rgb(image_path))

    def predict_instances(
        self, image_rgb: np.ndarray, *, prob_map: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], bool]:
        if prob_map is None:
            prob_map = self.predict(image_rgb)
        instances = postprocess_prob(prob_map, self.postprocess)
        return self._maybe_apply_fft_gate(image_rgb, prob_map, instances)

    def _maybe_apply_fft_gate(
        self, image_rgb: np.ndarray, prob_map: np.ndarray, instances: list[np.ndarray]
    ) -> tuple[list[np.ndarray], bool]:
        if self.fft_gate is None:
            return instances, False
        if instances:
            return instances, False

        p_forged = self.fft_gate.predict_prob_forged(image_rgb)
        if p_forged < float(self.fft_gate.threshold):
            return instances, False

        relaxed_post = dataclasses.replace(self.postprocess, authentic_area_max=None, authentic_conf_max=None)
        relaxed_instances = postprocess_prob(prob_map, relaxed_post)
        if not relaxed_instances:
            return instances, False
        return relaxed_instances, True

    def _predict_prob_maps_batched(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        if len(images) == 0:
            return []

        padded: list[np.ndarray] = []
        metas: list[LetterboxMeta] = []
        for img in images:
            pad, meta = letterbox_reflect(img, int(self.input_size))
            padded.append(pad)
            metas.append(meta)

        x = torch.stack(
            [torch.from_numpy(im).permute(2, 0, 1).contiguous().float() / 255.0 for im in padded],
            dim=0,
        ).to(self.device)

        transforms = self.tta_transforms or [IdentityTTA()]
        weights = self.tta_weights or [1.0]

        with torch.no_grad():
            if self.amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prob = predict_with_tta(self.model, x, transforms=transforms, weights=weights)
            else:
                prob = predict_with_tta(self.model, x, transforms=transforms, weights=weights)

        prob_np = prob[:, 0].detach().cpu().numpy().astype(np.float32)
        return [unletterbox(prob_np[i], metas[i]).astype(np.float32) for i in range(len(metas))]
