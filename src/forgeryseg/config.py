from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .paths import resolve_existing_path
from .typing import Pathish


def _parse_override_value(raw: str) -> Any:
    """
    Parse CLI override values.

    Tries JSON first (so numbers/bools/lists/dicts work), otherwise falls back to string.
    """
    s = raw.strip()
    if s == "":
        return ""
    try:
        return json.loads(s)
    except Exception:
        return raw


def _set_deep(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = [k for k in dotted_key.split(".") if k]
    if not keys:
        raise ValueError("Empty override key")
    cur: dict[str, Any] = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    if not overrides:
        return cfg
    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override inválido (esperado key=value): {item!r}")
        key, raw = item.split("=", 1)
        _set_deep(out, key.strip(), _parse_override_value(raw))
    return out


def load_config_data(config_path: Pathish, *, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """
    Load a JSON/YAML config file into a dict and apply optional overrides.
    """
    path = Path(config_path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("Para ler YAML, instale PyYAML (pip install pyyaml).") from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config deve ser um objeto (dict), recebeu {type(data)}")
    return apply_overrides(data, overrides)


def default_path_roots(config_path: Pathish, *, extra_roots: Iterable[Path] | None = None) -> list[Path]:
    roots = [Path(config_path).parent, Path.cwd()]
    if extra_roots is not None:
        roots.extend(list(extra_roots))
    # de-dup preserving order
    out: list[Path] = []
    seen: set[Path] = set()
    for r in roots:
        rr = Path(r).resolve()
        if rr not in seen:
            out.append(rr)
            seen.add(rr)
    return out


def resolve_config_path(path: Pathish, *, config_path: Pathish, roots: Iterable[Path] | None = None) -> Path:
    """
    Resolve a config-relative path, searching:
      1) the path itself
      2) roots (defaults to [config_dir, cwd])
      3) /kaggle/input (via resolve_existing_path)
    """
    roots_eff = list(roots) if roots is not None else default_path_roots(config_path)
    return resolve_existing_path(path, roots=roots_eff, search_kaggle_input=True)


# ------------------------
# Segmentation config schema
# ------------------------

SegModelType = Literal["dinov2", "dinov2_freq_fusion"]
AugMode = Literal["none", "basic", "robust"]
SchedulerMode = Literal["none", "cosine", "onecycle"]


@dataclass
class EncoderConfig:
    model_name: str = "vit_base_patch14_dinov2"
    checkpoint_path: str | None = None
    pretrained: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> EncoderConfig:
        d = d or {}
        return cls(
            model_name=str(d.get("model_name", cls.model_name)),
            checkpoint_path=d.get("checkpoint_path"),
            pretrained=bool(d.get("pretrained", cls.pretrained)),
        )


@dataclass
class SegmentationModelConfig:
    type: SegModelType = "dinov2"
    input_size: int = 518
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    checkpoint: str | None = None
    decoder_hidden_channels: int = 256
    decoder_dropout: float = 0.0
    freeze_encoder: bool = True
    freq_fusion: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SegmentationModelConfig:
        d = d or {}
        model_type = d.get("type", d.get("model_type", cls.type))
        return cls(
            type=str(model_type),
            input_size=int(d.get("input_size", cls.input_size)),
            encoder=EncoderConfig.from_dict(d.get("encoder")),
            checkpoint=d.get("checkpoint"),
            decoder_hidden_channels=int(d.get("decoder_hidden_channels", cls.decoder_hidden_channels)),
            decoder_dropout=float(d.get("decoder_dropout", cls.decoder_dropout)),
            freeze_encoder=bool(d.get("freeze_encoder", cls.freeze_encoder)),
            freq_fusion=dict(d.get("freq_fusion", {})) if isinstance(d.get("freq_fusion", {}), dict) else {},
        )

    def validate(self) -> None:
        if str(self.type) not in {"dinov2", "dinov2_freq_fusion"}:
            raise ValueError("model.type must be one of: dinov2, dinov2_freq_fusion")
        if int(self.input_size) <= 0:
            raise ValueError("model.input_size must be > 0")
        if int(self.decoder_hidden_channels) <= 0:
            raise ValueError("model.decoder_hidden_channels must be > 0")


@dataclass
class TTAConfig:
    zoom_scale: float = 0.9
    weights: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.25])

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> TTAConfig:
        d = d or {}
        weights = d.get("weights", cls().weights)
        if not isinstance(weights, list):
            weights = list(weights)  # type: ignore[arg-type]
        return cls(
            zoom_scale=float(d.get("zoom_scale", cls.zoom_scale)),
            weights=[float(x) for x in weights],
        )


@dataclass
class TilingConfig:
    tile_size: int = 0
    overlap: int = 0
    batch_size: int = 4

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> TilingConfig:
        d = d or {}
        return cls(
            tile_size=int(d.get("tile_size", cls.tile_size)),
            overlap=int(d.get("overlap", cls.overlap)),
            batch_size=int(d.get("batch_size", cls.batch_size)),
        )


@dataclass
class PostprocessConfig:
    prob_threshold: float = 0.5
    gaussian_sigma: float = 0.0
    sobel_weight: float = 0.0
    open_kernel: int = 0
    close_kernel: int = 0
    min_area: int = 0
    min_mean_conf: float = 0.0
    min_prob_std: float = 0.0
    small_area: int | None = None
    small_min_mean_conf: float | None = None
    authentic_area_max: int | None = None
    authentic_conf_max: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PostprocessConfig:
        d = d or {}
        return cls(
            prob_threshold=float(d.get("prob_threshold", cls.prob_threshold)),
            gaussian_sigma=float(d.get("gaussian_sigma", cls.gaussian_sigma)),
            sobel_weight=float(d.get("sobel_weight", cls.sobel_weight)),
            open_kernel=int(d.get("open_kernel", cls.open_kernel)),
            close_kernel=int(d.get("close_kernel", cls.close_kernel)),
            min_area=int(d.get("min_area", cls.min_area)),
            min_mean_conf=float(d.get("min_mean_conf", cls.min_mean_conf)),
            min_prob_std=float(d.get("min_prob_std", cls.min_prob_std)),
            small_area=None if d.get("small_area") is None else int(d.get("small_area")),
            small_min_mean_conf=None
            if d.get("small_min_mean_conf") is None
            else float(d.get("small_min_mean_conf")),
            authentic_area_max=None if d.get("authentic_area_max") is None else int(d.get("authentic_area_max")),
            authentic_conf_max=None if d.get("authentic_conf_max") is None else float(d.get("authentic_conf_max")),
        )


@dataclass
class FFTParamsConfig:
    mode: str = "logmag"
    input_size: int = 256
    hp_radius_fraction: float = 0.1
    normalize_percentiles: list[float] = field(default_factory=lambda: [5.0, 95.0])

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FFTParamsConfig:
        d = d or {}
        percentiles = d.get("normalize_percentiles", cls().normalize_percentiles)
        if not isinstance(percentiles, list):
            percentiles = list(percentiles)  # type: ignore[arg-type]
        return cls(
            mode=str(d.get("mode", cls.mode)),
            input_size=int(d.get("input_size", cls.input_size)),
            hp_radius_fraction=float(d.get("hp_radius_fraction", cls.hp_radius_fraction)),
            normalize_percentiles=[float(x) for x in percentiles],
        )


@dataclass
class FFTGateConfig:
    enabled: bool = True
    checkpoint: str | None = None
    backbone: str = "resnet18"
    dropout: float = 0.0
    threshold: float = 0.5
    fft: FFTParamsConfig = field(default_factory=FFTParamsConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FFTGateConfig | None:
        if not isinstance(d, dict):
            return None
        if d.get("enabled", True) is False:
            return None
        return cls(
            enabled=bool(d.get("enabled", True)),
            checkpoint=d.get("checkpoint"),
            backbone=str(d.get("backbone", cls.backbone)),
            dropout=float(d.get("dropout", cls.dropout)),
            threshold=float(d.get("threshold", cls.threshold)),
            fft=FFTParamsConfig.from_dict(d.get("fft")),
        )


@dataclass
class SegmentationInferenceConfig:
    tta: TTAConfig = field(default_factory=TTAConfig)
    tiling: TilingConfig | None = None
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    fft_gate: FFTGateConfig | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SegmentationInferenceConfig:
        d = d or {}
        tiling_raw = d.get("tiling")
        tiling = TilingConfig.from_dict(tiling_raw) if isinstance(tiling_raw, dict) else None
        return cls(
            tta=TTAConfig.from_dict(d.get("tta")),
            tiling=tiling,
            postprocess=PostprocessConfig.from_dict(d.get("postprocess")),
            fft_gate=FFTGateConfig.from_dict(d.get("fft_gate")),
        )


@dataclass
class SegmentationTrainConfig:
    epochs: int = 5
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    val_fraction: float = 0.1
    seed: int = 42
    folds: int = 1
    fold: int = -1
    aug: AugMode = "basic"
    scheduler: SchedulerMode = "none"
    lr_min: float = 1e-6
    max_lr: float = 0.0
    pct_start: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SegmentationTrainConfig:
        d = d or {}
        return cls(
            epochs=int(d.get("epochs", cls.epochs)),
            batch_size=int(d.get("batch_size", cls.batch_size)),
            lr=float(d.get("lr", cls.lr)),
            weight_decay=float(d.get("weight_decay", cls.weight_decay)),
            num_workers=int(d.get("num_workers", cls.num_workers)),
            val_fraction=float(d.get("val_fraction", cls.val_fraction)),
            seed=int(d.get("seed", cls.seed)),
            folds=int(d.get("folds", cls.folds)),
            fold=int(d.get("fold", cls.fold)),
            aug=str(d.get("aug", cls.aug)),
            scheduler=str(d.get("scheduler", cls.scheduler)),
            lr_min=float(d.get("lr_min", cls.lr_min)),
            max_lr=float(d.get("max_lr", cls.max_lr)),
            pct_start=float(d.get("pct_start", cls.pct_start)),
        )


@dataclass
class SegmentationExperimentConfig:
    name: str
    model: SegmentationModelConfig
    inference: SegmentationInferenceConfig
    train: SegmentationTrainConfig = field(default_factory=SegmentationTrainConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, default_name: str) -> SegmentationExperimentConfig:
        # Legacy flat configs: lift keys into model/inference sections.
        if "model" not in d and ("input_size" in d or "encoder" in d):
            d = {
                "name": d.get("name", default_name),
                "model": {
                    "type": d.get("model_type", d.get("type", "dinov2")),
                    "input_size": d.get("input_size"),
                    "encoder": d.get("encoder"),
                    "checkpoint": d.get("checkpoint"),
                    "decoder_hidden_channels": d.get("decoder_hidden_channels"),
                    "decoder_dropout": d.get("decoder_dropout"),
                    "freeze_encoder": d.get("freeze_encoder"),
                    "freq_fusion": d.get("freq_fusion"),
                },
                "inference": {
                    "tta": d.get("tta"),
                    "tiling": d.get("tiling"),
                    "postprocess": d.get("postprocess"),
                    "fft_gate": d.get("fft_gate"),
                },
                "train": d.get("train"),
            }

        name = str(d.get("name", default_name))
        model = SegmentationModelConfig.from_dict(d.get("model"))
        model.validate()
        return cls(
            name=name,
            model=model,
            inference=SegmentationInferenceConfig.from_dict(d.get("inference")),
            train=SegmentationTrainConfig.from_dict(d.get("train")),
        )


def load_segmentation_config(
    config_path: Pathish, *, overrides: Iterable[str] | None = None
) -> SegmentationExperimentConfig:
    path = Path(config_path)
    data = load_config_data(path, overrides=overrides)
    return SegmentationExperimentConfig.from_dict(data, default_name=path.stem)


# ------------------------
# FFT classifier schema
# ------------------------


@dataclass
class FFTClassifierModelConfig:
    backbone: str = "resnet18"
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FFTClassifierModelConfig:
        d = d or {}
        return cls(
            backbone=str(d.get("backbone", cls.backbone)),
            dropout=float(d.get("dropout", cls.dropout)),
        )


@dataclass
class FFTClassifierTrainConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    val_fraction: float = 0.1
    seed: int = 42
    folds: int = 1
    fold: int = -1
    no_aug: bool = False
    scheduler: SchedulerMode = "none"
    lr_min: float = 1e-6
    max_lr: float = 0.0
    pct_start: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FFTClassifierTrainConfig:
        d = d or {}
        return cls(
            epochs=int(d.get("epochs", cls.epochs)),
            batch_size=int(d.get("batch_size", cls.batch_size)),
            lr=float(d.get("lr", cls.lr)),
            weight_decay=float(d.get("weight_decay", cls.weight_decay)),
            num_workers=int(d.get("num_workers", cls.num_workers)),
            val_fraction=float(d.get("val_fraction", cls.val_fraction)),
            seed=int(d.get("seed", cls.seed)),
            folds=int(d.get("folds", cls.folds)),
            fold=int(d.get("fold", cls.fold)),
            no_aug=bool(d.get("no_aug", cls.no_aug)),
            scheduler=str(d.get("scheduler", cls.scheduler)),
            lr_min=float(d.get("lr_min", cls.lr_min)),
            max_lr=float(d.get("max_lr", cls.max_lr)),
            pct_start=float(d.get("pct_start", cls.pct_start)),
        )


@dataclass
class FFTClassifierExperimentConfig:
    name: str
    fft: FFTParamsConfig
    model: FFTClassifierModelConfig
    train: FFTClassifierTrainConfig = field(default_factory=FFTClassifierTrainConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, default_name: str) -> FFTClassifierExperimentConfig:
        name = str(d.get("name", default_name))
        return cls(
            name=name,
            fft=FFTParamsConfig.from_dict(d.get("fft")),
            model=FFTClassifierModelConfig.from_dict(d.get("model")),
            train=FFTClassifierTrainConfig.from_dict(d.get("train")),
        )


def load_fft_classifier_config(
    config_path: Pathish, *, overrides: Iterable[str] | None = None
) -> FFTClassifierExperimentConfig:
    path = Path(config_path)
    data = load_config_data(path, overrides=overrides)
    return FFTClassifierExperimentConfig.from_dict(data, default_name=path.stem)
