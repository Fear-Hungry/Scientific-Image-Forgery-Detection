from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
from tqdm import tqdm

from ..config import apply_overrides, load_config_data
from ..dataset import list_cases, load_mask_instances
from ..eval import ScoreSummary
from ..inference import load_rgb, predict_prob_map_tiled
from ..inference_engine import InferenceEngine
from ..metric import of1_score
from ..postprocess import PostprocessParams, postprocess_prob
from ..training.utils import stratified_splits
from ..tta import IdentityTTA
from ..typing import Pathish, Split

Objective = Literal["mean_score", "mean_forged", "combo"]


@dataclass(frozen=True)
class TuneResult:
    best_value: float
    best_summary: ScoreSummary
    best_postprocess: PostprocessParams
    best_overrides: list[str]
    tuned_config_path: Path
    cache_path: Path
    n_cases: int
    seconds_inference: float
    seconds_optuna: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_value": float(self.best_value),
            "best_summary": self.best_summary.as_dict(),
            "best_postprocess": dataclasses.asdict(self.best_postprocess),
            "best_overrides": list(self.best_overrides),
            "tuned_config_path": str(self.tuned_config_path),
            "cache_path": str(self.cache_path),
            "n_cases": int(self.n_cases),
            "seconds_inference": float(self.seconds_inference),
            "seconds_optuna": float(self.seconds_optuna),
        }


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    out: list[float] = []
    x = float(start)
    while x <= float(stop) + 1e-12:
        out.append(float(x))
        x += float(step)
    return out


def _slug_float(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _select_stratified_subset(
    cases: list,
    *,
    seed: int,
    limit: int,
) -> list:
    if limit <= 0 or len(cases) <= limit:
        return cases
    rng = np.random.default_rng(int(seed))
    idx_auth = [i for i, c in enumerate(cases) if c.mask_path is None]
    idx_forg = [i for i, c in enumerate(cases) if c.mask_path is not None]

    p_forg = len(idx_forg) / max(1, len(cases))
    n_forg = int(round(limit * p_forg))
    n_forg = min(n_forg, len(idx_forg))
    n_auth = int(limit - n_forg)
    n_auth = min(n_auth, len(idx_auth))

    chosen: list[int] = []
    if n_auth > 0:
        chosen.extend(rng.choice(idx_auth, size=n_auth, replace=False).tolist())
    if n_forg > 0:
        chosen.extend(rng.choice(idx_forg, size=n_forg, replace=False).tolist())
    rng.shuffle(chosen)
    return [cases[i] for i in chosen]


def _cache_name(
    *,
    config_path: Path,
    split: Split,
    val_fraction: float,
    seed: int,
    limit: int,
    use_tta: bool,
    tiling: tuple[int, int] | None,
) -> str:
    tile = f"tile{tiling[0]}_ov{tiling[1]}" if tiling is not None else "notile"
    return (
        f"prob_cache_{config_path.stem}_{split}_val{_slug_float(val_fraction)}"
        f"_seed{int(seed)}_limit{int(limit)}_tta{int(use_tta)}_{tile}.npz"
    )


def _save_prob_cache(path: Path, *, case_ids: list[str], probs: list[np.ndarray]) -> None:
    arrays: dict[str, Any] = {"case_ids": np.asarray(case_ids, dtype=object)}
    for i, p in enumerate(probs):
        arrays[f"prob_{i}"] = p.astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_prob_cache(path: Path) -> tuple[list[str], list[np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    case_ids = [str(x) for x in data["case_ids"].tolist()]
    probs: list[np.ndarray] = []
    for i in range(len(case_ids)):
        probs.append(data[f"prob_{i}"].astype(np.float32))
    return case_ids, probs


def _predict_prob_maps(
    engine: InferenceEngine,
    cases: list,
    *,
    use_tta: bool,
    batch_size: int,
) -> list[np.ndarray]:
    # If tiling is enabled, fall back to per-image (tiling already batches tiles internally).
    if engine.tiling is not None:
        transforms, weights = (engine.tta_transforms, engine.tta_weights) if use_tta else ([IdentityTTA()], [1.0])
        out: list[np.ndarray] = []
        for case in tqdm(cases, desc="Infer (tiling)"):
            img = load_rgb(case.image_path)
            out.append(
                predict_prob_map_tiled(
                    engine.model,
                    img,
                    input_size=int(engine.input_size),
                    device=engine.device,
                    tiling=engine.tiling,
                    tta_transforms=transforms,
                    tta_weights=weights,
                )
            )
        return out

    # Non-tiling: batched inference
    eff = engine if use_tta else dataclasses.replace(engine, tta_transforms=[IdentityTTA()], tta_weights=[1.0])
    paths = [c.image_path for c in cases]
    bs = int(max(1, batch_size))
    out: list[np.ndarray] = []
    for i in tqdm(range(0, len(paths), bs), desc="Infer"):
        chunk = paths[i : i + bs]
        images = [load_rgb(p) for p in chunk]
        out.extend(eff._predict_prob_maps_batched(images))
    return out


def _score_from_probs(
    *,
    cases: list,
    probs: list[np.ndarray],
    post: PostprocessParams,
) -> ScoreSummary:
    scores_all: list[float] = []
    scores_auth: list[float] = []
    scores_forg: list[float] = []
    auth_pred_as_forged = 0
    forg_pred_as_auth = 0

    for case, prob in zip(cases, probs, strict=True):
        pred_instances = postprocess_prob(prob, post)
        pred_is_auth = len(pred_instances) == 0

        if case.mask_path is None:
            s = 1.0 if pred_is_auth else 0.0
            if not pred_is_auth:
                auth_pred_as_forged += 1
            scores_all.append(float(s))
            scores_auth.append(float(s))
            continue

        if pred_is_auth:
            forg_pred_as_auth += 1
            s = 0.0
        else:
            gt_instances = load_mask_instances(case.mask_path)
            s = float(of1_score(pred_instances, gt_instances))

        scores_all.append(float(s))
        scores_forg.append(float(s))

    def _mean(x: list[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    return ScoreSummary(
        mean_score=_mean(scores_all),
        mean_authentic=_mean(scores_auth),
        mean_forged=_mean(scores_forg),
        n_cases=len(scores_all),
        n_authentic=len(scores_auth),
        n_forged=len(scores_forg),
        auth_pred_as_forged=int(auth_pred_as_forged),
        forg_pred_as_auth=int(forg_pred_as_auth),
        decode_errors_scoring=0,
    )


def tune_postprocess_optuna(
    *,
    config_path: Pathish,
    data_root: Pathish,
    split: Split = "train",
    out_dir: Pathish = "outputs/optuna",
    device: str | torch.device = "cuda",
    base_overrides: Iterable[str] | None = None,
    val_fraction: float = 0.10,
    seed: int = 42,
    folds: int = 1,
    fold: int = 0,
    limit: int = 0,
    use_tta: bool = True,
    batch_size: int = 4,
    n_trials: int = 100,
    timeout_sec: int | None = None,
    objective: Objective = "mean_score",
    cache_path: Pathish | None = None,
) -> TuneResult:
    """
    Bayesian optimization (Optuna) of `inference.postprocess.*` using local oF1 scoring.

    Strategy:
      1) Build a stratified validation subset (authentic vs forged).
      2) Run model inference once and cache prob_maps to disk.
      3) Optimize postprocess parameters on cached prob_maps (fast trials).
      4) Write a tuned JSON config ready to use in submission notebook.
    """
    try:
        import optuna  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Optuna não está instalado. No Kaggle (internet ON): `pip install optuna` "
            "ou adicione optuna no requirements-kaggle.txt."
        ) from e

    config_path = Path(config_path)
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)

    base_overrides_list = list(base_overrides) if base_overrides is not None else []

    engine = InferenceEngine.from_config(
        config_path=config_path,
        device=device_t,
        overrides=base_overrides_list,
        path_roots=[out_dir, config_path.parent, Path.cwd()],
        amp=True,
    )

    # Validation subset
    all_cases = list_cases(data_root, split, include_authentic=True, include_forged=True)
    labels = np.asarray([1 if c.mask_path is not None else 0 for c in all_cases], dtype=np.int64)
    splits = stratified_splits(labels, folds=int(folds), val_fraction=float(val_fraction), seed=int(seed))
    if int(folds) > 1:
        if not (0 <= int(fold) < int(folds)):
            raise ValueError("fold must be in [0, folds)")
        _, _, val_idx = next((f_id, tr, va) for f_id, tr, va in splits if int(f_id) == int(fold))
    else:
        _, _, val_idx = splits[0]
    val_cases = [all_cases[int(i)] for i in val_idx.tolist()]
    val_cases = _select_stratified_subset(val_cases, seed=int(seed), limit=int(limit))

    tiling = (int(engine.tiling.tile_size), int(engine.tiling.overlap)) if engine.tiling is not None else None
    cache_path = (
        Path(cache_path)
        if cache_path is not None
        else out_dir
        / _cache_name(
            config_path=config_path,
            split=split,
            val_fraction=float(val_fraction),
            seed=int(seed),
            limit=int(limit),
            use_tta=bool(use_tta),
            tiling=tiling,
        )
    )

    # Cache prob maps
    t0 = time.time()
    if cache_path.exists():
        cached_ids, probs = _load_prob_cache(cache_path)
        by_id = {c.case_id: c for c in val_cases}
        missing = [cid for cid in cached_ids if cid not in by_id]
        if missing:
            raise RuntimeError(f"Cache {cache_path} contém ids desconhecidos (ex.: {missing[:5]}). Delete o cache.")
        val_cases = [by_id[cid] for cid in cached_ids]
    else:
        probs = _predict_prob_maps(engine, val_cases, use_tta=bool(use_tta), batch_size=int(batch_size))
        _save_prob_cache(cache_path, case_ids=[c.case_id for c in val_cases], probs=probs)
    seconds_infer = float(time.time() - t0)

    # Baseline (current postprocess from config after base_overrides)
    base_post = engine.postprocess
    base_score = _score_from_probs(cases=val_cases, probs=probs, post=base_post)
    print(
        f"[baseline] mean={base_score.mean_score:.6f} mean_forged={base_score.mean_forged:.6f} "
        f"auth_pred_as_forged={base_score.auth_pred_as_forged} forg_pred_as_auth={base_score.forg_pred_as_auth}"
    )

    # Optuna
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def _objective(trial) -> float:
        prob_threshold = trial.suggest_float("prob_threshold", 0.20, 0.60)
        use_hysteresis = trial.suggest_categorical("use_hysteresis", [False, True])
        hysteresis_delta = (
            trial.suggest_float("hysteresis_delta", 0.05, 0.25) if bool(use_hysteresis) else None
        )
        prob_threshold_low = (
            max(0.0, float(prob_threshold) - float(hysteresis_delta)) if hysteresis_delta is not None else None
        )

        gaussian_sigma = trial.suggest_float("gaussian_sigma", 0.0, 1.5)
        sobel_weight = trial.suggest_float("sobel_weight", 0.0, 0.25)
        open_kernel = trial.suggest_categorical("open_kernel", [0, 3, 5, 7])
        close_kernel = trial.suggest_categorical("close_kernel", [0, 3, 5, 7, 9, 11])
        morph_order = trial.suggest_categorical("morph_order", ["open_close", "close_open"])
        use_final_morph = trial.suggest_categorical("use_final_morph", [False, True])
        final_open_kernel = (
            trial.suggest_categorical("final_open_kernel", [0, 3, 5]) if bool(use_final_morph) else 0
        )
        final_close_kernel = (
            trial.suggest_categorical("final_close_kernel", [0, 3, 5, 7]) if bool(use_final_morph) else 0
        )
        fill_holes = trial.suggest_categorical("fill_holes", [False, True])
        min_area = trial.suggest_int("min_area", 0, 400, step=8)
        min_mean_conf = trial.suggest_float("min_mean_conf", 0.0, 0.35)
        min_prob_std = trial.suggest_float("min_prob_std", 0.0, 0.35)

        use_small_gate = trial.suggest_categorical("use_small_gate", [False, True])
        small_area = trial.suggest_int("small_area", 0, 1500, step=16) if use_small_gate else None
        small_min_mean_conf = trial.suggest_float("small_min_mean_conf", 0.40, 0.90) if use_small_gate else None

        use_auth_gate = trial.suggest_categorical("use_auth_gate", [False, True])
        authentic_area_max = trial.suggest_int("authentic_area_max", 0, 1500, step=16) if use_auth_gate else None
        authentic_conf_max = trial.suggest_float("authentic_conf_max", 0.40, 0.90) if use_auth_gate else None

        post = PostprocessParams(
            prob_threshold=float(prob_threshold),
            prob_threshold_low=None if prob_threshold_low is None else float(prob_threshold_low),
            gaussian_sigma=float(gaussian_sigma),
            sobel_weight=float(sobel_weight),
            open_kernel=int(open_kernel),
            close_kernel=int(close_kernel),
            morph_order=str(morph_order),  # type: ignore[arg-type]
            final_open_kernel=int(final_open_kernel),
            final_close_kernel=int(final_close_kernel),
            fill_holes=bool(fill_holes),
            min_area=int(min_area),
            min_mean_conf=float(min_mean_conf),
            min_prob_std=float(min_prob_std),
            small_area=None if small_area is None else int(small_area),
            small_min_mean_conf=None if small_min_mean_conf is None else float(small_min_mean_conf),
            authentic_area_max=None if authentic_area_max is None else int(authentic_area_max),
            authentic_conf_max=None if authentic_conf_max is None else float(authentic_conf_max),
        )

        # Evaluate with pruning
        scores_all: list[float] = []
        scores_auth: list[float] = []
        scores_forg: list[float] = []
        auth_pred_as_forged = 0
        forg_pred_as_auth = 0

        for i, (case, prob) in enumerate(zip(val_cases, probs, strict=True), start=1):
            pred_instances = postprocess_prob(prob, post)
            pred_is_auth = len(pred_instances) == 0

            if case.mask_path is None:
                s = 1.0 if pred_is_auth else 0.0
                if not pred_is_auth:
                    auth_pred_as_forged += 1
                scores_all.append(float(s))
                scores_auth.append(float(s))
            else:
                if pred_is_auth:
                    forg_pred_as_auth += 1
                    s = 0.0
                else:
                    gt_instances = load_mask_instances(case.mask_path)
                    s = float(of1_score(pred_instances, gt_instances))
                scores_all.append(float(s))
                scores_forg.append(float(s))

            if i % 25 == 0:
                mean_partial = float(np.mean(scores_all))
                trial.report(mean_partial, step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_score = float(np.mean(scores_all)) if scores_all else 0.0
        mean_forged = float(np.mean(scores_forg)) if scores_forg else 0.0
        mean_auth = float(np.mean(scores_auth)) if scores_auth else 0.0

        # Store useful diagnostics
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("mean_forged", mean_forged)
        trial.set_user_attr("mean_authentic", mean_auth)
        trial.set_user_attr("auth_pred_as_forged", int(auth_pred_as_forged))
        trial.set_user_attr("forg_pred_as_auth", int(forg_pred_as_auth))

        if objective == "mean_forged":
            return float(mean_forged)
        if objective == "combo":
            return float(mean_score + 0.25 * mean_forged)
        return float(mean_score)

    t1 = time.time()
    study.optimize(_objective, n_trials=int(n_trials), timeout=timeout_sec)
    seconds_optuna = float(time.time() - t1)

    best_params = dict(study.best_params)
    best_post = PostprocessParams(
        prob_threshold=float(best_params["prob_threshold"]),
        prob_threshold_low=(
            max(0.0, float(best_params["prob_threshold"]) - float(best_params["hysteresis_delta"]))
            if bool(best_params.get("use_hysteresis")) and best_params.get("hysteresis_delta") is not None
            else None
        ),
        gaussian_sigma=float(best_params["gaussian_sigma"]),
        sobel_weight=float(best_params["sobel_weight"]),
        open_kernel=int(best_params["open_kernel"]),
        close_kernel=int(best_params["close_kernel"]),
        morph_order=str(best_params.get("morph_order", "open_close")),  # type: ignore[arg-type]
        final_open_kernel=(
            int(best_params["final_open_kernel"])
            if bool(best_params.get("use_final_morph")) and best_params.get("final_open_kernel") is not None
            else 0
        ),
        final_close_kernel=(
            int(best_params["final_close_kernel"])
            if bool(best_params.get("use_final_morph")) and best_params.get("final_close_kernel") is not None
            else 0
        ),
        fill_holes=bool(best_params.get("fill_holes", False)),
        min_area=int(best_params["min_area"]),
        min_mean_conf=float(best_params["min_mean_conf"]),
        min_prob_std=float(best_params["min_prob_std"]),
        small_area=(
            int(best_params["small_area"])
            if bool(best_params.get("use_small_gate")) and best_params.get("small_area") is not None
            else None
        ),
        small_min_mean_conf=(
            float(best_params["small_min_mean_conf"])
            if bool(best_params.get("use_small_gate")) and best_params.get("small_min_mean_conf") is not None
            else None
        ),
        authentic_area_max=(
            int(best_params["authentic_area_max"])
            if bool(best_params.get("use_auth_gate")) and best_params.get("authentic_area_max") is not None
            else None
        ),
        authentic_conf_max=(
            float(best_params["authentic_conf_max"])
            if bool(best_params.get("use_auth_gate")) and best_params.get("authentic_conf_max") is not None
            else None
        ),
    )

    best_summary = _score_from_probs(cases=val_cases, probs=probs, post=best_post)

    def _ov(k: str, v: Any) -> str:
        if v is None:
            return f"{k}=null"
        if isinstance(v, bool):
            return f"{k}={'true' if v else 'false'}"
        if isinstance(v, (int, float)):
            return f"{k}={v}"
        return f"{k}={json.dumps(v)}"

    best_overrides = list(base_overrides_list)
    best_overrides.extend(
        [
            _ov("inference.postprocess.prob_threshold", best_post.prob_threshold),
            _ov("inference.postprocess.prob_threshold_low", best_post.prob_threshold_low),
            _ov("inference.postprocess.gaussian_sigma", best_post.gaussian_sigma),
            _ov("inference.postprocess.sobel_weight", best_post.sobel_weight),
            _ov("inference.postprocess.open_kernel", best_post.open_kernel),
            _ov("inference.postprocess.close_kernel", best_post.close_kernel),
            _ov("inference.postprocess.morph_order", best_post.morph_order),
            _ov("inference.postprocess.final_open_kernel", best_post.final_open_kernel),
            _ov("inference.postprocess.final_close_kernel", best_post.final_close_kernel),
            _ov("inference.postprocess.fill_holes", best_post.fill_holes),
            _ov("inference.postprocess.min_area", best_post.min_area),
            _ov("inference.postprocess.min_mean_conf", best_post.min_mean_conf),
            _ov("inference.postprocess.min_prob_std", best_post.min_prob_std),
            _ov("inference.postprocess.small_area", best_post.small_area),
            _ov("inference.postprocess.small_min_mean_conf", best_post.small_min_mean_conf),
            _ov("inference.postprocess.authentic_area_max", best_post.authentic_area_max),
            _ov("inference.postprocess.authentic_conf_max", best_post.authentic_conf_max),
        ]
    )

    tuned_cfg = load_config_data(config_path)
    tuned_cfg = apply_overrides(tuned_cfg, best_overrides)
    tuned_path = out_dir / f"tuned_{config_path.stem}_optuna_{objective}.json"
    tuned_path.write_text(json.dumps(tuned_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Save study results
    try:
        df = study.trials_dataframe()
        df.to_csv(out_dir / "optuna_trials.csv", index=False)
    except Exception:
        pass

    result = TuneResult(
        best_value=float(study.best_value),
        best_summary=best_summary,
        best_postprocess=best_post,
        best_overrides=best_overrides,
        tuned_config_path=tuned_path,
        cache_path=cache_path,
        n_cases=len(val_cases),
        seconds_inference=float(seconds_infer),
        seconds_optuna=float(seconds_optuna),
    )

    (out_dir / "optuna_best.json").write_text(
        json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"[best] mean={best_summary.mean_score:.6f} mean_forged={best_summary.mean_forged:.6f} "
        f"auth_pred_as_forged={best_summary.auth_pred_as_forged} forg_pred_as_auth={best_summary.forg_pred_as_auth}"
    )
    print(f"[best] wrote {tuned_path}")
    return result
