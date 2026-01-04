from __future__ import annotations

# %% [markdown]
# # Recod.ai/LUC — Training (Kaggle, internet ON)
#
# Este notebook gera os **pesos (`*.pth`)** necessários para rodar a submissão offline:
#
# - Segmentação (DINOv2 + decoder) → `outputs/models/r69.pth`
# - (Opcional) Classificador FFT → `outputs/models/fft_cls.pth`
#
# Fluxo recomendado:
#
# 1. Kaggle Notebook **com internet ON** + **GPU**.
# 2. Anexe o dataset da competição.
# 3. Anexe um dataset com **este repo** (ou clone).
# 4. Rode as células para treinar e salvar os checkpoints em `/kaggle/working/outputs/models/`.
# 5. Empacote um folder `kaggle_bundle/` para criar um Kaggle Dataset com código + pesos.
#
# Observação: por regra do repo, a lógica nasce aqui (`.py`) e é espelhada no `.ipynb`.
#
# %%
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

print(f"python={sys.version.split()[0]} platform={platform.platform()}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")


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

# %%
# -------------------------
# (Opcional) Instalar deps
# -------------------------
#
# No Kaggle, normalmente já existe torch/torchvision. Se faltar timm/albumentations/etc,
# use INSTALL_DEPS=True com internet ON.
INSTALL_DEPS = False

if INSTALL_DEPS:
    req = CODE_ROOT / "requirements-kaggle.txt"
    print(f"Installing: {req}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])

# %%
from forgeryseg.kaggle import package_kaggle_dataset
from forgeryseg.submission import write_submission_csv
from forgeryseg.training.dino_decoder import train_dino_decoder
from forgeryseg.training.fft_classifier import train_fft_classifier

# %%
# -------------------------
# Config (edite aqui)
# -------------------------

DATA_ROOT: Path | None = None  # None => auto-detect (Kaggle -> local)

SEG_TRAIN_CONFIG = CONFIG_ROOT / "dino_v3_518_r69.json"
FFT_TRAIN_CONFIG = CONFIG_ROOT / "fft_classifier_logmag_256.json"

TRAIN_SEG = True
TRAIN_FFT = True

SEG_FOLDS = 1  # use 1 para gerar r69.pth diretamente; >1 cria r69_fold{i}.pth
FFT_FOLDS = 1  # use 1 para gerar fft_cls.pth diretamente; >1 cria fft_cls_fold{i}.pth

SEG_EPOCHS = 5
SEG_BATCH = 4
SEG_LR = 1e-3
SEG_WD = 1e-4
SEG_NUM_WORKERS = 2
SEG_AUG = "robust"  # none | basic | robust
SEG_SCHEDULER = "cosine"  # none | cosine | onecycle
SEG_PATIENCE = 3  # early stopping em val_of1 (0 desliga)

FFT_EPOCHS = 5
FFT_BATCH = 32
FFT_LR = 1e-3
FFT_WD = 1e-4
FFT_NUM_WORKERS = 2
FFT_SCHEDULER = "cosine"  # none | cosine | onecycle

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT_MODELS = OUT_DIR / "outputs" / "models"

SEG_OUT = OUT_MODELS / "r69.pth"
FFT_OUT = OUT_MODELS / "fft_cls.pth"

# (Opcional) checar score local rapidamente após treinar:
EVAL_AFTER_TRAIN = True
EVAL_SPLIT = "train"  # train | supplemental
EVAL_LIMIT = 0  # 0 = sem limite (usa tudo)

# (Opcional) tunar pós-processamento (rápido) num subset de validação.
#
# Meta: aumentar mean_forged (e reduzir forg_pred_as_auth) sem destruir mean_authentic.
#
# Observação: este sweep roda por padrão em um subset (val_fraction) para ser viável no Kaggle.
TUNE_POSTPROCESS = True
TUNE_CONFIG = CONFIG_ROOT / "dino_v3_518_r69_fft_gate.json"
TUNE_SPLIT = EVAL_SPLIT
TUNE_VAL_FRACTION = 0.10
TUNE_SEED = 42
TUNE_LIMIT = 0  # 0 = usa todo o subset de validação
TUNE_BATCH = 4  # ajuste conforme VRAM
TUNE_USE_TTA = False  # True = mais fiel ao config, porém mais lento
TUNE_THR_START = 0.20
TUNE_THR_STOP = 0.60
TUNE_THR_STEP = 0.05
TUNE_WRITE_TUNED_CONFIG = True

# Empacotar um folder pronto para upload como Kaggle Dataset (offline):
DO_PACKAGE = True
PKG_OUT = OUT_DIR / "kaggle_bundle"

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
            if (d / "sample_submission.csv").exists() and (
                (d / "train_images").exists() or (d / "test_images").exists()
            ):
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


# %%
# -------------------------
# Run
# -------------------------

data_root = _find_recodai_root()
print(f"data_root={data_root}")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print(f"device={device} (Dica: ative GPU em Settings -> Accelerator)")

OUT_MODELS.mkdir(parents=True, exist_ok=True)

# %%
# -----
# Train (Segmentation)
# -----

seg_result = None
if TRAIN_SEG:
    seg_result = train_dino_decoder(
        config_path=SEG_TRAIN_CONFIG,
        data_root=data_root,
        out_path=SEG_OUT,
        device=device_str,
        split="train",
        epochs=int(SEG_EPOCHS),
        batch_size=int(SEG_BATCH),
        lr=float(SEG_LR),
        weight_decay=float(SEG_WD),
        num_workers=int(SEG_NUM_WORKERS),
        folds=int(SEG_FOLDS),
        fold=None,
        aug=SEG_AUG,  # type: ignore[arg-type]
        scheduler=SEG_SCHEDULER,  # type: ignore[arg-type]
        patience=int(SEG_PATIENCE),
    )

    # Se treinou k-fold, copia o melhor fold para o path "base" (r69.pth),
    # para facilitar o uso em configs que apontam para outputs/models/r69.pth.
    if seg_result is not None and int(SEG_FOLDS) > 1:
        best = max(seg_result.fold_results, key=lambda fr: fr.best_val_of1)
        if best.checkpoint_path != SEG_OUT:
            SEG_OUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best.checkpoint_path, SEG_OUT)
            print(f"Copied best fold checkpoint -> {SEG_OUT} (from {best.checkpoint_path})")

# %%
# -----
# Train (FFT classifier)
# -----

fft_saved = None
if TRAIN_FFT:
    fft_saved = train_fft_classifier(
        config_path=FFT_TRAIN_CONFIG,
        data_root=data_root,
        out_path=FFT_OUT,
        device=device,
        epochs=int(FFT_EPOCHS),
        batch_size=int(FFT_BATCH),
        lr=float(FFT_LR),
        weight_decay=float(FFT_WD),
        num_workers=int(FFT_NUM_WORKERS),
        folds=int(FFT_FOLDS),
        scheduler=FFT_SCHEDULER,  # type: ignore[arg-type]
    )

    if fft_saved and int(FFT_FOLDS) > 1:
        # escolhe melhor fold por menor val_loss no checkpoint
        best_path = min(
            fft_saved,
            key=lambda p: float(torch.load(p, map_location="cpu").get("val_loss", float("inf"))),
        )
        if best_path != FFT_OUT:
            FFT_OUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, FFT_OUT)
            print(f"Copied best FFT fold -> {FFT_OUT} (from {best_path})")

# %%
# -------------------------
# Quick evaluation (local)
# -------------------------
#
# Gera um submission no split train/supplemental e calcula oF1 local.

if EVAL_AFTER_TRAIN:
    from forgeryseg.eval import score_submission_csv, validate_submission_format

    eval_cfg = CONFIG_ROOT / "dino_v3_518_r69_fft_gate.json"
    eval_csv = OUT_DIR / f"submission_{EVAL_SPLIT}.csv"

    stats = write_submission_csv(
        config_path=eval_cfg,
        data_root=data_root,
        split=EVAL_SPLIT,  # type: ignore[arg-type]
        out_path=eval_csv,
        device=device,
        limit=int(EVAL_LIMIT),
        path_roots=[OUT_DIR, CODE_ROOT, CONFIG_ROOT],
    )
    print(stats)

    fmt = validate_submission_format(eval_csv, data_root=data_root, split=EVAL_SPLIT)  # type: ignore[arg-type]
    print("\n[Format check]")
    print(json.dumps(fmt, indent=2, ensure_ascii=False))

    score = score_submission_csv(eval_csv, data_root=data_root, split=EVAL_SPLIT)  # type: ignore[arg-type]
    print("\n[Local score]")
    print(json.dumps(score.as_dict(csv_path=eval_csv, split=EVAL_SPLIT), indent=2, ensure_ascii=False))

# %%
# -------------------------
# Postprocess sweep (threshold)
# -------------------------
#
# Faz um sweep do `prob_threshold` com overrides "relaxados" (desliga filtros agressivos)
# em um subset de validação estratificado (authentic vs forged).

if TUNE_POSTPROCESS:
    import dataclasses
    import math
    import time

    import numpy as np
    from tqdm import tqdm

    from forgeryseg.config import apply_overrides, load_config_data
    from forgeryseg.dataset import list_cases, load_mask_instances
    from forgeryseg.eval import ScoreSummary
    from forgeryseg.inference import load_rgb
    from forgeryseg.inference_engine import InferenceEngine
    from forgeryseg.metric import of1_score
    from forgeryseg.postprocess import PostprocessParams, postprocess_prob
    from forgeryseg.training.utils import stratified_splits

    def _frange(start: float, stop: float, step: float) -> list[float]:
        if step <= 0:
            raise ValueError("step must be > 0")
        out: list[float] = []
        x = float(start)
        while x <= float(stop) + 1e-12:
            out.append(float(x))
            x += float(step)
        return out

    def _select_stratified_subset(cases: list, *, seed: int, limit: int) -> list:
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
        chosen = []
        if n_auth > 0:
            chosen.extend(rng.choice(idx_auth, size=n_auth, replace=False).tolist())
        if n_forg > 0:
            chosen.extend(rng.choice(idx_forg, size=n_forg, replace=False).tolist())
        rng.shuffle(chosen)
        return [cases[i] for i in chosen]

    # Overrides sugeridos (relaxar filtros agressivos)
    base_overrides = [
        "inference.fft_gate.enabled=false",
        "inference.postprocess.min_prob_std=0.0",
        "inference.postprocess.small_area=null",
        "inference.postprocess.small_min_mean_conf=null",
        "inference.postprocess.authentic_area_max=null",
        "inference.postprocess.authentic_conf_max=null",
        "inference.postprocess.min_area=32",
        "inference.postprocess.open_kernel=0",
        "inference.postprocess.close_kernel=0",
        "inference.postprocess.gaussian_sigma=0.0",
        "inference.postprocess.sobel_weight=0.0",
    ]

    thresholds = _frange(float(TUNE_THR_START), float(TUNE_THR_STOP), float(TUNE_THR_STEP))
    if not thresholds:
        raise RuntimeError("No thresholds configured")

    print(
        f"[tune] split={TUNE_SPLIT} val_fraction={TUNE_VAL_FRACTION} "
        f"n_thresholds={len(thresholds)} use_tta={TUNE_USE_TTA} batch={TUNE_BATCH}"
    )

    # Load engine once (model + input_size). Postprocess will be replaced per-threshold.
    engine = InferenceEngine.from_config(
        config_path=TUNE_CONFIG,
        device=device,
        overrides=base_overrides,
        path_roots=[OUT_DIR, CODE_ROOT, CONFIG_ROOT],
        amp=True,
    )

    # Define validation subset (stratified).
    all_cases = list_cases(data_root, TUNE_SPLIT, include_authentic=True, include_forged=True)
    labels = np.asarray([1 if c.mask_path is not None else 0 for c in all_cases], dtype=np.int64)
    split_iter = stratified_splits(
        labels,
        folds=1,
        val_fraction=float(TUNE_VAL_FRACTION),
        seed=int(TUNE_SEED),
    )
    _, _, val_idx = next(iter(split_iter))
    val_cases = [all_cases[int(i)] for i in val_idx.tolist()]
    val_cases = _select_stratified_subset(val_cases, seed=int(TUNE_SEED), limit=int(TUNE_LIMIT))
    n_auth = sum(1 for c in val_cases if c.mask_path is None)
    n_forg = sum(1 for c in val_cases if c.mask_path is not None)
    print(f"[tune] val_cases={len(val_cases)} authentic={n_auth} forged={n_forg}")

    # Fixed postprocess (with overrides applied), except prob_threshold.
    base_post: PostprocessParams = engine.postprocess

    # Accumulators per threshold
    acc: dict[float, dict[str, float | int]] = {
        thr: {
            "sum_all": 0.0,
            "sum_auth": 0.0,
            "sum_forg": 0.0,
            "n_all": 0,
            "n_auth": 0,
            "n_forg": 0,
            "auth_pred_as_forged": 0,
            "forg_pred_as_auth": 0,
        }
        for thr in thresholds
    }

    def _predict_prob_maps_no_tta(images: list[np.ndarray]) -> list[np.ndarray]:
        from forgeryseg.image import letterbox_reflect, unletterbox

        padded = []
        metas = []
        for img in images:
            pad, meta = letterbox_reflect(img, int(engine.input_size))
            padded.append(pad)
            metas.append(meta)

        x = torch.stack(
            [torch.from_numpy(im).permute(2, 0, 1).contiguous().float() / 255.0 for im in padded],
            dim=0,
        ).to(engine.device)

        with torch.no_grad():
            if engine.amp and engine.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = engine.model(x)
            else:
                logits = engine.model(x)
            prob = torch.sigmoid(logits)[:, 0].float()

        prob_np = prob.detach().cpu().numpy().astype(np.float32)
        return [unletterbox(prob_np[i], metas[i]).astype(np.float32) for i in range(len(metas))]

    t0 = time.time()
    bs = int(max(1, TUNE_BATCH))
    for i in tqdm(range(0, len(val_cases), bs), desc="[tune] infer"):
        batch_cases = val_cases[i : i + bs]
        images = [load_rgb(c.image_path) for c in batch_cases]

        if bool(TUNE_USE_TTA):
            probs = engine._predict_prob_maps_batched(images)
        else:
            probs = _predict_prob_maps_no_tta(images)

        for case, prob in zip(batch_cases, probs, strict=True):
            gt_instances = [] if case.mask_path is None else load_mask_instances(case.mask_path)
            gt_is_auth = case.mask_path is None

            for thr in thresholds:
                post = dataclasses.replace(base_post, prob_threshold=float(thr))
                pred_instances = postprocess_prob(prob, post)
                pred_is_auth = len(pred_instances) == 0

                if gt_is_auth:
                    s = 1.0 if pred_is_auth else 0.0
                    if not pred_is_auth:
                        acc[thr]["auth_pred_as_forged"] = int(acc[thr]["auth_pred_as_forged"]) + 1
                    acc[thr]["sum_auth"] = float(acc[thr]["sum_auth"]) + float(s)
                    acc[thr]["n_auth"] = int(acc[thr]["n_auth"]) + 1
                else:
                    if pred_is_auth:
                        acc[thr]["forg_pred_as_auth"] = int(acc[thr]["forg_pred_as_auth"]) + 1
                        s = 0.0
                    else:
                        s = float(of1_score(pred_instances, gt_instances))
                    acc[thr]["sum_forg"] = float(acc[thr]["sum_forg"]) + float(s)
                    acc[thr]["n_forg"] = int(acc[thr]["n_forg"]) + 1

                acc[thr]["sum_all"] = float(acc[thr]["sum_all"]) + float(s)
                acc[thr]["n_all"] = int(acc[thr]["n_all"]) + 1

    dt = time.time() - t0
    print(f"[tune] done in {dt:.1f}s")

    best_thr = None
    best_score = -math.inf
    results: list[tuple[float, ScoreSummary]] = []
    for thr in thresholds:
        a = acc[thr]
        n_all = int(a["n_all"])
        n_auth = int(a["n_auth"])
        n_forg = int(a["n_forg"])
        mean_all = float(a["sum_all"]) / max(1, n_all)
        mean_auth = float(a["sum_auth"]) / max(1, n_auth)
        mean_forg = float(a["sum_forg"]) / max(1, n_forg)
        summary = ScoreSummary(
            mean_score=mean_all,
            mean_authentic=mean_auth,
            mean_forged=mean_forg,
            n_cases=n_all,
            n_authentic=n_auth,
            n_forged=n_forg,
            auth_pred_as_forged=int(a["auth_pred_as_forged"]),
            forg_pred_as_auth=int(a["forg_pred_as_auth"]),
            decode_errors_scoring=0,
        )
        results.append((thr, summary))
        if mean_all > best_score:
            best_score = mean_all
            best_thr = thr

    assert best_thr is not None
    results.sort(key=lambda x: x[0])
    print("\n[tune] Results (val subset):")
    for thr, s in results:
        print(
            f"thr={thr:.2f} mean={s.mean_score:.4f} mean_forged={s.mean_forged:.4f} "
            f"auth_pred_as_forged={s.auth_pred_as_forged} forg_pred_as_auth={s.forg_pred_as_auth}"
        )

    best_summary = dict(results)[best_thr]
    best_overrides = list(base_overrides) + [f"inference.postprocess.prob_threshold={best_thr}"]
    print(
        f"\n[tune] BEST thr={best_thr:.2f} mean={best_summary.mean_score:.4f} "
        f"mean_forged={best_summary.mean_forged:.4f}"
    )
    print("[tune] Suggested overrides:")
    print(json.dumps(best_overrides, indent=2, ensure_ascii=False))

    tuned_path = OUT_DIR / "tuned_postprocess.json"
    tuned_path.write_text(
        json.dumps(
            {
                "config": str(TUNE_CONFIG),
                "split": str(TUNE_SPLIT),
                "val_fraction": float(TUNE_VAL_FRACTION),
                "seed": int(TUNE_SEED),
                "limit": int(TUNE_LIMIT),
                "best_threshold": float(best_thr),
                "best_summary": best_summary.as_dict(),
                "overrides": best_overrides,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[tune] Wrote {tuned_path}")

    tuned_config_path = None
    if TUNE_WRITE_TUNED_CONFIG:

        def _slug_float(x: float) -> str:
            s = f"{float(x):.4f}".rstrip("0").rstrip(".")
            return s.replace(".", "p")

        tuned_cfg = load_config_data(TUNE_CONFIG)
        tuned_cfg = apply_overrides(tuned_cfg, best_overrides)
        tuned_config_path = OUT_DIR / f"tuned_{Path(TUNE_CONFIG).stem}_thr{_slug_float(best_thr)}.json"
        tuned_config_path.write_text(json.dumps(tuned_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[tune] Wrote {tuned_config_path}")

# %%
# -------------------------
# Package Kaggle Dataset folder
# -------------------------
#
# Cria um folder pronto para upload como Kaggle Dataset:
# - código (src/scripts/configs/notebooks/docs)
# - + `outputs/models/*.pth` (opcional)
#
# Depois, anexe esse dataset no notebook de submissão (internet OFF).

if DO_PACKAGE:
    out_root = package_kaggle_dataset(
        out_dir=PKG_OUT,
        include_models=True,
        models_dir=OUT_MODELS,
        repo_root=CODE_ROOT,
    )
    print(f"Wrote Kaggle bundle at: {out_root.resolve()}")

    if TUNE_POSTPROCESS and "tuned_config_path" in globals():
        p = globals().get("tuned_config_path")
        if isinstance(p, Path) and p.exists():
            dst = out_root / "configs" / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            print(f"Copied tuned config into bundle: {dst}")

    print("Crie um Kaggle Dataset a partir desse folder e anexe no notebook de submissão offline.")
