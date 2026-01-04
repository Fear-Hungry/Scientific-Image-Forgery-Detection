from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from ..dataset import load_mask_instances
from ..inference import TilingParams, default_tta, load_rgb, predict_prob_map, predict_prob_map_tiled
from ..metric import of1_score
from ..postprocess import PostprocessParams, postprocess_prob
from ..tta import IdentityTTA
from ..typing import Case


@dataclass(frozen=True)
class EvalSummary:
    mean_of1: float
    mean_authentic: float
    mean_forged: float
    n_cases: int
    n_authentic: int
    n_forged: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_of1": float(self.mean_of1),
            "mean_authentic": float(self.mean_authentic),
            "mean_forged": float(self.mean_forged),
            "n_cases": int(self.n_cases),
            "n_authentic": int(self.n_authentic),
            "n_forged": int(self.n_forged),
        }


@torch.no_grad()
def evaluate_of1(
    model: torch.nn.Module,
    cases: Sequence[Case],
    *,
    device: torch.device,
    input_size: int,
    postprocess: PostprocessParams,
    tiling: TilingParams | None = None,
    use_tta: bool = False,
    progress: bool = True,
) -> EvalSummary:
    """
    Compute competition-like oF1 on a labeled split (train/supplemental).

    Notes:
    - Uses the same inference primitives as submission generation (letterbox + optional tiling).
    - If `use_tta=False`, runs with Identity-only for speed.
    """
    model.eval()

    if use_tta:
        tta_transforms, tta_weights = default_tta()
    else:
        tta_transforms, tta_weights = [IdentityTTA()], [1.0]

    scores_all: list[float] = []
    scores_auth: list[float] = []
    scores_forg: list[float] = []

    it = tqdm(cases, desc="Eval oF1") if progress else cases
    for case in it:
        image = load_rgb(case.image_path)
        if tiling is None:
            prob = predict_prob_map(
                model,
                image,
                input_size=int(input_size),
                device=device,
                tta_transforms=tta_transforms,
                tta_weights=tta_weights,
            )
        else:
            prob = predict_prob_map_tiled(
                model,
                image,
                input_size=int(input_size),
                device=device,
                tiling=tiling,
                tta_transforms=tta_transforms,
                tta_weights=tta_weights,
            )

        pred_instances = postprocess_prob(prob, postprocess)
        gt_instances = [] if case.mask_path is None else load_mask_instances(case.mask_path)
        s = float(of1_score(pred_instances, gt_instances))

        scores_all.append(s)
        if case.mask_path is None:
            scores_auth.append(s)
        else:
            scores_forg.append(s)

    def _mean(x: list[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    return EvalSummary(
        mean_of1=_mean(scores_all),
        mean_authentic=_mean(scores_auth),
        mean_forged=_mean(scores_forg),
        n_cases=len(scores_all),
        n_authentic=len(scores_auth),
        n_forged=len(scores_forg),
    )
