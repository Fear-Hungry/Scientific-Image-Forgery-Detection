from __future__ import annotations

import numpy as np
import torch

from forgeryseg.inference_engine import InferenceEngine
from forgeryseg.postprocess import PostprocessParams


class DummyFFTGate:
    def __init__(self, *, p_forged: float, threshold: float) -> None:
        self._p_forged = float(p_forged)
        self.threshold = float(threshold)

    def predict_prob_forged(self, image_rgb: np.ndarray) -> float:  # noqa: ARG002
        return float(self._p_forged)


def _make_engine(*, postprocess: PostprocessParams, gate: DummyFFTGate | None) -> InferenceEngine:
    return InferenceEngine(
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        input_size=8,
        postprocess=postprocess,
        tta_transforms=[],
        tta_weights=[],
        tiling=None,
        batch_size=1,
        fft_gate=gate,  # type: ignore[arg-type]
        amp=False,
    )


def test_fft_gate_overrides_authentic_heuristic_when_high_prob() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    prob = np.zeros((8, 8), dtype=np.float32)
    prob[0:2, 0:2] = 0.6  # area=4, mean_conf=0.6

    strict = PostprocessParams(prob_threshold=0.5, authentic_area_max=10, authentic_conf_max=0.7)
    engine = _make_engine(postprocess=strict, gate=DummyFFTGate(p_forged=0.9, threshold=0.5))

    instances, overridden = engine._maybe_apply_fft_gate(image, prob, [])
    assert overridden is True
    assert len(instances) == 1


def test_fft_gate_does_not_override_when_low_prob() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    prob = np.zeros((8, 8), dtype=np.float32)

    strict = PostprocessParams(prob_threshold=0.5, authentic_area_max=10, authentic_conf_max=0.7)
    engine = _make_engine(postprocess=strict, gate=DummyFFTGate(p_forged=0.1, threshold=0.5))

    instances, overridden = engine._maybe_apply_fft_gate(image, prob, [])
    assert overridden is False
    assert instances == []


def test_fft_gate_does_not_override_when_instances_exist() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    prob = np.zeros((8, 8), dtype=np.float32)
    inst = np.zeros((8, 8), dtype=np.uint8)
    inst[0:2, 0:2] = 1

    strict = PostprocessParams(prob_threshold=0.5, authentic_area_max=10, authentic_conf_max=0.7)
    engine = _make_engine(postprocess=strict, gate=DummyFFTGate(p_forged=0.9, threshold=0.5))

    instances, overridden = engine._maybe_apply_fft_gate(image, prob, [inst])
    assert overridden is False
    assert len(instances) == 1
    assert int(instances[0].sum()) == 4
