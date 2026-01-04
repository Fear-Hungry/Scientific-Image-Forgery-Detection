from __future__ import annotations

import numpy as np

from forgeryseg.postprocess import PostprocessParams, postprocess_prob


def test_postprocess_min_area_filters_small_components() -> None:
    prob = np.zeros((10, 10), dtype=np.float32)
    prob[0:2, 0:2] = 0.9  # area=4
    prob[5:9, 5:9] = 0.9  # area=16

    params = PostprocessParams(prob_threshold=0.5, min_area=10)
    instances = postprocess_prob(prob, params)
    assert len(instances) == 1
    assert int(instances[0].sum()) == 16


def test_postprocess_small_area_min_mean_conf_filters_small_low_conf() -> None:
    prob = np.zeros((6, 6), dtype=np.float32)
    prob[0:2, 0:2] = 0.6  # area=4, mean_conf=0.6

    params = PostprocessParams(prob_threshold=0.5, small_area=10, small_min_mean_conf=0.8)
    assert postprocess_prob(prob, params) == []

    params_relaxed = PostprocessParams(prob_threshold=0.5, small_area=10, small_min_mean_conf=0.5)
    instances = postprocess_prob(prob, params_relaxed)
    assert len(instances) == 1
    assert int(instances[0].sum()) == 4


def test_postprocess_authentic_heuristic_discards_small_low_conf_union() -> None:
    prob = np.zeros((6, 6), dtype=np.float32)
    prob[1:3, 1:3] = 0.6  # area=4, mean_conf=0.6

    params = PostprocessParams(
        prob_threshold=0.5,
        authentic_area_max=10,
        authentic_conf_max=0.7,
    )
    assert postprocess_prob(prob, params) == []

    params_relaxed = PostprocessParams(prob_threshold=0.5)
    instances = postprocess_prob(prob, params_relaxed)
    assert len(instances) == 1
    assert int(instances[0].sum()) == 4


def test_postprocess_min_prob_std_returns_empty_for_flat_maps() -> None:
    prob = np.full((8, 8), 0.2, dtype=np.float32)
    params = PostprocessParams(prob_threshold=0.1, min_prob_std=0.01)
    assert postprocess_prob(prob, params) == []
