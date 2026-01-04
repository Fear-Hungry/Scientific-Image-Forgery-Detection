from __future__ import annotations

import numpy as np
import pytest

from forgeryseg.metric import of1_score


pytest.importorskip("scipy")


def test_metric_authentic_cases() -> None:
    assert of1_score([], []) == 1.0
    assert of1_score([np.ones((3, 3), dtype=np.uint8)], []) == 0.0
    assert of1_score([], [np.ones((3, 3), dtype=np.uint8)]) == 0.0


def test_metric_single_instance_perfect() -> None:
    m = np.zeros((5, 5), dtype=np.uint8)
    m[1:4, 2:4] = 1
    assert of1_score([m], [m]) == 1.0


def test_metric_missing_instance_penalized() -> None:
    m1 = np.zeros((5, 5), dtype=np.uint8)
    m1[0:2, 0:2] = 1
    m2 = np.zeros((5, 5), dtype=np.uint8)
    m2[3:5, 3:5] = 1
    assert of1_score([m1], [m1, m2]) == 0.5


def test_metric_extra_instance_penalized() -> None:
    m1 = np.zeros((5, 5), dtype=np.uint8)
    m1[0:2, 0:2] = 1
    m2 = np.zeros((5, 5), dtype=np.uint8)
    m2[3:5, 3:5] = 1
    m3 = np.zeros((5, 5), dtype=np.uint8)
    m3[2:3, 2:3] = 1

    score = of1_score([m1, m2, m3], [m1, m2])
    assert score == pytest.approx(2.0 / 3.0, abs=1e-6)
