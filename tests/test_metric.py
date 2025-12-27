import numpy as np
import pytest

pytest.importorskip("scipy")

from forgeryseg.metric import score_image


def test_score_empty_masks():
    assert score_image([], []) == 1.0


def test_score_perfect_match():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 1
    assert score_image([mask], [mask]) == 1.0


def test_score_overseg_penalty():
    gt = np.zeros((3, 3), dtype=np.uint8)
    gt[0, 0] = 1
    pred1 = gt.copy()
    pred2 = np.zeros((3, 3), dtype=np.uint8)
    pred2[2, 2] = 1

    score_one = score_image([gt], [pred1])
    score_two = score_image([gt], [pred1, pred2])

    assert score_one == 1.0
    assert score_two == 0.5


def test_score_missing_instance_penalized():
    gt1 = np.zeros((3, 3), dtype=np.uint8)
    gt1[0, 0] = 1
    gt2 = np.zeros((3, 3), dtype=np.uint8)
    gt2[0, 1] = 1
    pred = gt1.copy()

    score = score_image([gt1, gt2], [pred])
    assert score == pytest.approx(0.5)


def test_score_missing_prediction():
    gt = np.zeros((3, 3), dtype=np.uint8)
    gt[0, 0] = 1
    assert score_image([gt], []) == 0.0
