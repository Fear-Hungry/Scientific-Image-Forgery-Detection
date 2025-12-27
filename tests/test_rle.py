import json

import numpy as np

from forgeryseg.rle import decode_annotation, encode_instances, rle_decode, rle_encode


def test_rle_column_major_order():
    mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    assert rle_encode(mask) == [1, 2, 4, 1]


def test_rle_round_trip():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, mask.shape)
    assert np.array_equal(mask, decoded)


def test_encode_instances_authentic():
    mask = np.zeros((2, 2), dtype=np.uint8)
    assert encode_instances([mask]) == "authentic"


def test_decode_annotation_multiple_instances():
    mask1 = np.zeros((2, 2), dtype=np.uint8)
    mask1[0, 0] = 1
    mask2 = np.zeros((2, 2), dtype=np.uint8)
    mask2[1, 1] = 1
    annotation = ";".join([json.dumps(rle_encode(mask1)), json.dumps(rle_encode(mask2))])
    decoded = decode_annotation(annotation, (2, 2))
    assert len(decoded) == 2
    assert np.array_equal(decoded[0], mask1)
    assert np.array_equal(decoded[1], mask2)
