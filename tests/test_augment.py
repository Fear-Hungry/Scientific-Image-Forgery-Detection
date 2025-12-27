import numpy as np
import pytest

from forgeryseg.augment import CopyMoveTransform, get_train_augment, get_val_augment
from forgeryseg.dataset import build_supplemental_index


def test_train_augment_preserves_shape_and_binary_mask():
    h = w = 128
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[32:96, 32:96] = 1

    try:
        aug = get_train_augment(patch_size=(h, w))
    except ImportError:
        pytest.skip("albumentations is not installed")

    out = aug(image=image, mask=mask)
    assert out["image"].shape == image.shape
    assert out["mask"].shape == mask.shape
    assert set(np.unique(out["mask"]).tolist()).issubset({0, 1})


def test_copy_move_transform_adds_mask_on_empty_input():
    h = w = 96
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    try:
        import albumentations as A
    except ImportError:
        pytest.skip("albumentations is not installed")
    if CopyMoveTransform is None:
        pytest.skip("CopyMoveTransform requires albumentations")

    aug = A.Compose([CopyMoveTransform(p=1.0, rotation_limit=0.0, scale_range=(1.0, 1.0))])
    out = aug(image=image, mask=mask)
    assert out["mask"].shape == mask.shape
    assert set(np.unique(out["mask"]).tolist()).issubset({0, 1})
    assert int(out["mask"].sum()) > 0


def test_copy_move_transform_skips_when_mask_non_empty():
    h = w = 96
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:30, 20:40] = 1

    try:
        import albumentations as A
    except ImportError:
        pytest.skip("albumentations is not installed")
    if CopyMoveTransform is None:
        pytest.skip("CopyMoveTransform requires albumentations")

    aug = A.Compose([CopyMoveTransform(p=1.0)])
    out = aug(image=image, mask=mask)
    assert np.array_equal(out["mask"], mask)


def test_val_augment_is_identity():
    h = w = 64
    rng = np.random.default_rng(321)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = rng.integers(0, 2, size=(h, w), dtype=np.uint8)

    try:
        aug = get_val_augment()
    except ImportError:
        pytest.skip("albumentations is not installed")

    out = aug(image=image, mask=mask)
    assert np.array_equal(out["image"], image)
    assert np.array_equal(out["mask"], mask)


def test_build_supplemental_index_sets_is_authentic_false(tmp_path):
    from PIL import Image

    (tmp_path / "supplemental_images").mkdir(parents=True)
    (tmp_path / "supplemental_masks").mkdir(parents=True)

    case_id = "case_1"
    image_path = tmp_path / "supplemental_images" / f"{case_id}.png"
    mask_path = tmp_path / "supplemental_masks" / f"{case_id}.npy"

    img = np.zeros((10, 12, 3), dtype=np.uint8)
    Image.fromarray(img).save(image_path)

    mask = np.zeros((1, 10, 12), dtype=np.uint8)
    mask[0, 2:5, 3:7] = 1
    np.save(mask_path, mask)

    samples = build_supplemental_index(tmp_path, strict=True)
    assert len(samples) == 1
    assert samples[0].case_id == case_id
    assert samples[0].is_authentic is False
    assert samples[0].mask_path == mask_path
