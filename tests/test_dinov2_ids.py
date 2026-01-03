from __future__ import annotations


def test_normalize_model_id_maps_legacy_ids():
    from forgeryseg.models import dinov2

    assert dinov2._normalize_model_id("dinov2") == "facebook/dinov2-base"
    assert dinov2._normalize_model_id("metaresearch/dinov2") == "facebook/dinov2-base"
    assert dinov2._normalize_model_id("facebookresearch/dinov2") == "facebook/dinov2-base"


def test_normalize_model_id_preserves_existing_local_paths(tmp_path, monkeypatch):
    from forgeryseg.models import dinov2

    (tmp_path / "metaresearch" / "dinov2").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    assert dinov2._normalize_model_id("metaresearch/dinov2") == "metaresearch/dinov2"

