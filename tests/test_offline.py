import os


def test_configure_cache_dirs_overrides_env_and_creates_dirs(tmp_path, monkeypatch):
    from forgeryseg.offline import configure_cache_dirs

    monkeypatch.setenv("TORCH_HOME", "/tmp/should_be_overridden")
    monkeypatch.setenv("HF_HOME", "/tmp/should_be_overridden")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/tmp/should_be_overridden")

    cache_root = tmp_path / "cache_root"
    configure_cache_dirs(cache_root)

    assert os.environ["TORCH_HOME"] == str(cache_root / "torch")
    assert os.environ["HF_HOME"] == str(cache_root / "hf")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(cache_root / "hf" / "hub")
    assert (cache_root / "torch").is_dir()
    assert (cache_root / "hf" / "hub").is_dir()


def test_configure_cache_dirs_force_false_respects_existing_env(tmp_path, monkeypatch):
    from forgeryseg.offline import configure_cache_dirs

    monkeypatch.setenv("HF_HOME", str(tmp_path / "existing_hf"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "existing_torch"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "existing_hub"))

    cache_root = tmp_path / "cache_root"
    configure_cache_dirs(cache_root, force=False)

    assert os.environ["TORCH_HOME"] == str(tmp_path / "existing_torch")
    assert os.environ["HF_HOME"] == str(tmp_path / "existing_hf")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(tmp_path / "existing_hub")

