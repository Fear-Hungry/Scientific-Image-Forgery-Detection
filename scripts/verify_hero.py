#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from forgeryseg.checkpoints import build_segmentation_from_config, load_checkpoint
from forgeryseg.models.correlation import SmpCorrelationWrapper


def main() -> int:
    print(">>> Verifying HERO model construction...")

    config_path = PROJECT_ROOT / "configs" / "seg_hero_effnet_b7.json"
    if not config_path.exists():
        print(f"FAIL: Config not found: {config_path}")
        return 1

    cfg = json.loads(config_path.read_text())
    cfg["backend"] = "smp"

    try:
        model = build_segmentation_from_config(cfg)
    except Exception as exc:
        print(f"FAIL: Model build failed: {exc}")
        traceback.print_exc()
        return 1

    print(f"SUCCESS: Model built: {type(model)}")

    if not isinstance(model, SmpCorrelationWrapper):
        print("FAIL: Expected SmpCorrelationWrapper (use_correlation=true)")
        return 1
    print("SUCCESS: SmpCorrelationWrapper detected")

    in_channels = int(cfg.get("in_channels", 3))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Testing forward pass (device={device}, B=1, C={in_channels}, H=64, W=64)...")
    model = model.to(device).eval()

    dummy = torch.randn(1, in_channels, 64, 64, device=device)
    try:
        with torch.no_grad():
            out = model(dummy)
    except Exception as exc:
        print(f"FAIL: Forward pass failed: {exc}")
        traceback.print_exc()
        return 1
    print(f"SUCCESS: Forward pass output shape: {tuple(out.shape)}")

    print(">>> Testing save/load cycle...")
    fd, tmp_name = tempfile.mkstemp(prefix="hero_ckpt_", suffix=".pt")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        torch.save({"model_state": model.state_dict(), "config": cfg}, tmp_path)
        state, loaded_cfg = load_checkpoint(tmp_path)
        model_loaded = build_segmentation_from_config(loaded_cfg)
        model_loaded.load_state_dict(state)
    except Exception as exc:
        print(f"FAIL: Save/load cycle failed: {exc}")
        traceback.print_exc()
        return 1
    finally:
        tmp_path.unlink(missing_ok=True)

    if not isinstance(model_loaded, SmpCorrelationWrapper):
        print("FAIL: Loaded model lost SmpCorrelationWrapper")
        return 1
    print("SUCCESS: Loaded model preserved SmpCorrelationWrapper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
