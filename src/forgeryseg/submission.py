from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .dataset import list_cases
from .inference_engine import InferenceEngine
from .typing import Pathish, Split


@dataclass(frozen=True)
class SubmissionStats:
    out_path: Path
    n_rows: int
    n_authentic: int
    fft_gate_overrides: int = 0


def list_ordered_cases(data_root: Path, split: Split) -> list:
    """
    List cases for a split.

    For `split="test"`, if `sample_submission.csv` exists, it is used to define the expected
    `case_id` set and ordering.
    """
    cases = list_cases(data_root, split, include_authentic=True, include_forged=True)
    if split != "test":
        return cases

    sample_path = data_root / "sample_submission.csv"
    if not sample_path.exists():
        return cases

    sample = pd.read_csv(sample_path)
    if "case_id" not in sample.columns:
        return cases
    case_ids = sample["case_id"].astype(str).tolist()
    case_by_id = {c.case_id: c for c in cases}
    missing = [cid for cid in case_ids if cid not in case_by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} case_id(s) do sample_submission não foram encontrados em {split}_images "
            f"(ex.: {missing[:5]}). Verifique o data_root."
        )
    return [case_by_id[cid] for cid in case_ids]


@dataclass
class SubmissionWriter:
    config_path: Path
    data_root: Path
    split: Split
    out_path: Path
    device: torch.device
    limit: int = 0
    overrides: list[str] | None = None
    path_roots: list[Path] | None = None
    amp: bool = False

    def run(self) -> SubmissionStats:
        ordered = list_ordered_cases(self.data_root, self.split)
        if self.limit and self.limit > 0:
            ordered = ordered[: int(self.limit)]

        engine = InferenceEngine.from_config(
            config_path=self.config_path,
            device=self.device,
            overrides=self.overrides,
            path_roots=self.path_roots,
            amp=bool(self.amp),
        )

        image_paths = [c.image_path for c in ordered]
        anns, n_overrides = engine.predict_annotations(image_paths)

        rows: list[dict[str, str]] = []
        for case, ann in tqdm(zip(ordered, anns, strict=True), total=len(ordered), desc="Writing submission"):
            rows.append({"case_id": case.case_id, "annotation": ann})

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(rows)
        out_df.to_csv(self.out_path, index=False)
        n_auth = int((out_df["annotation"] == "authentic").sum())
        print(f"Wrote {self.out_path} ({n_auth}/{len(out_df)} authentic)")
        if n_overrides:
            print(f"fft_gate overrides: {n_overrides}")

        return SubmissionStats(
            out_path=self.out_path,
            n_rows=int(len(out_df)),
            n_authentic=n_auth,
            fft_gate_overrides=int(n_overrides),
        )


def write_submission_csv(
    *,
    config_path: Pathish,
    data_root: Pathish,
    split: Split = "test",
    out_path: Pathish,
    device: torch.device,
    limit: int = 0,
    overrides: list[str] | None = None,
    path_roots: list[Path] | None = None,
    amp: bool = False,
) -> SubmissionStats:
    """
    Convenience wrapper around SubmissionWriter.
    """
    writer = SubmissionWriter(
        config_path=Path(config_path),
        data_root=Path(data_root),
        split=split,
        out_path=Path(out_path),
        device=device,
        limit=int(limit),
        overrides=overrides,
        path_roots=path_roots,
        amp=bool(amp),
    )
    return writer.run()
