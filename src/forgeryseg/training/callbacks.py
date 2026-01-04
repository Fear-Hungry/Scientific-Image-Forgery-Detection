from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Callback:
    def on_fold_start(self, *, fold: int, folds: int) -> None:  # noqa: D401
        """Called at the start of each fold."""

    def on_epoch_end(self, metrics: dict[str, Any]) -> None:
        pass

    def on_fold_end(self, summary: dict[str, Any]) -> None:
        pass

    def on_train_end(self, summary: dict[str, Any]) -> None:
        pass


@dataclass
class CSVLoggerCallback(Callback):
    path: Path
    fieldnames: list[str] | None = None

    _file: Any = None
    _writer: csv.DictWriter | None = None

    def on_fold_start(self, *, fold: int, folds: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")

    def on_epoch_end(self, metrics: dict[str, Any]) -> None:
        if self._file is None:
            return
        if self._writer is None:
            if self.fieldnames is None:
                self.fieldnames = list(metrics.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
        row = {k: metrics.get(k) for k in (self.fieldnames or metrics.keys())}
        self._writer.writerow(row)
        self._file.flush()

    def on_fold_end(self, summary: dict[str, Any]) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None


@dataclass
class JSONLoggerCallback(Callback):
    path: Path
    _items: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._items = []

    def on_epoch_end(self, metrics: dict[str, Any]) -> None:
        self._items.append(dict(metrics))

    def on_fold_end(self, summary: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"epochs": self._items, "summary": summary}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._items = []
