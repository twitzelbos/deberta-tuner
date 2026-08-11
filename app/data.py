"""JSONL parsing and per-task validation.

Every failure carries the offending line number: a caller who uploads 50k rows
with one bad record needs to know which one.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import TaskType


class DataError(ValueError):
    """Raised for malformed or inconsistent training data."""


@dataclass
class Dataset:
    task: TaskType
    rows: list[dict[str, Any]]
    labels: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataError(f"line {lineno}: invalid JSON ({exc.msg})") from None
            if not isinstance(obj, dict):
                raise DataError(f"line {lineno}: expected a JSON object")
            yield lineno, obj


def _need_text(obj: dict[str, Any], lineno: int) -> str:
    text = obj.get("text")
    if not isinstance(text, str) or not text.strip():
        raise DataError(f"line {lineno}: 'text' must be a non-empty string")
    return text


def parse(path: Path, task: TaskType) -> Dataset:
    """Read a JSONL file and validate it against the task's expected schema."""
    rows: list[dict[str, Any]] = []
    labels: set[str] = set()

    for lineno, obj in _iter_jsonl(path):
        if task is TaskType.TOKEN_CLASSIFICATION:
            tokens, tags = obj.get("tokens"), obj.get("tags")
            if not isinstance(tokens, list) or not tokens:
                raise DataError(f"line {lineno}: 'tokens' must be a non-empty list")
            if not isinstance(tags, list):
                raise DataError(f"line {lineno}: 'tags' must be a list")
            if len(tokens) != len(tags):
                raise DataError(
                    f"line {lineno}: {len(tokens)} tokens but {len(tags)} tags"
                )
            if not all(isinstance(t, str) for t in tokens):
                raise DataError(f"line {lineno}: every token must be a string")
            if not all(isinstance(t, str) for t in tags):
                raise DataError(f"line {lineno}: every tag must be a string")
            labels.update(tags)
            rows.append({"tokens": tokens, "tags": tags})

        elif task is TaskType.MULTI_LABEL_CLASSIFICATION:
            text = _need_text(obj, lineno)
            raw = obj.get("labels")
            if not isinstance(raw, list):
                raise DataError(f"line {lineno}: 'labels' must be a list of strings")
            if not all(isinstance(x, str) for x in raw):
                raise DataError(f"line {lineno}: every entry of 'labels' must be a string")
            labels.update(raw)
            rows.append({"text": text, "text_pair": obj.get("text_pair"), "labels": raw})

        elif task is TaskType.REGRESSION:
            text = _need_text(obj, lineno)
            val = obj.get("label")
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise DataError(f"line {lineno}: 'label' must be a number for regression")
            rows.append(
                {"text": text, "text_pair": obj.get("text_pair"), "label": float(val)}
            )

        else:  # SEQUENCE_CLASSIFICATION
            text = _need_text(obj, lineno)
            val = obj.get("label")
            if isinstance(val, bool) or not isinstance(val, (str, int)):
                raise DataError(f"line {lineno}: 'label' must be a string or integer")
            label = str(val)
            labels.add(label)
            rows.append({"text": text, "text_pair": obj.get("text_pair"), "label": label})

    if not rows:
        raise DataError("file contains no records")

    if task is TaskType.REGRESSION:
        ordered: list[str] = []
    else:
        ordered = sorted(labels)
        if task is TaskType.TOKEN_CLASSIFICATION and "O" in ordered:
            # Keep the outside tag at index 0; it is the natural pad/ignore label.
            ordered = ["O"] + [x for x in ordered if x != "O"]
        if len(ordered) < 2 and task is TaskType.SEQUENCE_CLASSIFICATION:
            raise DataError(
                f"need at least 2 distinct labels, found {ordered or 'none'}"
            )

    return Dataset(task=task, rows=rows, labels=ordered)


def split(ds: Dataset, eval_fraction: float, seed: int) -> tuple[Dataset, Dataset | None]:
    """Hold out a random slice for evaluation. Returns (train, eval|None)."""
    if eval_fraction <= 0 or len(ds) < 4:
        return ds, None
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    n_eval = max(1, int(len(ds) * eval_fraction))
    if len(ds) - n_eval < 1:
        return ds, None
    eval_rows = [ds.rows[i] for i in idx[:n_eval]]
    train_rows = [ds.rows[i] for i in idx[n_eval:]]
    return (
        Dataset(ds.task, train_rows, ds.labels),
        Dataset(ds.task, eval_rows, ds.labels),
    )


def merge_labels(train: Dataset, evl: Dataset | None) -> list[str]:
    """Union the label sets so an eval-only label does not blow up at scoring."""
    if evl is None:
        return train.labels
    combined = sorted(set(train.labels) | set(evl.labels))
    if train.task is TaskType.TOKEN_CLASSIFICATION and "O" in combined:
        combined = ["O"] + [x for x in combined if x != "O"]
    return combined
