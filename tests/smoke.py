"""End-to-end smoke test: every task type, tiny data, CPU, real DeBERTa weights.

    TUNER_FORCE_CPU=1 python -m tests.smoke

Uses deberta-v3-xsmall so the whole thing finishes in a couple of minutes
without touching the GPU (which SGLang is holding).
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TUNER_FORCE_CPU", "1")
os.environ.setdefault("TUNER_SGLANG_CONTROL", "0")
_TMP = tempfile.mkdtemp(prefix="tuner-smoke-")
os.environ["TUNER_STATE_DIR"] = _TMP

from app import data, training                      # noqa: E402
from app.schemas import JobConfig, TaskType         # noqa: E402

BASE = "microsoft/deberta-v3-xsmall"
POS = ["great value", "loved it", "works perfectly", "excellent build", "very happy"]
NEG = ["broke instantly", "terrible quality", "waste of money", "awful", "returned it"]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def make_rows(task: TaskType) -> list[dict]:
    rng = random.Random(0)
    rows: list[dict] = []
    for i in range(24):
        if task is TaskType.SEQUENCE_CLASSIFICATION:
            pos = i % 2 == 0
            rows.append({"text": rng.choice(POS if pos else NEG),
                         "label": "positive" if pos else "negative"})
        elif task is TaskType.MULTI_LABEL_CLASSIFICATION:
            labels = ["quality"] if i % 2 else ["price", "quality"]
            rows.append({"text": rng.choice(POS + NEG), "labels": labels})
        elif task is TaskType.REGRESSION:
            pos = i % 2 == 0
            rows.append({"text": rng.choice(POS if pos else NEG),
                         "label": 4.5 if pos else 1.2})
        else:  # TOKEN_CLASSIFICATION
            rows.append({
                "tokens": ["Ada", "works", "at", "Acme", "in", "Berlin"],
                "tags": ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"],
            })
    return rows


def run_task(task: TaskType, workdir: Path) -> dict:
    print(f"\n{'=' * 60}\n{task.value}\n{'=' * 60}", flush=True)
    jsonl = workdir / f"{task.value}.jsonl"
    _write(jsonl, make_rows(task))

    ds = data.parse(jsonl, task)
    train_ds, eval_ds = data.split(ds, 0.25, 42)
    labels = data.merge_labels(train_ds, eval_ds)
    print(f"parsed {len(ds)} rows, labels={labels}")

    # Checkpointing off: this suite is about the four task pipelines, and
    # tests/test_resume.py covers checkpoint/resume/preemption on its own.
    cfg = JobConfig(base_model=BASE, task=task, epochs=1, batch_size=4,
                    max_length=64, eval_split=0.25, checkpoint_every_epochs=0)
    metrics = training.run(
        train_ds=train_ds, eval_ds=eval_ds, labels=labels, cfg=cfg,
        out_dir=workdir / task.value / "output",
        ckpt_dir=workdir / task.value / "checkpoint",
        log=lambda m: print("  " + m, flush=True),
        on_progress=lambda p: None,
        on_checkpoint=lambda e: None,
        should_cancel=lambda: False,
        should_stop=lambda: False,
        should_yield=lambda: False,
    )

    out = workdir / task.value / "output"
    assert (out / "config.json").exists(), "model config not saved"
    weights = list(out.glob("*.safetensors")) + list(out.glob("*.bin"))
    assert weights, f"no weight file written to {out}"
    assert metrics, "no metrics returned"
    print(f"PASS {task.value}: {json.dumps(metrics)}")
    return metrics


def test_validation_errors() -> None:
    print(f"\n{'=' * 60}\nvalidation\n{'=' * 60}", flush=True)
    tmp = Path(_TMP)
    cases = [
        ('{"text": "hi", "label": "a"}\n{"text": "", "label": "b"}',
         TaskType.SEQUENCE_CLASSIFICATION, "line 2"),
        ('{"tokens": ["a","b"], "tags": ["O"]}',
         TaskType.TOKEN_CLASSIFICATION, "2 tokens but 1 tags"),
        ('{"text": "hi", "label": "notanumber"}',
         TaskType.REGRESSION, "must be a number"),
        ('not json', TaskType.SEQUENCE_CLASSIFICATION, "invalid JSON"),
        ('{"text": "hi", "label": "only-one-class"}',
         TaskType.SEQUENCE_CLASSIFICATION, "at least 2 distinct labels"),
    ]
    for i, (body, task, expect) in enumerate(cases):
        p = tmp / f"bad{i}.jsonl"
        p.write_text(body, encoding="utf-8")
        try:
            data.parse(p, task)
        except data.DataError as exc:
            assert expect in str(exc), f"expected {expect!r}, got {exc}"
            print(f"  ok: {exc}")
        else:
            raise AssertionError(f"case {i} should have raised")
    print("PASS validation")


def test_roc_auc() -> None:
    """AUC is threshold-free, so it is asserted against crafted scores."""
    import numpy as np

    print(f"\n{'=' * 60}\nroc_auc\n{'=' * 60}", flush=True)

    # Perfectly separable binary scores -> AUC 1.0, regardless of threshold.
    logits = np.array([[3.0, -3.0]] * 4 + [[-3.0, 3.0]] * 4)
    gold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    m = training._metrics(TaskType.SEQUENCE_CLASSIFICATION, logits, gold, ["a", "b"], 0.5)
    assert m["roc_auc"] == 1.0, m
    print(f"  ok: separable -> roc_auc={m['roc_auc']}")

    # Exactly inverted ranking -> 0.0, while accuracy also collapses.
    m = training._metrics(TaskType.SEQUENCE_CLASSIFICATION, logits, 1 - gold, ["a", "b"], 0.5)
    assert m["roc_auc"] == 0.0, m
    print(f"  ok: inverted  -> roc_auc={m['roc_auc']}")

    # AUC sees ranking that accuracy at 0.5 cannot: all predictions land on one
    # side of the threshold, yet the positives are ranked above the negatives.
    skew = np.array([[2.0, -1.0], [2.0, -0.9], [2.0, -0.8], [2.0, -0.7]])
    m = training._metrics(TaskType.SEQUENCE_CLASSIFICATION, skew,
                          np.array([0, 0, 1, 1]), ["a", "b"], 0.5)
    assert m["accuracy"] == 0.5 and m["roc_auc"] == 1.0, m
    print(f"  ok: accuracy={m['accuracy']} but roc_auc={m['roc_auc']}")

    # Undefined with a single class present: omitted rather than faked.
    m = training._metrics(TaskType.SEQUENCE_CLASSIFICATION, logits,
                          np.zeros(8, dtype=int), ["a", "b"], 0.5)
    assert "roc_auc" not in m, m
    print("  ok: single-class holdout -> roc_auc omitted")

    # Multi-label reports a macro AUC over the label columns.
    ml_logits = np.array([[3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [-3.0, 3.0]])
    ml_gold = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    m = training._metrics(TaskType.MULTI_LABEL_CLASSIFICATION, ml_logits, ml_gold,
                          ["x", "y"], 0.5)
    assert m["roc_auc_macro"] == 1.0, m
    print(f"  ok: multi-label -> roc_auc_macro={m['roc_auc_macro']}")

    # Regression has no notion of ranking classes.
    m = training._metrics(TaskType.REGRESSION, np.array([[1.0], [2.0]]),
                          np.array([1.0, 2.0]), [], 0.5)
    assert "roc_auc" not in m, m
    print("  ok: regression -> no roc_auc")
    print("PASS roc_auc")


def main() -> int:
    workdir = Path(_TMP)
    test_validation_errors()
    test_roc_auc()
    results = {t.value: run_task(t, workdir) for t in TaskType}
    print(f"\n{'=' * 60}\nALL PASSED\n{'=' * 60}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
