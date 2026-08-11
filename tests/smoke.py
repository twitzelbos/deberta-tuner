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


def main() -> int:
    workdir = Path(_TMP)
    test_validation_errors()
    results = {t.value: run_task(t, workdir) for t in TaskType}
    print(f"\n{'=' * 60}\nALL PASSED\n{'=' * 60}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
