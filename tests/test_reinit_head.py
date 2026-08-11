"""Fine-tuning an NLI checkpoint onto a different label set.

    python -m tests.test_reinit_head

Covers the real motivating case: a 3-class entailment checkpoint adapted to a
2-label task. Runs on CPU.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["TUNER_FORCE_CPU"] = "1"
os.environ["TUNER_SGLANG_CONTROL"] = "0"
_TMP = tempfile.mkdtemp(prefix="tuner-reinit-")
os.environ["TUNER_STATE_DIR"] = _TMP

from app import data, training  # noqa: E402
from app.schemas import JobConfig, TaskType  # noqa: E402

# 3-class NLI head (entailment/neutral/contradiction) vs a 2-label job.
NLI_BASE = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
PLAIN_BASE = "microsoft/deberta-v3-xsmall"

ROWS = [
    {"text": t, "text_pair": p, "label": lab}
    for t, p, lab in [
        ("The cat sat on the mat.", "A cat is on a mat.", "supported"),
        ("She flew to Berlin.", "She travelled to Germany.", "supported"),
        ("The report was filed Monday.", "A report was filed.", "supported"),
        ("Revenue grew 10%.", "Revenue increased.", "supported"),
        ("The cat sat on the mat.", "The dog swam in the sea.", "not_supported"),
        ("She flew to Berlin.", "She never left home.", "not_supported"),
        ("The report was filed Monday.", "No report exists.", "not_supported"),
        ("Revenue grew 10%.", "Revenue collapsed.", "not_supported"),
    ]
]

checks = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not cond:
        raise AssertionError(f"FAIL {label}: {detail}")
    print(f"  ok  {label}")


def dataset():
    p = Path(_TMP) / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS), encoding="utf-8")
    ds = data.parse(p, TaskType.SEQUENCE_CLASSIFICATION)
    train_ds, eval_ds = data.split(ds, 0.25, 42)
    return train_ds, eval_ds, data.merge_labels(train_ds, eval_ds)


def main() -> int:
    train_ds, eval_ds, labels = dataset()
    check("two labels inferred", labels == ["not_supported", "supported"], str(labels))

    common = dict(train_ds=train_ds, eval_ds=eval_ds, labels=labels,
                  on_progress=lambda p: None, on_checkpoint=lambda e: None,
                  should_cancel=lambda: False, should_stop=lambda: False,
                  should_yield=lambda: False)

    print("\ndefault (reinit_head=False) against a 3-class NLI head")
    cfg = JobConfig(base_model=NLI_BASE, epochs=1, batch_size=4, max_length=64,
                    eval_split=0.25, checkpoint_every_epochs=0)
    check("defaults to off", cfg.reinit_head is False)
    try:
        training.run(**common, cfg=cfg, out_dir=Path(_TMP) / "o1",
                     ckpt_dir=Path(_TMP) / "c1", log=lambda m: None)
    except RuntimeError as exc:
        check("fails loudly", "reinit_head" in str(exc), str(exc)[:200])
        check("names the base model", NLI_BASE in str(exc))
        print(f"     -> {str(exc)[:110]}...")
    else:
        raise AssertionError("expected a hard failure without reinit_head")

    print("\nreinit_head=True: encoder transfers, head rebuilt")
    logs: list[str] = []
    cfg2 = cfg.model_copy(update={"reinit_head": True})
    metrics = training.run(**common, cfg=cfg2, out_dir=Path(_TMP) / "o2",
                           ckpt_dir=Path(_TMP) / "c2",
                           log=lambda m: logs.append(m))

    reinit = [m for m in logs if "reinitialised due to shape mismatch" in m]
    check("logged the reinitialised tensors", len(reinit) == 1, str(logs[:6]))
    check("only the head was rebuilt",
          "classifier.weight" in reinit[0] and "classifier.bias" in reinit[0],
          reinit[0])
    check("encoder was NOT reinitialised",
          "encoder" not in reinit[0] and "embeddings" not in reinit[0], reinit[0])
    print(f"     -> {reinit[0]}")

    check("training completed", "accuracy" in metrics, str(metrics))
    check("model saved", (Path(_TMP) / "o2" / "config.json").exists())

    saved = json.loads((Path(_TMP) / "o2" / "config.json").read_text())
    # transformers derives num_labels from id2label and does not serialise it.
    check("head resized to 2", len(saved["id2label"]) == 2, str(saved["id2label"]))
    check("task labels replaced the NLI ones",
          sorted(saved["id2label"].values()) == ["not_supported", "supported"],
          str(saved["id2label"]))

    print("\nreinit_head=True is a no-op on a base with no head")
    logs3: list[str] = []
    cfg3 = JobConfig(base_model=PLAIN_BASE, epochs=1, batch_size=4, max_length=64,
                     eval_split=0.25, checkpoint_every_epochs=0, reinit_head=True)
    training.run(**common, cfg=cfg3, out_dir=Path(_TMP) / "o3",
                 ckpt_dir=Path(_TMP) / "c3", log=lambda m: logs3.append(m))
    check("nothing mismatched",
          not any("shape mismatch" in m for m in logs3), str(logs3[:6]))
    check("head still reported as newly initialised",
          any("newly initialised" in m for m in logs3), str(logs3[:6]))

    print("\nthe guard rejects mismatches beyond the head")
    try:
        training._check_load_report(
            {"missing_keys": set(),
             "mismatched_keys": {("classifier.weight", None, None),
                                 ("deberta.encoder.layer.0.attention.self.query_proj.weight",
                                  None, None)}},
            "some/incompatible-model", lambda m: None,
        )
    except ValueError as exc:
        check("refuses encoder mismatches", "query_proj" in str(exc), str(exc)[:160])
        check("explains the likely cause", "not a compatible base" in str(exc))
    else:
        raise AssertionError("guard should have rejected an encoder mismatch")

    print(f"\nALL {checks} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
