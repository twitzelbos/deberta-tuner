"""Checkpointing, resume and preemption against a real training run.

    python -m tests.test_resume

Uses deberta-v3-xsmall on CPU so it finishes in a couple of minutes.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["TUNER_FORCE_CPU"] = "1"
os.environ["TUNER_SGLANG_CONTROL"] = "0"
_TMP = tempfile.mkdtemp(prefix="tuner-resume-")
os.environ["TUNER_STATE_DIR"] = _TMP

from app import data, training  # noqa: E402
from app.schemas import JobConfig, TaskType  # noqa: E402

BASE = "microsoft/deberta-v3-xsmall"
ROWS = [
    {"text": t, "label": lab}
    for t, lab in [
        ("great value", "positive"), ("loved it", "positive"),
        ("works perfectly", "positive"), ("excellent build", "positive"),
        ("broke instantly", "negative"), ("terrible quality", "negative"),
        ("waste of money", "negative"), ("returned it", "negative"),
    ]
]

checks = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not cond:
        raise AssertionError(f"FAIL {label}: {detail}")
    print(f"  ok  {label}")


def make_run(tmp: Path):
    jsonl = tmp / "train.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in ROWS), encoding="utf-8")
    ds = data.parse(jsonl, TaskType.SEQUENCE_CLASSIFICATION)
    train_ds, eval_ds = data.split(ds, 0.25, 42)
    labels = data.merge_labels(train_ds, eval_ds)
    cfg = JobConfig(base_model=BASE, task=TaskType.SEQUENCE_CLASSIFICATION,
                    epochs=4, batch_size=4, max_length=32,
                    checkpoint_every_epochs=1)
    return train_ds, eval_ds, labels, cfg


def main() -> int:
    tmp = Path(_TMP)
    ckpt_dir = tmp / "checkpoint"
    train_ds, eval_ds, labels, cfg = make_run(tmp)

    common = dict(train_ds=train_ds, eval_ds=eval_ds, labels=labels, cfg=cfg,
                  out_dir=tmp / "output", ckpt_dir=ckpt_dir)

    # ---------------------------------------------------------------- #
    print("interrupt after the first checkpoint")
    logs: list[str] = []
    state = {"stop": False}

    try:
        training.run(
            **common,
            log=lambda m: (logs.append(m), print("   " + m)) and None,
            on_progress=lambda p: None,
            # Trip the shutdown flag as soon as epoch 1 is checkpointed.
            on_checkpoint=lambda e: state.__setitem__("stop", True),
            should_cancel=lambda: False,
            should_stop=lambda: state["stop"],
            should_yield=lambda: False,
        )
    except training.Interrupted:
        pass
    else:
        raise AssertionError("expected Interrupted")

    ckpt_file = ckpt_dir / training.CHECKPOINT_FILE
    check("checkpoint written", ckpt_file.exists())
    check("checkpoint is non-trivial", ckpt_file.stat().st_size > 10_000,
          str(ckpt_file.stat().st_size))
    check("logged the save", any("checkpoint saved at epoch 1" in m for m in logs))
    check("no model saved yet", not (tmp / "output" / "config.json").exists())

    import torch
    blob = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    check("records epochs_completed", blob["epochs_completed"] == 1, str(blob["epochs_completed"]))
    check("records step", blob["step"] > 0)
    check("records labels", blob["labels"] == labels, str(blob["labels"]))
    check("has optimizer state", "optimizer" in blob and blob["optimizer"]["state"])
    check("has scheduler state", "scheduler" in blob)

    # ---------------------------------------------------------------- #
    print("\nresume from the checkpoint")
    logs2: list[str] = []
    metrics = training.run(
        **common,
        log=lambda m: (logs2.append(m), print("   " + m)) and None,
        on_progress=lambda p: None,
        on_checkpoint=lambda e: None,
        should_cancel=lambda: False,
        should_stop=lambda: False,
        should_yield=lambda: False,
    )
    check("logged the resume", any("resumed from checkpoint" in m for m in logs2),
          "\n".join(logs2[:5]))
    check("skipped the finished epoch",
          any("1 epochs done" in m for m in logs2), "\n".join(logs2[:5]))
    check("did not redo epoch 1", not any("epoch 1 complete" in m for m in logs2),
          "resume must continue, not restart")
    check("ran to completion", any("epoch 4 complete" in m for m in logs2))
    check("returned metrics", "accuracy" in metrics, str(metrics))
    check("saved the model", (tmp / "output" / "config.json").exists())

    # ---------------------------------------------------------------- #
    print("\npreemption yields at an epoch boundary")
    ck2 = tmp / "checkpoint2"
    logs3: list[str] = []
    try:
        training.run(
            **{**common, "ckpt_dir": ck2, "out_dir": tmp / "output2"},
            log=lambda m: (logs3.append(m), print("   " + m)) and None,
            on_progress=lambda p: None,
            on_checkpoint=lambda e: None,
            should_cancel=lambda: False,
            should_stop=lambda: False,
            should_yield=lambda: True,          # higher-priority work waiting
        )
    except training.Preempted:
        pass
    else:
        raise AssertionError("expected Preempted")

    check("yielded after a whole epoch",
          any("yielding after epoch 1" in m for m in logs3), "\n".join(logs3[-3:]))
    check("checkpointed before yielding", (ck2 / training.CHECKPOINT_FILE).exists(),
          "preempting without a checkpoint would lose the epoch")
    check("did not save a model", not (tmp / "output2" / "config.json").exists())

    # ---------------------------------------------------------------- #
    print("\ncheckpointing disabled opts out of preemption")
    ck3 = tmp / "checkpoint3"
    cfg_off = cfg.model_copy(update={"checkpoint_every_epochs": 0, "epochs": 1})
    logs4: list[str] = []
    training.run(
        **{**common, "cfg": cfg_off, "ckpt_dir": ck3, "out_dir": tmp / "output3"},
        log=lambda m: logs4.append(m),
        on_progress=lambda p: None,
        on_checkpoint=lambda e: None,
        should_cancel=lambda: False,
        should_stop=lambda: False,
        should_yield=lambda: True,          # ignored: nothing to resume from
    )
    check("ran to completion despite yield request",
          (tmp / "output3" / "config.json").exists())
    check("wrote no checkpoint", not (ck3 / training.CHECKPOINT_FILE).exists())

    print(f"\nALL {checks} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
