"""Priority queue ordering, preemption bookkeeping and crash recovery.

    python -m tests.test_queue

Deterministic: exercises the job store and the /v1/queue view directly, so no
training happens and there is no race with the worker.
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ["TUNER_FORCE_CPU"] = "1"
os.environ["TUNER_SGLANG_CONTROL"] = "0"
os.environ["TUNER_STATE_DIR"] = tempfile.mkdtemp(prefix="tuner-queue-")

from app import db, training, worker  # noqa: E402
from app.schemas import JobStatus  # noqa: E402

checks = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not cond:
        raise AssertionError(f"FAIL {label}: {detail}")
    print(f"  ok  {label}")


def cfg(priority: str = "normal") -> dict:
    return {
        "task": "sequence_classification",
        "base_model": "microsoft/deberta-v3-xsmall",
        "priority": priority,
    }


def ids() -> list[str]:
    return [j["id"] for j in db.waiting_jobs()]


def main() -> int:
    db.init()

    print("priority ordering")
    # Created oldest-first; priority must override arrival order.
    for jid, prio in [("n1", "normal"), ("l1", "low"), ("h1", "high"), ("n2", "normal")]:
        db.create(jid, cfg(prio))

    check("high jumps the queue", ids() == ["h1", "n1", "n2", "l1"], str(ids()))
    check("fifo within a priority", db.queue_position("n1") < db.queue_position("n2"))
    check("low is last", db.queue_position("l1") == 4)
    check("unknown job has no position", db.queue_position("nope") is None)

    print("\npreemption predicate")
    check("something outranks low", db.higher_priority_waiting(0) is True)
    check("something outranks normal", db.higher_priority_waiting(1) is True)
    check("nothing outranks high", db.higher_priority_waiting(2) is False,
          "equal priority must not preempt, or two jobs trade the GPU forever")

    print("\nclaiming respects priority")
    check("claims the high one", db.claim_next()["id"] == "h1")
    check("running_job reports it", db.running_job()["id"] == "h1")
    check("claimed job leaves the queue", db.queue_position("h1") is None)
    check("normal now heads the queue", ids() == ["n1", "n2", "l1"], str(ids()))
    check("high no longer waiting", db.higher_priority_waiting(1) is False)

    print("\npaused jobs compete on equal terms")
    db.update("n1", status=JobStatus.PAUSED, progress=0.4, epochs_completed=2,
              preempted_count=1)
    check("paused keeps its slot", ids() == ["n1", "n2", "l1"], str(ids()))
    check("paused retains progress", db.get("n1")["progress"] == 0.4)
    check("paused retains epochs", db.get("n1")["epochs_completed"] == 2)
    check("paused is claimable", db.claim_next()["id"] == "n1")

    # A paused low-priority job must not outrank a queued high-priority one.
    db.update("n1", status=JobStatus.PAUSED)
    db.create("h2", cfg("high"))
    check("fresh high beats paused normal", ids()[0] == "h2", str(ids()))

    print("\ncrash recovery")
    db.update("h1", status=JobStatus.RUNNING)
    db.update("h2", status=JobStatus.RUNNING)
    # Give h2 a checkpoint; h1 has none.
    ck = worker.checkpoint_dir("h2")
    ck.mkdir(parents=True, exist_ok=True)
    (ck / training.CHECKPOINT_FILE).write_bytes(b"not a real checkpoint")

    worker.recover_interrupted()
    check("no checkpoint -> failed", db.get("h1")["status"] == JobStatus.FAILED.value,
          db.get("h1")["status"])
    check("failure explains why", "no checkpoint" in (db.get("h1")["error"] or ""))
    check("checkpoint -> paused", db.get("h2")["status"] == JobStatus.PAUSED.value,
          db.get("h2")["status"])
    check("resumable job is waiting again", "h2" in ids(), str(ids()))

    print("\ncancel drops the checkpoint")
    db.update("h2", status=JobStatus.RUNNING, cancel_requested=1)
    worker.recover_interrupted()
    check("cancel wins over resume", db.get("h2")["status"] == JobStatus.CANCELLED.value)
    check("checkpoint removed", not (ck / training.CHECKPOINT_FILE).exists())

    print("\n/v1/queue view")
    # Route functions are called directly rather than through TestClient: the
    # app's lifespan starts the worker thread, which would immediately claim the
    # paused job and race every assertion below. HTTP serialisation of these
    # same routes is covered by tests/test_api.py.
    from app.main import cancel_job, get_queue, healthz

    view = get_queue()
    check("positions are 1..n",
          [w.position for w in view.waiting] == list(range(1, len(view.waiting) + 1)),
          str([w.position for w in view.waiting]))
    check("nothing running", view.running is None)
    check("concurrency is 1", view.concurrency == 1)

    paused = [w for w in view.waiting if w.status is JobStatus.PAUSED]
    check("paused job is listed", len(paused) == 1,
          str([(w.id, w.status) for w in view.waiting]))
    check("paused shows where it stopped", paused[0].progress == 0.4, str(paused[0]))
    check("paused shows epochs done", paused[0].epochs_completed == 2)
    check("paused shows preempted_count", paused[0].preempted_count == 1)
    check("paused sorts ahead of same-priority queued",
          view.waiting[0].id == "n1", str([w.id for w in view.waiting]))
    check("counts split by status",
          view.paused_count == 1 and view.queued_count == len(view.waiting) - 1,
          f"{view.queued_count}/{view.paused_count}")

    check("healthz reports paused", healthz()["paused"] == 1)

    # Cancelling a paused job must be immediate, not deferred to a worker.
    cancelled = cancel_job("n1")
    check("cancel paused is immediate", cancelled.status is JobStatus.CANCELLED,
          str(cancelled.status))
    check("cancelled job leaves the queue", "n1" not in ids(), str(ids()))

    print(f"\nALL {checks} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
