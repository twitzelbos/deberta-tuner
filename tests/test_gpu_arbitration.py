"""GPU arbitration: hold across consecutive jobs, restart after an idle delay.

    python -m tests.test_gpu_arbitration

Training is stubbed out and the systemctl calls are replaced with recorders, so
this runs in seconds and needs no GPU. It tests the scheduling decisions, not
the training.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

os.environ["TUNER_FORCE_CPU"] = "1"
os.environ["TUNER_SGLANG_CONTROL"] = "1"
os.environ["TUNER_STATE_DIR"] = tempfile.mkdtemp(prefix="tuner-gpu-")
# Keep the test quick; production default is 60s.
os.environ["TUNER_GPU_IDLE_RESTART_SECONDS"] = "4"

from app import config, db, worker  # noqa: E402
from app.schemas import JobStatus  # noqa: E402

IDLE = config.GPU_IDLE_RESTART_SECONDS
events: list[str] = []
checks = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not cond:
        raise AssertionError(f"FAIL {label}: {detail}")
    print(f"  ok  {label}")


def install_stubs() -> None:
    """Replace the co-tenant control and the trainer with recorders."""

    def stop(_log) -> bool:
        events.append("stop")
        return True

    def start(_log) -> None:
        events.append("start")

    def fake_train(**kw):
        events.append("train")
        out = kw["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}")
        return {"train_steps": 1, "accuracy": 1.0}

    worker._stop_cotenant = stop
    worker._start_cotenant = start
    worker.gpu_free_mib = lambda _log: 23_000
    worker.training.run = fake_train


def submit(job_id: str, priority: str = "normal") -> None:
    jdir = worker.job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "train.jsonl").write_text(
        "\n".join(
            json.dumps({"text": t, "label": lab})
            for t, lab in [("good", "pos"), ("bad", "neg"), ("great", "pos"),
                           ("awful", "neg")]
        )
    )
    db.create(job_id, {
        "task": "sequence_classification",
        "base_model": "microsoft/deberta-v3-xsmall",
        "priority": priority,
        "epochs": 1,
        "checkpoint_every_epochs": 0,
    })


def wait_until(pred, timeout: float, what: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}; events={events}")


def statuses() -> dict[str, str]:
    return {j["id"]: j["status"] for j in db.list_jobs(limit=50)}


def main() -> int:
    db.init()
    install_stubs()

    print(f"idle restart delay = {IDLE}s")
    print("\nthree back-to-back jobs share one GPU acquisition")
    for n in ("a", "b", "c"):
        submit(n)

    worker.start()
    try:
        wait_until(
            lambda: all(s == JobStatus.SUCCEEDED.value for s in statuses().values()),
            60, "all three jobs to finish",
        )
        check("all three ran", events.count("train") == 3, str(events))
        check("co-tenant stopped exactly once", events.count("stop") == 1, str(events))
        check("not restarted between jobs", events.count("start") == 0, str(events))
        check("stop came first", events[0] == "stop", str(events))

        print("\nrestart only after the queue stays empty")
        # Well inside the debounce window: still held.
        time.sleep(IDLE / 2)
        check("still held mid-debounce", events.count("start") == 0, str(events))

        # A new job inside the window must cancel the pending restart.
        submit("d")
        wait_until(lambda: statuses().get("d") == JobStatus.SUCCEEDED.value,
                   60, "job d")
        check("no restart happened", events.count("start") == 0, str(events))
        check("no second stop needed", events.count("stop") == 1, str(events))
        check("job d reused the held GPU", events.count("train") == 4, str(events))

        print("\nafter the window elapses")
        wait_until(lambda: events.count("start") == 1, IDLE + 20, "the restart")
        check("co-tenant restarted once", events.count("start") == 1, str(events))
        check("restart came last", events[-1] == "start", str(events))

        print("\na later job re-acquires")
        submit("e")
        wait_until(lambda: statuses().get("e") == JobStatus.SUCCEEDED.value, 60, "job e")
        check("stopped again for new work", events.count("stop") == 2, str(events))
        check("ordering is stop,start,...,stop",
              events[-2:] == ["stop", "train"], str(events[-4:]))
    finally:
        worker.shutdown(timeout=30)

    print("\nshutdown hands the GPU back")
    check("restarted on shutdown", events.count("start") == 2, str(events))
    check("balanced stop/start", events.count("stop") == events.count("start"),
          str(events))

    print(f"\nALL {checks} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
