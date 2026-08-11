"""Single-threaded job runner.

One GPU means one job at a time: the worker is deliberately serial.

GPU arbitration wraps the whole drain loop rather than each individual job. The
co-tenant service is stopped on the first claim and stays stopped while work
keeps arriving; it is only restarted once the queue has been empty for
GPU_IDLE_RESTART_SECONDS. Restarting between back-to-back jobs would be pure
waste -- the co-tenant takes minutes to load its model and the next job would
stop it again, usually mid-load. The restore is guarded by a finally so a crash
never leaves the co-tenant down.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import threading
import time
import traceback
from pathlib import Path

from . import config, data, db, training
from .schemas import PRIORITY_RANK, JobConfig, JobStatus, TaskType

_stop = threading.Event()
_thread: threading.Thread | None = None


def job_dir(job_id: str) -> Path:
    return config.JOBS_DIR / job_id


def checkpoint_dir(job_id: str) -> Path:
    return job_dir(job_id) / "checkpoint"


def purge_checkpoint(job_id: str) -> None:
    """Checkpoints are large (~3x model size); drop them once unneeded."""
    shutil.rmtree(checkpoint_dir(job_id), ignore_errors=True)


def recover_interrupted() -> None:
    """Reconcile jobs left RUNNING by a crash or restart.

    Called once at startup, before the worker begins. A job with a usable
    checkpoint goes back on the queue and resumes; one without has nothing to
    resume from and is failed.
    """
    for job in db.interrupted_jobs():
        job_id = job["id"]
        has_ckpt = (checkpoint_dir(job_id) / training.CHECKPOINT_FILE).exists()
        if job.get("cancel_requested"):
            db.update(job_id, status=JobStatus.CANCELLED, finished_at=db.now())
            purge_checkpoint(job_id)
        elif has_ckpt:
            db.update(job_id, status=JobStatus.PAUSED, started_at=None, error=None)
        else:
            db.update(
                job_id,
                status=JobStatus.FAILED,
                finished_at=db.now(),
                error="service restarted while this job was running, and no "
                      "checkpoint was available to resume from",
            )


# --------------------------------------------------------------------------- #
# SGLang coordination
# --------------------------------------------------------------------------- #


def _unit_exists() -> bool:
    r = subprocess.run(
        ["systemctl", "list-unit-files", config.SGLANG_UNIT],
        capture_output=True, text=True,
    )
    return config.SGLANG_UNIT in r.stdout


def _svc_log(msg: str) -> None:
    """Log a GPU arbitration event to the journal.

    These events span jobs now, so they cannot go in any single job's log.
    """
    print(f"{db.now()} [gpu] {msg}", flush=True)


def _systemctl(action: str, log) -> bool:
    cmd = ["sudo", "-n", "systemctl", action, config.SGLANG_UNIT]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"WARNING: `{' '.join(cmd)}` failed rc={r.returncode}: {r.stderr.strip()}")
        return False
    log(f"systemctl {action} {config.SGLANG_UNIT}: ok")
    return True


def _stop_cotenant(log) -> bool:
    """Stop the co-tenant service so the GPU is free.

    Returns True if it was actually stopped, meaning we now owe it a restart.
    """
    if not config.SGLANG_CONTROL:
        log("SGLang control disabled; assuming the GPU is free")
        return False
    if not _unit_exists():
        log(f"{config.SGLANG_UNIT} not installed; nothing to stop")
        return False
    was_active = subprocess.run(
        ["systemctl", "is-active", "--quiet", config.SGLANG_UNIT]
    ).returncode == 0
    if not was_active:
        log(f"{config.SGLANG_UNIT} already inactive")
        return False
    _systemctl("stop", log)
    # systemctl returns once the process is gone, but the driver can take a
    # moment to reclaim the allocation.
    for _ in range(30):
        time.sleep(1)
        if gpu_free_mib(log) > 20_000:
            break
    return True


def _start_cotenant(log) -> None:
    _systemctl("start", log)


def gpu_free_mib(log) -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        return int(r.stdout.strip().splitlines()[0])
    except Exception as exc:  # nvidia-smi missing, or the driver is unhappy
        log(f"could not read GPU memory: {exc}")
        return 0


# --------------------------------------------------------------------------- #
# Job execution
# --------------------------------------------------------------------------- #


def _run_job(job: dict) -> None:
    job_id = job["id"]
    jdir = job_dir(job_id)
    log_path = jdir / "train.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)

    def log(msg: str) -> None:
        log_file.write(f"{db.now()} {msg}\n")

    try:
        cfg = JobConfig(**job["config"])
        log(f"job {job_id} starting")

        train_raw = data.parse(jdir / "train.jsonl", cfg.task)
        eval_path = jdir / "eval.jsonl"
        if eval_path.exists():
            eval_raw = data.parse(eval_path, cfg.task)
            train_ds, eval_ds = train_raw, eval_raw
        else:
            train_ds, eval_ds = data.split(train_raw, cfg.eval_split, cfg.seed)

        labels = data.merge_labels(train_ds, eval_ds)
        db.update(
            job_id,
            labels=labels,
            num_train=len(train_ds),
            num_eval=len(eval_ds) if eval_ds else 0,
        )

        # The co-tenant was already stopped by the drain loop, which holds the
        # GPU across consecutive jobs. Record what we actually got.
        log(f"GPU free: {gpu_free_mib(log)} MiB")

        metrics = training.run(
            train_ds=train_ds,
            eval_ds=eval_ds,
            labels=labels,
            cfg=cfg,
            out_dir=jdir / "output",
            ckpt_dir=checkpoint_dir(job_id),
            log=log,
            on_progress=lambda p: db.update(job_id, progress=round(p, 4)),
            on_checkpoint=lambda e: db.update(job_id, epochs_completed=e),
            should_cancel=lambda: db.cancel_requested(job_id),
            should_stop=_stop.is_set,
            should_yield=lambda: db.higher_priority_waiting(
                PRIORITY_RANK[job["priority"]]
            ),
        )

        artifact = jdir / "model.tar.gz"
        with tarfile.open(artifact, "w:gz") as tar:
            tar.add(jdir / "output", arcname=job_id)
        log(f"artifact {artifact.name} ({artifact.stat().st_size} bytes)")

        db.update(
            job_id,
            status=JobStatus.SUCCEEDED,
            finished_at=db.now(),
            progress=1.0,
            metrics=metrics,
            artifact_bytes=artifact.stat().st_size,
        )
        purge_checkpoint(job_id)
        log("job succeeded")

    except training.Preempted:
        # Higher-priority work is waiting. Not a failure: keep the checkpoint,
        # keep the recorded progress, and go back to the queue. Ordering by
        # priority means this job only resumes once nothing better is waiting.
        log("preempted by higher-priority work; paused at its last checkpoint")
        db.update(
            job_id,
            status=JobStatus.PAUSED,
            started_at=None,
            preempted_count=(job.get("preempted_count") or 0) + 1,
        )
    except training.Interrupted:
        # Graceful shutdown, not a failure. Leave the checkpoint in place and
        # pause so the job resumes after the restart.
        log("service shutting down; job paused to resume from its checkpoint")
        db.update(job_id, status=JobStatus.PAUSED, started_at=None)
    except training.Cancelled:
        log("job cancelled")
        db.update(job_id, status=JobStatus.CANCELLED, finished_at=db.now())
        purge_checkpoint(job_id)
    except Exception as exc:
        log("job failed:\n" + traceback.format_exc())
        db.update(
            job_id,
            status=JobStatus.FAILED,
            finished_at=db.now(),
            error=f"{type(exc).__name__}: {exc}"[:2000],
        )
        purge_checkpoint(job_id)
    finally:
        # The GPU is handed back by the drain loop, not here: another job may
        # be waiting and would only have to stop the co-tenant again.
        log_file.close()


def _loop() -> None:
    """Drain the queue, holding the GPU across consecutive jobs."""
    holding = False       # true once we have stopped the co-tenant
    idle_since = None     # monotonic timestamp of the first empty poll

    try:
        while not _stop.is_set():
            job = db.claim_next()

            if job is not None:
                idle_since = None
                if not holding:
                    holding = _stop_cotenant(_svc_log)
                _run_job(job)
                continue

            # Nothing runnable. claim_next() covers both queued and paused
            # jobs, so an empty result really does mean there is no work --
            # including no job paused waiting on higher-priority work.
            if holding:
                if idle_since is None:
                    idle_since = time.monotonic()
                    _svc_log(
                        "queue empty; restarting the co-tenant in "
                        f"{config.GPU_IDLE_RESTART_SECONDS}s unless work arrives"
                    )
                elif time.monotonic() - idle_since >= config.GPU_IDLE_RESTART_SECONDS:
                    _start_cotenant(_svc_log)
                    holding = False
                    idle_since = None

            _stop.wait(2.0)
    finally:
        # Shutting down, or the loop died. Either way, give the GPU back.
        if holding:
            try:
                _start_cotenant(_svc_log)
            except Exception:
                _svc_log("failed to restart the co-tenant:\n" + traceback.format_exc())


def start() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="tuner-worker", daemon=True)
    _thread.start()


def shutdown(timeout: float = 120.0) -> None:
    """Signal the worker to stop and wait for an in-flight job to unwind.

    A running job raises Interrupted on its next step, re-queues itself and,
    crucially, still restarts the co-tenant GPU service in its finally block.
    The unit's TimeoutStopSec must exceed this timeout for that to complete.
    """
    _stop.set()
    if _thread:
        _thread.join(timeout=timeout)
        if _thread.is_alive():
            print("worker did not stop within timeout; abandoning it", flush=True)


def purge(job_id: str) -> None:
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
