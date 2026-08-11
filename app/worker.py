"""Single-threaded job runner.

One GPU means one job at a time: the worker is deliberately serial. Before
training it stops the SGLang unit to free ~21.6 GiB of VRAM, and restarts it
afterwards -- including on failure, so a crashed job never leaves the inference
endpoint down.
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
from .schemas import JobConfig, JobStatus, TaskType

_stop = threading.Event()
_thread: threading.Thread | None = None


def job_dir(job_id: str) -> Path:
    return config.JOBS_DIR / job_id


# --------------------------------------------------------------------------- #
# SGLang coordination
# --------------------------------------------------------------------------- #


def _unit_exists() -> bool:
    r = subprocess.run(
        ["systemctl", "list-unit-files", config.SGLANG_UNIT],
        capture_output=True, text=True,
    )
    return config.SGLANG_UNIT in r.stdout


def _systemctl(action: str, log) -> bool:
    cmd = ["sudo", "-n", "systemctl", action, config.SGLANG_UNIT]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"WARNING: `{' '.join(cmd)}` failed rc={r.returncode}: {r.stderr.strip()}")
        return False
    log(f"systemctl {action} {config.SGLANG_UNIT}: ok")
    return True


def _release_gpu(log) -> bool:
    """Stop SGLang so the GPU is free. Returns True if we must restart it."""
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
        if _gpu_free_mib(log) > 20_000:
            break
    return True


def _restore_gpu(log) -> None:
    _systemctl("start", log)


def _gpu_free_mib(log) -> int:
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

    restart_needed = False
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

        restart_needed = _release_gpu(log)
        free = _gpu_free_mib(log)
        log(f"GPU free: {free} MiB")

        metrics = training.run(
            train_ds=train_ds,
            eval_ds=eval_ds,
            labels=labels,
            cfg=cfg,
            out_dir=jdir / "output",
            log=log,
            on_progress=lambda p: db.update(job_id, progress=round(p, 4)),
            should_cancel=lambda: db.cancel_requested(job_id),
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
        log("job succeeded")

    except training.Cancelled:
        log("job cancelled")
        db.update(job_id, status=JobStatus.CANCELLED, finished_at=db.now())
    except Exception as exc:
        log("job failed:\n" + traceback.format_exc())
        db.update(
            job_id,
            status=JobStatus.FAILED,
            finished_at=db.now(),
            error=f"{type(exc).__name__}: {exc}"[:2000],
        )
    finally:
        # Always hand the GPU back, even if training blew up.
        if restart_needed:
            try:
                _restore_gpu(log)
            except Exception:
                log("failed to restart SGLang:\n" + traceback.format_exc())
        log_file.close()


def _loop() -> None:
    while not _stop.is_set():
        job = db.claim_next()
        if job is None:
            _stop.wait(2.0)
            continue
        _run_job(job)


def start() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="tuner-worker", daemon=True)
    _thread.start()


def shutdown() -> None:
    _stop.set()
    if _thread:
        _thread.join(timeout=5)


def purge(job_id: str) -> None:
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
