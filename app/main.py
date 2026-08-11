"""HTTP API for the DeBERTa tuning service."""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import ValidationError

from . import config, data, db, worker
from .schemas import (
    PRIORITY_RANK,
    JobConfig,
    JobDetail,
    JobStatus,
    JobSummary,
    QueuedJob,
    QueueView,
    RunningJob,
)

CHUNK = 1 << 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.init()
    # Reconcile anything orphaned by the previous shutdown before accepting work.
    worker.recover_interrupted()
    worker.start()
    yield
    worker.shutdown()


app = FastAPI(
    title="DeBERTa tuning service",
    version="1.1.0",
    description="Submit a JSONL dataset, get back a fine-tuned DeBERTa model.",
    lifespan=lifespan,
)


def _age_seconds(stamp: str | None) -> int:
    """Whole seconds between an ISO-8601 timestamp and now."""
    if not stamp:
        return 0
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return 0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - then).total_seconds()))


def _save_upload(upload: UploadFile, dest: Path) -> int:
    """Stream an upload to disk, enforcing the size cap as we go."""
    total = 0
    with dest.open("wb") as fh:
        while chunk := upload.file.read(CHUNK):
            total += len(chunk)
            if total > config.MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"upload exceeds {config.MAX_UPLOAD_MB} MB limit"
                )
            fh.write(chunk)
    if total == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"{dest.name} is empty")
    return total


def _detail(job_id: str) -> JobDetail:
    row = db.get(job_id)
    if row is None:
        raise HTTPException(404, f"no such job: {job_id}")
    return JobDetail(**row, queue_position=db.queue_position(job_id))


@app.get("/healthz")
def healthz() -> dict:
    import torch

    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_free_mib": worker.gpu_free_mib(lambda _m: None),
        "queued": len(db.list_jobs(status=JobStatus.QUEUED.value, limit=1000)),
        "paused": len(db.list_jobs(status=JobStatus.PAUSED.value, limit=1000)),
        "running": len(db.list_jobs(status=JobStatus.RUNNING.value, limit=10)),
    }


@app.get("/v1/base-models")
def base_models() -> dict:
    return {"base_models": sorted(config.ALLOWED_BASE_MODELS)}


@app.get("/v1/queue", response_model=QueueView)
def get_queue() -> QueueView:
    """What is running, what is waiting, in what order, and why."""
    run = db.running_job()
    running = None
    if run is not None:
        elapsed = _age_seconds(run["started_at"])
        progress = run["progress"] or 0.0
        # Straight-line extrapolation. Deliberately crude: it ignores evaluation
        # and model saving, so treat it as a floor rather than a promise.
        eta = int(elapsed * (1 - progress) / progress) if progress > 0.01 else None
        running = RunningJob(
            id=run["id"],
            name=run["name"],
            task=run["task"],
            base_model=run["base_model"],
            priority=run["priority"],
            started_at=run["started_at"],
            progress=progress,
            epochs_completed=run["epochs_completed"],
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            yielding=db.higher_priority_waiting(PRIORITY_RANK[run["priority"]]),
        )

    waiting = [
        QueuedJob(
            position=i,
            id=j["id"],
            name=j["name"],
            status=j["status"],
            priority=j["priority"],
            task=j["task"],
            base_model=j["base_model"],
            created_at=j["created_at"],
            waiting_seconds=_age_seconds(j["created_at"]),
            # Non-zero for a paused job: where it stopped and will resume from.
            progress=j["progress"] or 0.0,
            epochs_completed=j["epochs_completed"],
            preempted_count=j["preempted_count"],
        )
        for i, j in enumerate(db.waiting_jobs(), start=1)
    ]

    return QueueView(
        running=running,
        waiting=waiting,
        queued_count=sum(1 for j in waiting if j.status is JobStatus.QUEUED),
        paused_count=sum(1 for j in waiting if j.status is JobStatus.PAUSED),
    )


@app.post("/v1/jobs", response_model=JobDetail, status_code=201)
def create_job(
    config_json: str = Form(
        ..., alias="config", description="JobConfig as a JSON object"
    ),
    train_file: UploadFile = File(..., description="Training data, JSONL"),
    eval_file: UploadFile | None = File(None, description="Optional eval data, JSONL"),
) -> JobDetail:
    try:
        cfg = JobConfig(**json.loads(config_json))
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"config is not valid JSON: {exc.msg}") from None
    except ValidationError as exc:
        raise HTTPException(400, json.loads(exc.json())) from None

    job_id = uuid.uuid4().hex[:16]
    jdir = worker.job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=False)

    try:
        _save_upload(train_file, jdir / "train.jsonl")
        if eval_file is not None and eval_file.filename:
            _save_upload(eval_file, jdir / "eval.jsonl")

        # Validate now so the caller gets a 400 immediately, rather than a job
        # that sits in the queue and then fails minutes later.
        data.parse(jdir / "train.jsonl", cfg.task)
        if (jdir / "eval.jsonl").exists():
            data.parse(jdir / "eval.jsonl", cfg.task)
    except HTTPException:
        worker.purge(job_id)
        raise
    except data.DataError as exc:
        worker.purge(job_id)
        raise HTTPException(400, f"invalid training data: {exc}") from None
    except Exception:
        worker.purge(job_id)
        raise

    payload = cfg.model_dump(mode="json")
    (jdir / "config.json").write_text(json.dumps(payload, indent=2))
    db.create(job_id, payload)
    return _detail(job_id)


@app.get("/v1/jobs", response_model=list[JobSummary])
def list_jobs(
    status: JobStatus | None = None, limit: int = Query(100, ge=1, le=1000)
) -> list[JobSummary]:
    rows = db.list_jobs(status=status.value if status else None, limit=limit)
    # Resolve queue positions in one pass rather than a query per row.
    order = {j["id"]: i for i, j in enumerate(db.waiting_jobs(), start=1)}
    return [JobSummary(**r, queue_position=order.get(r["id"])) for r in rows]


@app.get("/v1/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str) -> JobDetail:
    return _detail(job_id)


@app.get("/v1/jobs/{job_id}/logs", response_class=PlainTextResponse)
def get_logs(job_id: str, tail: int = Query(0, ge=0)) -> str:
    _detail(job_id)
    path = worker.job_dir(job_id) / "train.log"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if tail:
        return "\n".join(text.splitlines()[-tail:])
    return text


@app.post("/v1/jobs/{job_id}/cancel", response_model=JobDetail)
def cancel_job(job_id: str) -> JobDetail:
    row = db.get(job_id)
    if row is None:
        raise HTTPException(404, f"no such job: {job_id}")
    status = JobStatus(row["status"])
    if status.terminal:
        raise HTTPException(409, f"job is already {status.value}")
    if status.runnable:
        # Not executing, so no loop is watching the cancel flag. Drop any
        # checkpoint too, or a paused job would leave gigabytes behind.
        db.update(job_id, status=JobStatus.CANCELLED, finished_at=db.now())
        worker.purge_checkpoint(job_id)
    else:
        db.request_cancel(job_id)
    return _detail(job_id)


@app.get("/v1/jobs/{job_id}/artifact")
def download_artifact(job_id: str) -> FileResponse:
    row = db.get(job_id)
    if row is None:
        raise HTTPException(404, f"no such job: {job_id}")
    if row["status"] != JobStatus.SUCCEEDED.value:
        raise HTTPException(409, f"job is {row['status']}, no artifact available")
    path = worker.job_dir(job_id) / "model.tar.gz"
    if not path.exists():
        raise HTTPException(410, "artifact has been deleted")
    return FileResponse(
        path, media_type="application/gzip", filename=f"{job_id}.tar.gz"
    )


@app.delete("/v1/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    row = db.get(job_id)
    if row is None:
        raise HTTPException(404, f"no such job: {job_id}")
    if row["status"] == JobStatus.RUNNING.value:
        raise HTTPException(409, "cancel the job before deleting it")
    worker.purge(job_id)
    db.delete(job_id)
