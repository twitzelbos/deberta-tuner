"""SQLite-backed job store.

A connection per operation, guarded by a lock. The worker and the API threads
both touch this and the volumes involved are tiny, so simplicity beats pooling.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from . import config
from .schemas import JobStatus

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    name            TEXT,
    status          TEXT NOT NULL,
    task            TEXT NOT NULL,
    base_model      TEXT NOT NULL,
    config          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    progress        REAL NOT NULL DEFAULT 0,
    error           TEXT,
    metrics         TEXT,
    labels          TEXT,
    num_train       INTEGER,
    num_eval        INTEGER,
    artifact_bytes  INTEGER,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init() -> None:
    config.ensure_dirs()
    with _lock, _connect() as conn:
        conn.executescript(_SCHEMA)
        # A job left RUNNING can only be a crash or restart mid-training; there
        # is no process to reattach to, so surface it as failed rather than
        # leaving it to hang in the UI forever.
        conn.execute(
            "UPDATE jobs SET status=?, error=?, finished_at=? WHERE status=?",
            (
                JobStatus.FAILED.value,
                "service restarted while this job was running",
                now(),
                JobStatus.RUNNING.value,
            ),
        )


def create(job_id: str, cfg: dict[str, Any]) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, name, status, task, base_model, config, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                job_id,
                cfg.get("name"),
                JobStatus.QUEUED.value,
                cfg["task"],
                cfg["base_model"],
                json.dumps(cfg),
                now(),
            ),
        )


def update(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    for key in ("metrics", "labels"):
        if key in fields and fields[key] is not None:
            fields[key] = json.dumps(fields[key])
    if isinstance(fields.get("status"), JobStatus):
        fields["status"] = fields["status"].value
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock, _connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["config"] = json.loads(d["config"])
    for key in ("metrics", "labels"):
        d[key] = json.loads(d[key]) if d[key] else None
    d["cancel_requested"] = bool(d["cancel_requested"])
    return d


def get(job_id: str) -> dict[str, Any] | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    sql = "SELECT * FROM jobs"
    args: list[Any] = []
    if status:
        sql += " WHERE status=?"
        args.append(status)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with _lock, _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_row_to_dict(r) for r in rows]


def claim_next() -> dict[str, Any] | None:
    """Atomically move the oldest queued job to RUNNING and return it."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status=? ORDER BY created_at LIMIT 1",
            (JobStatus.QUEUED.value,),
        ).fetchone()
        if row is None:
            return None
        cur = conn.execute(
            "UPDATE jobs SET status=?, started_at=? WHERE id=? AND status=?",
            (JobStatus.RUNNING.value, now(), row["id"], JobStatus.QUEUED.value),
        )
        if cur.rowcount != 1:
            return None
        job = _row_to_dict(row)
    job["status"] = JobStatus.RUNNING.value
    return job


def request_cancel(job_id: str) -> None:
    with _lock, _connect() as conn:
        conn.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))


def cancel_requested(job_id: str) -> bool:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
    return bool(row and row["cancel_requested"])


def delete(job_id: str) -> None:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
