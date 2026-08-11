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
from .schemas import PRIORITY_RANK, RANK_PRIORITY, JobStatus

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
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    epochs_completed INTEGER NOT NULL DEFAULT 0,
    priority        INTEGER NOT NULL DEFAULT 1,
    preempted_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, priority, created_at, id);
"""

# Columns added after the first release. Applied on startup to existing files.
_MIGRATIONS = {
    "epochs_completed": "ALTER TABLE jobs ADD COLUMN epochs_completed INTEGER NOT NULL DEFAULT 0",
    "priority": "ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 1",
    "preempted_count": "ALTER TABLE jobs ADD COLUMN preempted_count INTEGER NOT NULL DEFAULT 0",
}

# Statuses eligible for the worker to claim, newest schedule first.
_RUNNABLE = (JobStatus.QUEUED.value, JobStatus.PAUSED.value)
# priority DESC then age ASC: higher priority always wins; ties go to whoever
# has been waiting longest, with id as a final deterministic tiebreak.
_QUEUE_ORDER = "ORDER BY priority DESC, created_at, id"


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
        have = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        for column, ddl in _MIGRATIONS.items():
            if column not in have:
                conn.execute(ddl)
    # Jobs left RUNNING are handled by worker.recover_interrupted(), which can
    # see whether a checkpoint exists and therefore whether they can resume.


def create(job_id: str, cfg: dict[str, Any]) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, name, status, task, base_model, config,"
            " created_at, priority) VALUES (?,?,?,?,?,?,?,?)",
            (
                job_id,
                cfg.get("name"),
                JobStatus.QUEUED.value,
                cfg["task"],
                cfg["base_model"],
                json.dumps(cfg),
                now(),
                PRIORITY_RANK[cfg.get("priority", "normal")],
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
    # Stored as an int for ordering; exposed as the API's string form.
    d["priority"] = RANK_PRIORITY.get(d.get("priority", 1), "normal")
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
    """Atomically move the highest-priority waiting job to RUNNING.

    Paused jobs compete on equal terms with never-started ones, so a paused
    high-priority job outranks a fresh normal-priority one.
    """
    with _lock, _connect() as conn:
        row = conn.execute(
            f"SELECT * FROM jobs WHERE status IN (?,?) {_QUEUE_ORDER} LIMIT 1",
            _RUNNABLE,
        ).fetchone()
        if row is None:
            return None
        cur = conn.execute(
            "UPDATE jobs SET status=?, started_at=? WHERE id=? AND status IN (?,?)",
            (JobStatus.RUNNING.value, now(), row["id"], *_RUNNABLE),
        )
        if cur.rowcount != 1:
            return None
        job = _row_to_dict(row)
    job["status"] = JobStatus.RUNNING.value
    return job


def waiting_jobs() -> list[dict[str, Any]]:
    """Queued and paused jobs, in the order the worker will run them."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM jobs WHERE status IN (?,?) {_QUEUE_ORDER}", _RUNNABLE
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def higher_priority_waiting(rank: int) -> bool:
    """Is strictly higher-priority work waiting?

    Strictly, so equal-priority jobs never preempt each other -- otherwise two
    same-priority jobs would trade the GPU at every epoch and neither finish.
    """
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE status IN (?,?) AND priority > ? LIMIT 1",
            (*_RUNNABLE, rank),
        ).fetchone()
    return row is not None


def running_job() -> dict[str, Any] | None:
    """The job currently training, if any. At most one: the worker is serial."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status=? ORDER BY started_at LIMIT 1",
            (JobStatus.RUNNING.value,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def queue_position(job_id: str) -> int | None:
    """1-based position among waiting jobs, or None if not waiting."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT created_at, priority FROM jobs WHERE id=? AND status IN (?,?)",
            (job_id, *_RUNNABLE),
        ).fetchone()
        if row is None:
            return None
        # Mirrors _QUEUE_ORDER: strictly higher priority is ahead, and within
        # the same priority, anything older is ahead.
        ahead = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status IN (?,?) AND ("
            "  priority > ? OR (priority = ? AND (created_at, id) < (?, ?))"
            ")",
            (*_RUNNABLE, row["priority"], row["priority"], row["created_at"], job_id),
        ).fetchone()["n"]
    return ahead + 1


def interrupted_jobs() -> list[dict[str, Any]]:
    """Rows still marked RUNNING, i.e. orphaned by a crash or restart."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status=?", (JobStatus.RUNNING.value,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


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
