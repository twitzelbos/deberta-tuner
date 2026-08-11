"""Request/response models and the job state machine."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from . import config


class TaskType(str, Enum):
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    REGRESSION = "regression"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


# Stored as an integer so the queue can ORDER BY it directly.
PRIORITY_RANK: dict[str, int] = {
    Priority.LOW.value: 0,
    Priority.NORMAL.value: 1,
    Priority.HIGH.value: 2,
}
RANK_PRIORITY: dict[int, str] = {v: k for k, v in PRIORITY_RANK.items()}


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    # Preempted at a checkpoint by higher-priority work, or interrupted by a
    # service restart. Retains progress and resumes from its checkpoint.
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}

    @property
    def runnable(self) -> bool:
        """Waiting for the worker: eligible to be claimed."""
        return self in {JobStatus.QUEUED, JobStatus.PAUSED}


class JobConfig(BaseModel):
    """Tuning hyperparameters. Sent as the JSON `config` part of the upload."""

    base_model: str = "microsoft/deberta-v3-base"
    task: TaskType = TaskType.SEQUENCE_CLASSIFICATION
    epochs: float = Field(default=3.0, gt=0, le=100)
    batch_size: int = Field(default=16, ge=1, le=256)
    eval_batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=2e-5, gt=0, le=1e-2)
    weight_decay: float = Field(default=0.01, ge=0, le=1.0)
    warmup_ratio: float = Field(default=0.06, ge=0, lt=1.0)
    max_length: int = Field(default=256, ge=8, le=1024)
    # When no eval file is uploaded, hold out this fraction of the training set.
    eval_split: float = Field(default=0.1, ge=0.0, lt=0.9)
    seed: int = 42
    # Sigmoid cutoff for multi-label prediction.
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    name: str | None = None
    # Write a resumable checkpoint every N completed epochs; 0 disables.
    # Costs roughly 3x the model size on disk (weights + AdamW moments) and one
    # write per interval, so turn it off for short jobs if that matters.
    #
    # NOTE: setting this to 0 also opts the job out of preemption -- with no
    # checkpoint there is nothing to resume from, so a higher-priority job must
    # wait for it to finish.
    checkpoint_every_epochs: int = Field(default=1, ge=0, le=100)
    # Higher priority preempts lower at the next epoch boundary.
    priority: Priority = Priority.NORMAL

    @field_validator("base_model")
    @classmethod
    def _known_base(cls, v: str) -> str:
        if v not in config.ALLOWED_BASE_MODELS:
            allowed = ", ".join(sorted(config.ALLOWED_BASE_MODELS))
            raise ValueError(f"base_model {v!r} not allowed; choose one of: {allowed}")
        return v


class JobSummary(BaseModel):
    id: str
    name: str | None
    status: JobStatus
    task: TaskType
    base_model: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: float = 0.0
    error: str | None = None
    priority: Priority = Priority.NORMAL
    # Epochs finished and checkpointed. Non-zero on a waiting job means it was
    # paused and will resume rather than start over.
    epochs_completed: int = 0
    # 1-based place in the queue; null unless the job is waiting to run.
    queue_position: int | None = None
    # How many times higher-priority work has bumped this job.
    preempted_count: int = 0


class JobDetail(JobSummary):
    config: dict[str, Any]
    metrics: dict[str, Any] | None = None
    labels: list[str] | None = None
    num_train: int | None = None
    num_eval: int | None = None
    artifact_bytes: int | None = None


class RunningJob(BaseModel):
    id: str
    name: str | None
    task: TaskType
    base_model: str
    priority: Priority
    started_at: str | None
    progress: float
    epochs_completed: int
    elapsed_seconds: int
    # Naive extrapolation from elapsed time and progress; null until the first
    # step lands. Ignores evaluation and model saving, so it runs optimistic.
    eta_seconds: int | None
    # True when higher-priority work is waiting, so this job will yield at its
    # next epoch boundary.
    yielding: bool = False


class QueuedJob(BaseModel):
    position: int
    id: str
    name: str | None
    # `queued` = never started. `paused` = started, then preempted or
    # interrupted; `progress` and `epochs_completed` show where it stopped.
    status: JobStatus
    priority: Priority
    task: TaskType
    base_model: str
    created_at: str
    waiting_seconds: int
    progress: float
    epochs_completed: int
    preempted_count: int


class QueueView(BaseModel):
    """Everything a caller needs to answer 'when will my job run?'."""

    running: RunningJob | None
    # Ordered exactly as the worker will run them: priority first, then age.
    waiting: list[QueuedJob]
    queued_count: int
    paused_count: int
    # Jobs executed at once. One GPU, one worker, so always 1 for now.
    concurrency: int = 1
