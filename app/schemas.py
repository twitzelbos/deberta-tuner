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


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


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


class JobDetail(JobSummary):
    config: dict[str, Any]
    metrics: dict[str, Any] | None = None
    labels: list[str] | None = None
    num_train: int | None = None
    num_eval: int | None = None
    artifact_bytes: int | None = None
