# API reference

Complete endpoint reference. If you are calling the service from another
machine, [client-guide.md](client-guide.md) is the friendlier starting point.

Base URL in production: `$TUNER_URL`. OpenAPI schema at
`/openapi.json`, interactive UI at `/docs`.

No authentication. All endpoints are unauthenticated and the service binds
`0.0.0.0`.

> The generated OpenAPI schema declares only the success code and `422` for each
> route. The `400`/`404`/`409`/`410`/`413` responses documented below are real
> but raised via `HTTPException` and not declared in the schema, so they do not
> appear in `/docs`.

---

## `GET /healthz`

Liveness plus enough state to decide whether to submit.

**200**

```json
{
  "status": "ok",
  "cuda_available": true,
  "gpu_free_mib": 2470,
  "queued": 0,
  "paused": 0,
  "running": 0
}
```

| Field | Meaning |
|---|---|
| `cuda_available` | whether torch can see the GPU |
| `gpu_free_mib` | free VRAM now; low is normal while a co-tenant service holds the GPU |
| `queued` / `running` | queue depth |

A low `gpu_free_mib` is not a fault: a co-tenant inference service typically
holds most of the VRAM until a job stops it.

---

## `GET /v1/base-models`

**200**

```json
{"base_models": ["microsoft/deberta-base", "microsoft/deberta-v3-base", "..."]}
```

Reflects `TUNER_ALLOWED_BASE_MODELS`. Any other `base_model` is rejected with
`400`.

---

## `GET /v1/queue`

What is running, what is waiting, and in what order.

**200**

```json
{
  "running": {
    "id": "9f2c...", "name": "sentiment-v2", "task": "sequence_classification",
    "base_model": "microsoft/deberta-v3-base", "priority": "normal",
    "started_at": "2026-08-11T09:14:03+00:00", "progress": 0.62,
    "epochs_completed": 1, "elapsed_seconds": 412, "eta_seconds": 252,
    "yielding": false
  },
  "waiting": [
    {"position": 1, "id": "a1b2...", "status": "paused",  "priority": "normal",
     "progress": 0.4, "epochs_completed": 2, "preempted_count": 1,
     "waiting_seconds": 900, "name": null, "task": "sequence_classification",
     "base_model": "microsoft/deberta-v3-base", "created_at": "..."},
    {"position": 2, "id": "c3d4...", "status": "queued", "priority": "low",
     "progress": 0.0, "epochs_completed": 0, "preempted_count": 0,
     "waiting_seconds": 120, "name": null, "task": "regression",
     "base_model": "microsoft/deberta-v3-base", "created_at": "..."}
  ],
  "queued_count": 1,
  "paused_count": 1,
  "concurrency": 1
}
```

`waiting` is ordered exactly as the worker will run the jobs: priority
descending, then oldest first.

| Field | Meaning |
|---|---|
| `running.eta_seconds` | straight-line extrapolation from elapsed time and progress; `null` until the first step. Ignores evaluation and saving, so it runs optimistic |
| `running.yielding` | `true` when higher-priority work is waiting, so this job will pause at its next epoch boundary |
| `waiting[].status` | `queued` = never started; `paused` = started then preempted or interrupted |
| `waiting[].progress` | where a paused job stopped and will resume from; `0.0` for never-started jobs |
| `waiting[].preempted_count` | how many times higher-priority work has bumped this job |
| `concurrency` | jobs run at once; always `1` |

---

## `POST /v1/jobs`

Submit a tuning job. `multipart/form-data`.

| Part | Type | Required |
|---|---|---|
| `config` | text — a JSON object ([JobConfig](configuration.md#jobconfig)) | yes |
| `train_file` | file — JSONL | yes |
| `eval_file` | file — JSONL | no |

Processing order: parse `config`, create the job directory, stream uploads to
disk enforcing the size cap, then parse and validate both files. Any failure
purges the directory so no partial job is left behind.

**201** — a [JobDetail](#jobdetail) with `"status": "queued"`.

**400**

- `config` is not valid JSON → `{"detail": "config is not valid JSON: ..."}`
- `config` fails validation → `detail` is the pydantic error array, e.g. a
  `base_model` outside the allow-list or `epochs` out of range
- data is invalid → `{"detail": "invalid training data: line 7: 5 tokens but 4 tags"}`
- an uploaded file is empty → `{"detail": "train.jsonl is empty"}`

**413** — upload exceeds `TUNER_MAX_UPLOAD_MB` (default 512). Enforced during
streaming; the partial file is removed.

**422** — malformed multipart, or a required part missing.

```bash
curl -sS -X POST $TUNER_URL/v1/jobs \
  -F 'config={"task":"token_classification","base_model":"microsoft/deberta-v3-base","epochs":5}' \
  -F 'train_file=@train.jsonl' \
  -F 'eval_file=@dev.jsonl'
```

---

## `GET /v1/jobs`

| Query | Type | Default | Notes |
|---|---|---|---|
| `status` | enum | — | `queued`, `running`, `succeeded`, `failed`, `cancelled` |
| `limit` | int 1–1000 | 100 | |

**200** — array of [JobSummary](#jobsummary), newest first.

**422** — unrecognised `status`, or `limit` out of range.

---

## `GET /v1/jobs/{job_id}`

**200** — [JobDetail](#jobdetail). **404** — unknown job.

---

## `GET /v1/jobs/{job_id}/logs`

Plain-text training log.

| Query | Type | Default | Notes |
|---|---|---|---|
| `tail` | int ≥ 0 | 0 | `0` returns the whole log |

**200** — `text/plain`; empty string if the job has not started.
**404** — unknown job.

Each line is prefixed with a UTC timestamp. Includes the resolved device, label
set, step losses, eval metrics, GPU arbitration results, and any traceback.

---

## `POST /v1/jobs/{job_id}/cancel`

- `queued` or `paused` → immediately `cancelled`, and any checkpoint is
  deleted (no worker is watching the flag).
- `running` → sets a flag; the training loop checks it every step and unwinds
  within seconds. The co-tenant service is still restarted.

**200** — updated [JobDetail](#jobdetail). For a running job the status may
still read `running` for a moment.
**404** — unknown job.
**409** — already `succeeded`, `failed` or `cancelled`.

---

## `GET /v1/jobs/{job_id}/artifact`

**200** — `application/gzip`, `Content-Disposition: attachment; filename=<job_id>.tar.gz`.

A gzipped tar containing one top-level directory named after the job id:

```
<job_id>/
├── config.json              # includes id2label / label2id
├── model.safetensors
├── tokenizer.json, tokenizer_config.json, spm.model
├── special_tokens_map.json
└── training_metrics.json
```

**404** — unknown job.
**409** — job is not `succeeded`.
**410** — job succeeded but the artifact is gone from disk.

```bash
curl -sS -OJ $TUNER_URL/v1/jobs/$JOB/artifact
```

---

## `DELETE /v1/jobs/{job_id}`

Removes the database row and the entire job directory: uploads, logs, model and
artifact.

**204** — deleted. **404** — unknown job. **409** — job is `running`; cancel it
first.

Irreversible.

---

## Schemas

### JobSummary

| Field | Type | Notes |
|---|---|---|
| `id` | string | 16 hex chars |
| `name` | string \| null | from `config.name` |
| `status` | enum | |
| `task` | enum | |
| `base_model` | string | |
| `created_at` | string | ISO 8601 UTC, seconds |
| `started_at` | string \| null | when the worker claimed it |
| `finished_at` | string \| null | terminal timestamp |
| `progress` | float | 0.0–1.0 |
| `error` | string \| null | truncated to 2000 chars |
| `priority` | enum | `low`, `normal`, `high` |
| `epochs_completed` | int | epochs finished and checkpointed |
| `queue_position` | int \| null | 1-based; null unless queued or paused |
| `preempted_count` | int | times bumped by higher-priority work |

### JobDetail

Everything in `JobSummary`, plus:

| Field | Type | Notes |
|---|---|---|
| `config` | object | the effective `JobConfig` with defaults filled in |
| `metrics` | object \| null | task-dependent; always includes `train_steps` |
| `labels` | array \| null | model index order; `[]` for regression |
| `num_train` | int \| null | rows used for training after splitting |
| `num_eval` | int \| null | `0` when no eval set |
| `artifact_bytes` | int \| null | tarball size |

### Metrics by task

| Task | Keys |
|---|---|
| `sequence_classification` | `accuracy`, `f1_macro` |
| `multi_label_classification` | `f1_micro`, `f1_macro`, `subset_accuracy` |
| `token_classification` | `precision`, `recall`, `f1` (entity-level, seqeval) |
| `regression` | `mse`, `mae`, `pearson` (omitted if either side is constant) |

`train_steps` is always present. With `eval_split: 0` and no `eval_file`, it is
the only key.

### Status values

| Status | Terminal | Meaning |
|---|---|---|
| `queued` | no | waiting for the worker, never started |
| `running` | no | training |
| `paused` | no | started, then preempted by higher-priority work or interrupted by a restart. Retains its checkpoint, `progress` and `epochs_completed`, and resumes when nothing better is waiting |
| `succeeded` | yes | artifact available |
| `failed` | yes | see `error` and the log |
| `cancelled` | yes | cancelled by request |
