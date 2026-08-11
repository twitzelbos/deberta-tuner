# Architecture

## Components

```
                  HTTP (FastAPI, uvicorn)
                          |
   POST /v1/jobs ---> validate ---> write files ---> INSERT job (queued)
                          |
                          |  SQLite (jobs.db)
                          v
                  worker thread (serial)
                          |
        stop co-tenant unit  ------------------> GPU freed
                          |
                    training.run()
                          |
       start co-tenant unit  <----------------- finally block, always
                          |
                 model.tar.gz on disk
```

| Module | Responsibility |
|---|---|
| `app/main.py` | HTTP surface, upload handling, request validation |
| `app/schemas.py` | `JobConfig`, `JobStatus`, `TaskType`, response models |
| `app/data.py` | JSONL parsing, per-task validation, train/eval split |
| `app/training.py` | the training loop, metrics, model saving |
| `app/worker.py` | job queue, GPU arbitration, artifact packaging |
| `app/db.py` | SQLite job store |

There is no external broker or task queue. One process holds the API and a
single background worker thread; the queue is a table in SQLite. This is
deliberate — see [why serial](#why-serial).

## Job lifecycle

```
   POST /v1/jobs
        |
        v
    [queued] --------- cancel ---------> [cancelled]
        |
        | worker claims (atomic UPDATE)
        v
   [running] --------- cancel ---------> [cancelled]
        |                                (checked once per training step)
        +----- success ----> [succeeded]  (artifact available)
        |
        +----- exception --> [failed]     (error field populated)
```

`succeeded`, `failed` and `cancelled` are terminal.

Claiming is atomic: the worker does a conditional
`UPDATE ... WHERE id=? AND status='queued'` and only proceeds if exactly one row
changed. That makes the design safe if a second worker is ever added.

**Crash recovery.** A job can only be `running` while the worker owns it. On
startup `db.init()` marks any surviving `running` row as `failed` with
`"service restarted while this job was running"`. There is no checkpoint or
resume; a job interrupted by a restart must be resubmitted.

## GPU arbitration

Fine-tuning needs most of a GPU. On a host where an inference server already
reserves the bulk of VRAM — LLM servers typically pre-allocate weights plus a KV
cache at startup — there is no room to train alongside it, and the reservation
usually cannot be shrunk far enough because the model weights themselves set a
hard floor.

Rather than fight for memory, the tuner takes exclusive use of the GPU for the
duration of a job by cycling the other service's systemd unit
(`TUNER_SGLANG_UNIT`, default `sglang.service`).

Sequence per job:

1. `_release_gpu()` checks whether the configured unit is installed and active.
2. If active, `sudo -n systemctl stop <unit>`.
3. Poll `nvidia-smi` for up to 30 s until free memory exceeds 20 GiB. `systemctl`
   returns as soon as the process is gone, but the driver takes a moment to
   reclaim the allocation.
4. Train.
5. In a `finally` block, `sudo -n systemctl start <unit>`.

The restart lives in `finally` specifically so that a crashed, cancelled or
failed job never leaves the inference endpoint down.

`_release_gpu()` returns `True` only if it actually stopped a running unit, so
the tuner will not start the co-tenant service if it was already stopped when
the job began.

Set `TUNER_SGLANG_CONTROL=0` to disable all of this — appropriate for
development, or when nothing else contends for the GPU.

### Requirements this imposes

- A sudoers grant for two exact commands (`systemctl stop|start <unit>`).
- `NoNewPrivileges=yes` **must not** be set on the tuner unit; it blocks setuid
  `sudo` and would break arbitration silently.

## Why serial

One GPU means one training job at a time. Rather than admit concurrency and then
fight over memory, the worker is a single thread that processes the queue
strictly in creation order. Consequences:

- Queue depth is visible at `/healthz`.
- A long job blocks everything behind it; cancel it if needed.
- No GPU OOM from two jobs colliding.

## Storage layout

```
$TUNER_STATE_DIR/                 # /var/lib/deberta-tuner in production
├── jobs.db                       # SQLite, WAL mode
└── jobs/
    └── <job_id>/
        ├── config.json           # the submitted JobConfig
        ├── train.jsonl           # uploaded training data
        ├── eval.jsonl            # uploaded eval data, if any
        ├── train.log             # timestamped training log
        ├── output/               # HuggingFace model directory
        │   ├── config.json       # includes id2label / label2id
        │   ├── model.safetensors
        │   ├── tokenizer.json, spm.model, ...
        │   └── training_metrics.json
        └── model.tar.gz          # the downloadable artifact
```

`model.tar.gz` contains a single top-level directory named after the job id, so
extracting it never scatters files into the current directory.

Nothing is garbage-collected automatically. Old jobs accumulate until deleted
with `DELETE /v1/jobs/{id}`; see
[operations](operations.md#disk-usage).

## Training loop

`app/training.py` implements the loop directly rather than using
`transformers.Trainer`. The per-task differences — loss shape, label alignment,
metric choice — stay visible in one file, and there is no coupling to Trainer's
evolving API.

- AdamW, linear warmup then linear decay, gradient clipping at 1.0.
- `torch.autocast` with bf16 on CUDA (no `GradScaler` — bf16 does not need one).
  Disabled on CPU.
- Parameters are forced to fp32 after loading. transformers 5 loads checkpoints
  in their native dtype, and several DeBERTa checkpoints are fp16, which breaks
  the regression head and degrades token classification. See
  [operations](operations.md#found-dtype-float-but-expected-half).
- Progress is written to the database every step; cancellation is checked every
  step.

Total steps are computed as `ceil(len(train_loader) * epochs)`, so fractional
epochs work (`"epochs": 0.5`).
