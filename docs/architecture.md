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
        | worker claims highest priority (atomic UPDATE)
        v
   [running] --------- cancel ---------> [cancelled]
        |  ^                             (checked once per training step)
        |  |
        |  +--- claimed again when nothing better is waiting
        |  |
        |  v
        +--> [paused] ---- cancel ------> [cancelled]
        |     ^  preempted by higher priority, or interrupted by shutdown;
        |     |  keeps its checkpoint, progress and epochs_completed
        |
        +----- success ----> [succeeded]  (artifact available)
        |
        +----- exception --> [failed]     (error field populated)
```

`succeeded`, `failed` and `cancelled` are terminal. `queued` and `paused` are
both *runnable*: the worker claims from either.

Claiming is atomic: the worker does a conditional
`UPDATE ... WHERE id=? AND status IN ('queued','paused')` and only proceeds if
exactly one row changed. That keeps the design safe if a second worker is ever
added.

**Crash recovery.** A job can only be `running` while the worker owns it, so a
surviving `running` row means the process died. At startup
`worker.recover_interrupted()` reconciles each one:

| Condition | Outcome |
|---|---|
| cancel was requested | `cancelled`, checkpoint deleted |
| a checkpoint exists | `paused` — goes back on the queue and resumes |
| no checkpoint | `failed`, explaining that nothing was resumable |

## Priority and preemption

Three levels: `low`, `normal` (default) and `high`. The queue is ordered by
priority descending, then by age, then by id as a deterministic tiebreak. Paused
jobs compete on equal terms with jobs that have never started, so a paused
high-priority job still outranks a fresh normal one.

A running job checks for **strictly** higher-priority waiting work at each epoch
boundary. If it finds any, it writes a checkpoint — even off its normal interval
— and raises `Preempted`. The worker marks it `paused` and increments
`preempted_count`; the queue ordering then guarantees it does not run again until
nothing higher-priority is waiting.

Two consequences worth being explicit about:

- **Strictly higher, never equal.** If equal priorities preempted each other,
  two same-priority jobs would trade the GPU at every epoch and neither would
  finish.
- **A job with `checkpoint_every_epochs: 0` cannot be preempted.** There would be
  nothing to resume from, so higher-priority work waits for it to finish. This is
  the one way to make a job non-interruptible.

Preemption is only ever evaluated at an epoch boundary, because that is the only
point where a checkpoint is coherent. A single very long epoch therefore delays
preemption until it completes.

## Checkpointing

Granularity is a whole epoch. Resuming restarts the interrupted epoch rather
than replaying the dataloader to an exact batch, which avoids having to restore
iteration order and keeps the state small enough to reason about.

A checkpoint holds the model, optimizer and scheduler state, the epoch and step
counters, the label list and the torch RNG state. It is written to
`jobs/<id>/checkpoint/checkpoint.pt` via a temporary file and an atomic rename,
so a crash mid-write can never replace a good checkpoint with a truncated one.

On load, the checkpoint's label list is compared against the current data; a
mismatch is logged and the checkpoint ignored rather than trusted.

Checkpoints are large — roughly three times the model size, since AdamW keeps
two moments per parameter — so they are deleted as soon as the job reaches a
terminal state.

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
