# Development

Work in a checkout of this repository; production runs from
`/opt/deberta-tuner`. Edit and test in the checkout, then run the installer to
deploy.

## Setup

```bash
uv sync              # creates .venv and installs from uv.lock, dev group included
source .venv/bin/activate
```

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. Never
install into the venv with bare `pip` — the next `uv sync` will undo it. To add
a dependency, edit `pyproject.toml` then run `uv lock && uv sync`, and commit
both files.

If a co-tenant service is holding the GPU, develop with `TUNER_FORCE_CPU=1` and
`TUNER_SGLANG_CONTROL=0` so you neither need the GPU nor take that service down.

```bash
TUNER_FORCE_CPU=1 TUNER_SGLANG_CONTROL=0 TUNER_STATE_DIR=./state \
  uvicorn app.main:app --reload --port 8100
```

`--reload` restarts on save, which kills any in-flight job.

## Tests

Neither suite needs pytest; both are plain scripts that exit non-zero on
failure.

```bash
python -m tests.test_queue                   # seconds: priority, preemption, recovery
python -m tests.test_gpu_arbitration         # ~20s: co-tenant hold and idle restart
TUNER_FORCE_CPU=1 python -m tests.smoke      # ~3 min: all 4 tasks, real weights
python -m tests.test_resume                  # ~4 min: checkpoint, resume, preempt
python -m tests.test_reinit_head             # ~2 min: NLI head adaptation
python -m tests.test_api                     # ~4 min: 28 checks over HTTP
```

### `tests/smoke.py`

Trains `deberta-v3-xsmall` on ~24 synthetic rows for each task type and asserts
that weights and config are written and metrics returned. Also asserts the five
validation errors produce the right messages.

It verifies **plumbing, not learning quality** — 5 steps on 18 examples. Metrics
land around 0.6–0.7, which proves the loss and metric wiring is connected, not
that the model is good. Do not tighten these into quality assertions; they will
flake.

### `tests/test_api.py`

Drives the real FastAPI app through `TestClient`, with the worker thread running
in-process against a temp state dir. Covers health, allow-list rejection, data
validation with line numbers, submit, queue, poll to `succeeded`, metrics,
artifact download, tar contents, `id2label` persistence, and the lifecycle
`409`/`404`/`204` paths.

### `tests/test_queue.py`

Priority ordering, preemption predicates, paused-job scheduling and crash
recovery, driven against the job store. It calls the route functions directly
rather than through `TestClient`, because the app's lifespan starts the worker
thread and it would claim the very jobs the assertions are inspecting.

### `tests/test_gpu_arbitration.py`

Stubs out both the trainer and the systemctl calls and drives the real worker
loop, asserting that consecutive jobs share one co-tenant stop, that a job
arriving inside the idle window cancels the pending restart, and that stops and
starts stay balanced across shutdown.

### `tests/test_resume.py`

A real training run interrupted after its first checkpoint, then resumed and
verified to continue rather than restart; plus preemption yielding at an epoch
boundary, and the guarantee that `checkpoint_every_epochs: 0` opts out.

All suites write to a fresh `tempfile.mkdtemp()`, so they never touch real
state.

## Layout

```
app/
├── config.py      env-derived settings, allow-list
├── schemas.py     JobConfig, TaskType, JobStatus, responses
├── data.py        JSONL parsing, validation, split
├── training.py    the training loop, metrics, saving
├── worker.py      queue, GPU arbitration, tarball
├── db.py          SQLite store
└── main.py        FastAPI routes
deploy/            unit, sudoers, installer, staged uv
docs/              this documentation
tests/             smoke.py, test_api.py
```

Dependencies flow one way: `main` → `worker` → `training` → `data` → `schemas` →
`config`. Nothing imports `main`.

## Adding a task type

1. Add the value to `TaskType` in `schemas.py`.
2. Extend `data.parse()` with a validation branch — always include the line
   number in errors.
3. In `training.py`: pick `num_labels`/`problem_type`, add a dataset branch if
   the encoding differs, extend `_make_collate` if the label tensor dtype
   differs, and add a `_metrics` branch.
4. Add a case to `tests/smoke.py`.
5. Document it in [data-formats.md](data-formats.md) and
   [client-guide.md](client-guide.md).

Most of the work is `_metrics` and the label dtype. Classification uses `long`;
regression and multi-label use `float`.

## Gotchas

**Force fp32 after `from_pretrained`.** transformers ≥5 loads in the
checkpoint's native dtype and several DeBERTa checkpoints are fp16, which
crashes the regression head and degrades token classification. Keep
`.to(dtype=torch.float32)`; bf16 autocast handles mixed precision on GPU.

**Do not use `Trainer`.** The loop is deliberately explicit so per-task
differences stay visible in one file.

**`config.py` reads the environment at import.** Tests must set env vars
*before* importing anything from `app`.

**The worker is a daemon thread.** Uncaught exceptions inside `_run_job` are
trapped and recorded on the job; exceptions in `_loop` itself would kill the
worker silently. Keep `_loop` trivial.

**Progress and cancellation are per-step.** Each writes to SQLite every step;
fine at these batch counts, but a very large dataset with tiny batches makes the
DB the bottleneck. Batch the writes if that ever matters.

## Deploying a change

```bash
TUNER_FORCE_CPU=1 python -m tests.smoke && python -m tests.test_api
sudo ./deploy/install-system.sh
```

The installer re-syncs `app/`, re-resolves requirements, restarts and waits for
`/healthz`. It does not roll back — check for running jobs first, since a
restart fails them.
