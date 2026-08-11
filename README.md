# DeBERTa tuning service

An HTTP service that fine-tunes DeBERTa models on uploaded JSONL data and
returns the trained model as a downloadable tarball.

Submit a dataset, poll for progress, download a standard HuggingFace model
directory. Four task types are supported: sequence classification, multi-label
classification, token classification (NER) and regression.

```bash
export TUNER_URL=http://your-server:8100

# 1. submit
curl -sS -X POST $TUNER_URL/v1/jobs \
  -F 'config={"task":"sequence_classification","epochs":3}' \
  -F 'train_file=@train.jsonl'
# -> {"id":"a1b2c3...","status":"queued", ...}

# 2. poll
curl -sS $TUNER_URL/v1/jobs/a1b2c3... | jq '{status,progress,metrics}'

# 3. download
curl -sS -OJ $TUNER_URL/v1/jobs/a1b2c3.../artifact
```

## Documentation

**Calling the service from another machine? Read
[docs/client-guide.md](docs/client-guide.md) — it is self-contained and
everything else here is about running the server.**

| Document | Audience | Contents |
|---|---|---|
| [docs/client-guide.md](docs/client-guide.md) | **remote users** | the whole workflow: data prep, submit, poll, download, load the model, errors, limits |
| [docs/data-formats.md](docs/data-formats.md) | everyone | JSONL schema per task, validation rules, label handling |
| [docs/api.md](docs/api.md) | everyone | endpoint reference, status codes, response schemas |
| [docs/configuration.md](docs/configuration.md) | operators | every `JobConfig` field and environment variable |
| [docs/architecture.md](docs/architecture.md) | maintainers | components, job lifecycle, GPU arbitration, storage |
| [docs/operations.md](docs/operations.md) | operators | deployment, monitoring, troubleshooting runbook |
| [docs/development.md](docs/development.md) | maintainers | local setup, tests, adding a task type |

Interactive OpenAPI UI is served at `/docs` on the running service.

## Requirements

- Linux with systemd
- An NVIDIA GPU with a working driver, plus a CUDA toolkit if you want kernels
  JIT-compiled
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) — dependencies are declared in
  `pyproject.toml` and pinned in `uv.lock`

## The one thing to know before using it

Training needs the whole GPU. If another service on the same host holds VRAM —
an LLM inference server, for example — **the tuner stops it for the duration of
each job** and restarts it afterwards.

That means submitting a tuning job takes the co-tenant service offline for the
length of the job plus its restart time. The unit to cycle is configured with
`TUNER_SGLANG_UNIT`, and the whole behaviour can be disabled with
`TUNER_SGLANG_CONTROL=0`. See
[GPU arbitration](docs/architecture.md#gpu-arbitration).

## Status

Working and tested. Two suites, both passing:

- `tests/smoke.py` — all four task types train end to end on CPU and save real
  weights.
- `tests/test_api.py` — 28 checks over the HTTP surface, including a real job
  submitted, polled to completion, and its artifact downloaded and inspected.

## Install

```bash
sudo ./deploy/install-system.sh
```

See [docs/operations.md](docs/operations.md#deployment).
