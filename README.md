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

## Alternatives, and why this exists

Plenty of tools fine-tune transformers. Very few expose it as a **persistent
HTTP service with a job queue and artifact download**, which is the only thing
this project does.

| Tool | What it is | Why you might prefer it |
|---|---|---|
| [HuggingFace AutoTrain](https://github.com/huggingface/autotrain-advanced) | No-code training for LLMs, encoders, vision, tabular; CLI, UI and Python SDK; runs locally or on HF infrastructure | Far broader task and modality coverage, a real UI, and sensible automatic hyperparameters. The closest comparable by far. |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Unsloth](https://github.com/unslothai/unsloth) | YAML/CLI-driven LLM fine-tuning, LoRA/QLoRA, memory-optimised | You are tuning generative LLMs, not encoders |
| [Ludwig](https://github.com/ludwig-ai/ludwig), [AutoGluon](https://github.com/autogluon/autogluon) | Declarative / AutoML pipelines | You want automatic model and hyperparameter search |
| [SetFit](https://github.com/huggingface/setfit) | Few-shot fine-tuning of sentence transformers | You have only tens of labelled examples |
| Determined, Ray Train, Kubeflow | Cluster-grade training platforms with REST APIs | You have multiple GPUs or nodes and need scheduling, HPO and experiment tracking |
| OpenAI / Vertex AI / SageMaker / Together / Modal | Hosted fine-tuning APIs | You would rather not run infrastructure at all |

### What this does differently

- **It is a service, not a tool run.** Callers submit over HTTP from any
  machine, poll, and download a tarball. They need no shell access, no shared
  filesystem, no Python environment, and no knowledge of the host. Most of the
  above are things *you* run per experiment; this one runs continuously and
  takes requests.
- **It cooperates with a co-tenant GPU service.** Before each job it stops a
  configured systemd unit — typically an LLM inference server holding most of
  the VRAM — and restarts it in a `finally` block. This lets one GPU host serve
  inference *and* accept tuning jobs without either side needing to know about
  the other. No tool above does this; they assume the GPU is theirs.
- **Errors arrive at submit time, with line numbers.** Data is parsed and
  validated in the request, so a bad row returns `400 line 7: 5 tokens but 4
  tags` rather than failing twenty minutes into a queued job.
- **Artifacts are self-describing.** Inferred labels are written into
  `id2label`/`label2id`, so the tarball loads with plain `AutoModelFor*` and no
  side-channel mapping file.
- **Long jobs survive restarts and yield to urgent work.** Epoch-boundary
  checkpointing means an interrupted job resumes where it stopped, and a
  three-level priority queue lets a `high` job preempt a running `normal` one at
  its next checkpoint. The queue is queryable, so callers can see their position
  and where a paused job stopped.
- **It is small enough to audit.** ~1,600 lines across seven modules, one
  process, SQLite for state, no broker, no Kubernetes, no container runtime.

### When to use something else

Being honest about the gaps — this project has:

- **No hyperparameter search.** You pass hyperparameters; it uses them.
  AutoTrain, Ludwig and AutoGluon will search for you.
- **No multi-GPU or distributed training,** and no scheduling beyond one job at
  a time. Use Determined or Ray Train on a cluster.
- **No LLM/LoRA support.** Encoders only. Use Axolotl or LLaMA-Factory.
- **No experiment tracking or UI** beyond job status, logs and the OpenAPI
  browser. No W&B or MLflow integration.
- **No authentication.** See
  [configuration.md](docs/configuration.md#adding-authentication).

If you want an AutoML system, use AutoTrain. If you want a training platform,
use Determined. If you want a small HTTP endpoint that turns JSONL into a
fine-tuned encoder on a GPU box you already own, that is this.

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

The co-tenant is held for as long as the queue has work, plus a short idle delay
(`TUNER_GPU_IDLE_RESTART_SECONDS`, default 60 s), so a burst of jobs costs one
outage rather than one per job. The unit to cycle is configured with
`TUNER_SGLANG_UNIT`, and the whole behaviour can be disabled with
`TUNER_SGLANG_CONTROL=0`. See
[GPU arbitration](docs/architecture.md#gpu-arbitration).

## Status

Working and tested. Six suites, all passing:

- `tests/smoke.py` — all four task types train end to end on CPU and save real
  weights.
- `tests/test_api.py` — 28 checks over the HTTP surface, including a real job
  submitted, polled to completion, and its artifact downloaded and inspected.
- `tests/test_queue.py` — 35 checks on priority ordering, preemption and crash
  recovery.
- `tests/test_resume.py` — 20 checks on checkpointing, resume and preemption
  against a real training run.
- `tests/test_gpu_arbitration.py` — 15 checks on holding the GPU across jobs and
  restarting the co-tenant after an idle delay.
- `tests/test_reinit_head.py` — 15 checks on adapting a 3-class NLI checkpoint
  to a 2-label task.

## Install

```bash
sudo ./deploy/install-system.sh
```

See [docs/operations.md](docs/operations.md#deployment).
