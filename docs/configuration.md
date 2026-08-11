# Configuration

Two layers: **`JobConfig`** is per-request; **environment variables** configure
the service and are set in the systemd unit.

## JobConfig

Sent as the JSON `config` part of `POST /v1/jobs`. Every field is optional.

| Field | Type | Default | Range | Notes |
|---|---|---|---|---|
| `base_model` | string | `microsoft/deberta-v3-base` | allow-list | rejected with `400` otherwise |
| `task` | enum | `sequence_classification` | 4 values | determines the data schema |
| `epochs` | float | `3.0` | `>0`, `≤100` | fractional allowed |
| `batch_size` | int | `16` | `1`–`256` | first thing to lower on OOM |
| `eval_batch_size` | int | `32` | `1`–`256` | no gradients, so can exceed `batch_size` |
| `learning_rate` | float | `2e-5` | `>0`, `≤1e-2` | `1e-5`–`3e-5` is the usual band |
| `weight_decay` | float | `0.01` | `0`–`1` | AdamW |
| `warmup_ratio` | float | `0.06` | `0`–`<1` | fraction of total steps |
| `max_length` | int | `256` | `8`–`1024` | tokens; cost grows quadratically |
| `eval_split` | float | `0.1` | `0`–`<0.9` | ignored if `eval_file` is uploaded |
| `seed` | int | `42` | — | seeds split, shuffling, init |
| `threshold` | float | `0.5` | `>0`, `<1` | multi-label only |
| `name` | string | `null` | — | free-text label |

Out-of-range values return `400` with the pydantic error array.

### Notes

**`epochs`** — total steps are `ceil(steps_per_epoch * epochs)`, so `0.5` runs
half an epoch. The LR schedule is computed from the total, so changing `epochs`
changes the decay curve, not just the duration.

**`batch_size`** — no gradient accumulation, so this is the true batch. Lower it
for OOM, and consider raising `learning_rate` slightly if you drop it a long way.

**`max_length`** — sequences beyond this are truncated. For token
classification, tags for truncated words are silently dropped, which quietly
depresses recall on long records.

**`eval_split`** — a seeded random split, **not stratified**. Set to `0` to train
on everything; you then get no quality metrics.

**`threshold`** — affects reported metrics and is not baked into the model. When
serving, apply your own sigmoid cutoff.

**`base_model`** — allow-listed deliberately. An open field would let any caller
on the subnet name an arbitrary hub repo, and model loading can execute
repo-supplied code.

## Environment variables

Set in `deploy/deberta-tuner.service`. Changing one means editing the unit,
`systemctl daemon-reload`, and restarting.

| Variable | Default | Purpose |
|---|---|---|
| `TUNER_STATE_DIR` | `./state` | jobs database and all job files |
| `TUNER_HOST` | `0.0.0.0` | bind address |
| `TUNER_PORT` | `8100` | bind port |
| `TUNER_MAX_UPLOAD_MB` | `512` | per-file upload cap |
| `TUNER_SGLANG_UNIT` | `sglang.service` | unit to stop and start around jobs |
| `TUNER_SGLANG_CONTROL` | `1` | `0` disables GPU arbitration entirely |
| `TUNER_FORCE_CPU` | `0` | `1` forces CPU even if CUDA is present |
| `TUNER_ALLOWED_BASE_MODELS` | 7 DeBERTa variants | comma-separated allow-list |
| `HF_HOME` | user default | where base models are cached |

The unit sets `TUNER_STATE_DIR=/var/lib/deberta-tuner` and
`HF_HOME=/var/lib/deberta-tuner/hf-cache`.

### `TUNER_SGLANG_CONTROL`

Set to `0` when no other service contends for the GPU, or during development.
With it off the worker assumes the GPU is already free and will OOM if something
else holds it.

### `TUNER_FORCE_CPU`

Used by the test suites so they can run while another service holds the GPU.
Training on CPU is viable only for the smallest models and toy datasets.

### `TUNER_ALLOWED_BASE_MODELS`

Comma-separated. To evaluate ModernBERT — faster and better at long context,
though DeBERTa-v3 tends to retain an accuracy edge on short text at modest data
sizes:

```ini
Environment=TUNER_ALLOWED_BASE_MODELS=microsoft/deberta-v3-base,microsoft/deberta-v3-large,answerdotai/ModernBERT-base,answerdotai/ModernBERT-large
```

Nothing in the service is DeBERTa-specific — it uses `AutoModelFor*` throughout,
so any encoder with a compatible head works. Long-context models still obey the
`max_length` ceiling of 1024.

## Adding authentication

There is none; the service is open to anyone who can reach the port. The
lightest fix is an authenticating proxy in front, or a FastAPI dependency
checking a shared header:

```python
from fastapi import Depends, Header, HTTPException

def require_key(x_api_key: str = Header(...)):
    if x_api_key != os.environ["TUNER_API_KEY"]:
        raise HTTPException(401, "bad api key")

app = FastAPI(lifespan=lifespan, dependencies=[Depends(require_key)])
```

Exempt `/healthz` if your monitoring cannot send headers.

## Resource ceilings

Not currently configurable, and worth knowing:

- No per-job time limit. A runaway job blocks the queue until cancelled.
- No disk quota. See [operations](operations.md#disk-usage).
- No concurrency control beyond the single worker.
- `TimeoutStopSec=300` in the unit gives an in-flight job five minutes to unwind
  on restart — enough for its `finally` block to restart the co-tenant service.
