# Client guide

Everything you need to fine-tune a model on this service from your own machine.
Nothing here requires access to the server.

**Base URL:** `$TUNER_URL` (host `the server`)
**Interactive API browser:** <$TUNER_URL/docs>

No authentication. Anyone on the subnet can submit jobs; be considerate.

---

## Before you start: three things

1. **Jobs run one at a time,** on a priority queue. The server has a single
   GPU. Your job waits behind anything already running and anything of higher
   priority. Check `GET /v1/queue` to see where you are.
2. **Submitting a job may take another service on the same host offline.**
   Training needs the whole GPU, so the server stops any co-tenant GPU service
   while jobs are running. It stays down for as long as the queue has work, plus
   a short idle delay (60 s by default) before being restarted — so a burst of
   jobs costs one outage rather than one per job. Ask your operator what shares
   the host before running long jobs.
3. **Nothing is deleted automatically.** Delete your job when you have
   downloaded the model.

---

## Quickstart

```bash
HOST=$TUNER_URL

cat > train.jsonl <<'EOF'
{"text": "great value for the money", "label": "positive"}
{"text": "broke after two days", "label": "negative"}
{"text": "works exactly as described", "label": "positive"}
{"text": "complete waste of money", "label": "negative"}
EOF

# submit
JOB=$(curl -sS -X POST $HOST/v1/jobs \
  -F 'config={"base_model":"microsoft/deberta-v3-base","task":"sequence_classification","epochs":3}' \
  -F 'train_file=@train.jsonl' | jq -r .id)
echo "job: $JOB"

# poll until terminal
while :; do
  S=$(curl -sS $HOST/v1/jobs/$JOB | jq -r .status)
  echo "$S"
  case $S in succeeded|failed|cancelled) break;; esac
  sleep 10
done

# inspect and download
curl -sS $HOST/v1/jobs/$JOB | jq '{status, metrics, labels}'
curl -sS -OJ $HOST/v1/jobs/$JOB/artifact       # -> <job_id>.tar.gz
tar xzf $JOB.tar.gz
```

---

## Preparing your data

JSONL: one JSON object per line, UTF-8. Pick the shape matching your `task`.

### `sequence_classification` — one label per text

```jsonl
{"text": "great value", "label": "positive"}
{"text": "broke instantly", "label": "negative"}
```

Needs at least 2 distinct labels. Add `"text_pair"` for sentence-pair tasks
(NLI, paraphrase):

```jsonl
{"text": "A man plays guitar.", "text_pair": "Someone plays an instrument.", "label": "entailment"}
```

Reports `accuracy`, `f1_macro`.

### `multi_label_classification` — zero or more labels per text

```jsonl
{"text": "cheap but poorly made", "labels": ["price", "quality"]}
{"text": "no strong opinion", "labels": []}
```

Empty list is valid. Reports `f1_micro`, `f1_macro`, `subset_accuracy`.

### `token_classification` — a label per word (NER)

```jsonl
{"tokens": ["Ada", "works", "at", "Acme"], "tags": ["B-PER", "O", "O", "B-ORG"]}
```

`tokens` and `tags` **must be the same length** — the most common mistake. Text
must already be split into words; subword handling is automatic.

Use **IOB2** tags (`B-TYPE`, `I-TYPE`, `O`). Scoring is entity-level, so a
prediction counts only if the whole span and its type match. Non-IOB2 tags will
train fine but produce misleading F1.

Reports `precision`, `recall`, `f1`.

### `regression` — a number per text

```jsonl
{"text": "it was fine", "label": 3.0}
{"text": "absolutely superb", "label": 4.8}
```

Targets are not normalised; scale them yourself if the range is large. Reports
`mse`, `mae`, `pearson`.

### Evaluation data

Either upload a second file as `eval_file`, or let the server hold out
`eval_split` (default 10%) of your training data. Set `"eval_split": 0` to train
on everything, in which case you get no quality metrics.

The automatic split is random, **not stratified** — with few examples per class,
a rare class can land entirely on one side.

---

## Submitting

`POST /v1/jobs` as `multipart/form-data` with:

| Part | Required | What |
|---|---|---|
| `config` | yes | JSON object of settings (below) |
| `train_file` | yes | your JSONL |
| `eval_file` | no | separate eval JSONL |

Everything in `config` is optional; defaults shown:

| Field | Default | Notes |
|---|---|---|
| `base_model` | `microsoft/deberta-v3-base` | must be allow-listed, see below |
| `task` | `sequence_classification` | one of the four above |
| `epochs` | `3.0` | fractional allowed (`0.5`) |
| `batch_size` | `16` | lower to 8 or 4 if you hit out-of-memory |
| `eval_batch_size` | `32` | |
| `learning_rate` | `2e-5` | `1e-5`–`3e-5` is the usual band |
| `weight_decay` | `0.01` | |
| `warmup_ratio` | `0.06` | fraction of steps spent warming up |
| `max_length` | `256` | tokens; longer costs quadratic time and memory |
| `eval_split` | `0.1` | ignored if you upload `eval_file` |
| `seed` | `42` | |
| `threshold` | `0.5` | multi-label sigmoid cutoff only |
| `name` | `null` | free-text label to find your job later |
| `priority` | `normal` | `low`, `normal` or `high` |
| `checkpoint_every_epochs` | `1` | `0` disables checkpointing and makes the job non-interruptible |

Available `base_model` values — `GET /v1/base-models`:

| Model | Total params | Backbone | Notes |
|---|---|---|---|
| `microsoft/deberta-v3-xsmall` | 70M | 22M | fastest; good for pipeline tests |
| `microsoft/deberta-v3-small` | 142M | 44M | |
| `microsoft/deberta-v3-base` | 183M | 86M | the sensible default |
| `microsoft/deberta-v3-large` | 434M | 304M | best accuracy, slowest |
| `microsoft/mdeberta-v3-base` | 278M | 86M | multilingual (250k vocab) |
| `microsoft/deberta-base` | 139M | 86M | v1, legacy |
| `microsoft/deberta-large` | 405M | 304M | v1, legacy |

DeBERTa-v3 uses a 128k-token vocabulary, so a large share of the total is the
embedding table. Training speed tracks the **backbone** column; memory tracks
the total.

Your data is validated **while you wait**, so a bad file returns `400` with the
offending line number immediately:

```json
{"detail": "invalid training data: line 7: 5 tokens but 4 tags"}
```

A successful submit returns `201` and the job record with `"status": "queued"`.

---

## Priority, pausing and resuming

Three levels: `low`, `normal` (the default) and `high`.

A `high` job does not wait for a running `normal` one to finish. At the running
job's next **epoch boundary** it writes a checkpoint, pauses, and hands over the
GPU. The paused job shows `status: "paused"` with the `progress` it stopped at,
and resumes from that checkpoint once nothing higher-priority is waiting. No
work is lost beyond the partially-completed epoch.

Equal priorities never preempt each other, so two `high` jobs run one after the
other rather than trading the GPU.

Checkpointing also means a **service restart no longer kills your job** — it
comes back as `paused` and resumes automatically.

Two consequences worth knowing:

- Setting `"checkpoint_every_epochs": 0` makes your job **non-interruptible**:
  with nothing to resume from it cannot be preempted, so higher-priority work
  waits for it. It also means a restart loses the job entirely.
- Preemption only happens at epoch boundaries. A job with one very long epoch
  will not yield until that epoch finishes.

Use `high` sparingly — it delays everyone else.

```bash
curl -sS -X POST $HOST/v1/jobs \
  -F 'config={"task":"sequence_classification","priority":"high","epochs":3}' \
  -F 'train_file=@train.jsonl'
```

## Seeing the queue

```bash
curl -sS $HOST/v1/queue | jq
```

```json
{
  "running": {"id": "9f2c...", "priority": "normal", "progress": 0.62,
              "elapsed_seconds": 412, "eta_seconds": 252, "yielding": false},
  "waiting": [
    {"position": 1, "id": "a1b2...", "status": "paused", "priority": "normal",
     "progress": 0.4, "preempted_count": 1},
    {"position": 2, "id": "c3d4...", "status": "queued", "priority": "low",
     "progress": 0.0, "preempted_count": 0}
  ],
  "queued_count": 1, "paused_count": 1, "concurrency": 1
}
```

`waiting` is in the exact order the server will run the jobs. `eta_seconds` is a
straight-line guess from elapsed time and ignores evaluation, so treat it as a
floor. `yielding: true` means the running job is about to pause for something
more urgent.

Your own job also carries `queue_position` in `GET /v1/jobs/{id}`.

---

## Tracking your job

```bash
curl -sS $HOST/v1/jobs/$JOB | jq
```

| Field | Meaning |
|---|---|
| `status` | `queued`, `running`, `paused`, `succeeded`, `failed`, `cancelled` |
| `queue_position` | 1-based place in the queue while waiting |
| `preempted_count` | times higher-priority work has bumped this job |
| `progress` | 0.0–1.0, updated every training step |
| `metrics` | populated on success |
| `labels` | inferred label list, in model index order |
| `num_train` / `num_eval` | rows actually used after splitting |
| `error` | why it failed |
| `artifact_bytes` | tarball size |

Live training log:

```bash
curl -sS "$HOST/v1/jobs/$JOB/logs?tail=30"
```

List your jobs, optionally filtered:

```bash
curl -sS "$HOST/v1/jobs?status=running" | jq '.[] | {id, name, progress}'
```

Cancel a queued or running job — cancellation is checked every training step, so
it takes effect within seconds:

```bash
curl -sS -X POST $HOST/v1/jobs/$JOB/cancel
```

---

## Using the model

```bash
curl -sS -OJ $HOST/v1/jobs/$JOB/artifact
tar xzf $JOB.tar.gz          # extracts into a directory named <job_id>/
```

It is a standard HuggingFace model directory:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./<job_id>")
tok = AutoTokenizer.from_pretrained("./<job_id>")
model.eval()

batch = tok(["surprisingly good", "fell apart"], return_tensors="pt", padding=True)
with torch.no_grad():
    probs = model(**batch).logits.softmax(-1)

for row in probs:
    print({model.config.id2label[i]: round(p.item(), 3) for i, p in enumerate(row)})
```

Use `AutoModelForTokenClassification` for NER. For regression the single logit
*is* the prediction — no softmax. For multi-label apply `sigmoid` and compare
against your `threshold`.

**Read `model.config.id2label`; never hardcode label indices.** Indices come
from sorted label order, so adding a class in a later run shifts them.

`training_metrics.json` inside the archive records the eval metrics.

---

## Clean up

```bash
curl -sS -X DELETE $HOST/v1/jobs/$JOB     # 204; removes data, logs and artifact
```

Irreversible. Download first.

---

## Complete Python example

```python
import io, json, tarfile, time, requests

HOST = "$TUNER_URL"

rows = [
    {"text": "great value", "label": "positive"},
    {"text": "broke instantly", "label": "negative"},
    {"text": "works perfectly", "label": "positive"},
    {"text": "waste of money", "label": "negative"},
]
train = "\n".join(json.dumps(r) for r in rows)

job = requests.post(
    f"{HOST}/v1/jobs",
    data={"config": json.dumps({
        "base_model": "microsoft/deberta-v3-base",
        "task": "sequence_classification",
        "epochs": 3,
        "name": "sentiment-v1",
    })},
    files={"train_file": ("train.jsonl", train)},
    timeout=120,
).json()

job_id = job["id"]
print("submitted", job_id)

while True:
    j = requests.get(f"{HOST}/v1/jobs/{job_id}", timeout=30).json()
    print(j["status"], j["progress"])
    if j["status"] in ("succeeded", "failed", "cancelled"):
        break
    time.sleep(10)

if j["status"] != "succeeded":
    raise SystemExit(
        f"{j['status']}: {j['error']}\n"
        + requests.get(f"{HOST}/v1/jobs/{job_id}/logs", params={"tail": 40}).text
    )

print("metrics:", j["metrics"])

blob = requests.get(f"{HOST}/v1/jobs/{job_id}/artifact", timeout=600).content
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
    tar.extractall("./models")
print("extracted to ./models/" + job_id)

requests.delete(f"{HOST}/v1/jobs/{job_id}", timeout=30)
```

---

## Errors

| Code | Meaning | What to do |
|---|---|---|
| `400` | bad `config` JSON, disallowed `base_model`, or invalid data | read `detail`; it names the line |
| `404` | no such job | check the id |
| `409` | artifact requested for a job that did not succeed, cancel of a finished job, or delete of a running one | check `status` first |
| `410` | artifact was deleted | resubmit |
| `413` | upload over 512 MB | shard your dataset |
| `422` | malformed multipart request | check you sent `config` and `train_file` |

If a job reaches `failed`, `error` gives the exception and
`/v1/jobs/{id}/logs` has the full traceback. The usual causes are CUDA
out-of-memory (lower `batch_size` or `max_length`) and label inconsistencies.

---

## Practical notes

- **Model size vs. time.** `deberta-v3-base` on a few thousand short examples
  takes minutes. `deberta-v3-large` is roughly 3× slower and needs a smaller
  `batch_size`.
- **Out of memory?** Halve `batch_size`, then reduce `max_length`. Memory grows
  quadratically with sequence length.
- **Testing your plumbing?** Use `microsoft/deberta-v3-xsmall` with
  `"epochs": 1` — it finishes in well under a minute. Do not judge quality from
  it.
- **Reproducibility.** Same `seed`, same data and same config give the same
  split and initialisation. Exact GPU determinism is not guaranteed.
- **Class imbalance** is not handled automatically — no class weighting or
  resampling. Balance upstream, and prefer `f1_macro` over `accuracy`.
