# Operations

Running and maintaining the service. For calling it, see
[client-guide.md](client-guide.md).

## Host requirements

| | |
|---|---|
| OS | Linux with systemd |
| GPU | one NVIDIA GPU with enough VRAM for your largest base model |
| CUDA | a matching driver; a CUDA toolkit if extensions must be JIT-compiled |
| Python | 3.12, plus `uv` |
| Ports | one TCP port for the API (default 8100) |

If another GPU-using service shares the host, note its systemd unit name — the
tuner cycles it around each job. See [GPU arbitration](architecture.md#gpu-arbitration).

## Deployment

```bash
sudo ./deploy/install-system.sh
```

Idempotent — re-run it after editing the source to re-sync code and restart.

| Step | Action |
|---|---|
| 1 | create the `deberta` system user (nologin, home `/var/lib/deberta-tuner`) |
| 2 | install `uv` to `/usr/local/bin` |
| 3 | sync `app/`, `pyproject.toml` and `uv.lock` to `/opt/deberta-tuner` |
| 4 | `uv sync --locked --no-dev` — install exactly the locked versions |
| 5 | install the sudoers grant, validated with `visudo -c` |
| 6 | install, enable and restart the unit |
| 7 | poll `/healthz` for up to 180 s and fail loudly if it never answers |

Step 5 installs the file, validates it, and **removes it and aborts** if
`visudo` rejects it — a malformed sudoers file can lock out `sudo` entirely.

Layout: code in `/opt/deberta-tuner`, state in `/var/lib/deberta-tuner`, unit at
`/etc/systemd/system/deberta-tuner.service`, grant at
`/etc/sudoers.d/deberta-tuner`.

### Two deliberate choices in the unit

**`NoNewPrivileges` is not set.** It would block setuid `sudo`, and the service
needs `sudo systemctl` to arbitrate the GPU. Setting it breaks arbitration in a
way that only shows up mid-job.

**`PATH` is set explicitly** to include `/opt/deberta-tuner/.venv/bin` and
`/usr/local/cuda/bin`. Invoking a venv binary by absolute path does *not* put
its `bin/` on `PATH`; torch extensions that shell out to `nvcc` or `ninja` fail
with `FileNotFoundError` otherwise. This is a common failure mode for any
venv-based unit.

### The sudoers grant

```
deberta ALL=(root) NOPASSWD: /usr/bin/systemctl stop sglang.service
deberta ALL=(root) NOPASSWD: /usr/bin/systemctl start sglang.service
```

Two exact commands, no wildcards. A wildcard such as `systemctl *` would be
equivalent to full root.

## Day-to-day

```bash
systemctl status deberta-tuner
journalctl -u deberta-tuner -f
systemctl restart deberta-tuner

curl -sS $TUNER_URL/healthz | jq
curl -sS "$TUNER_URL/v1/jobs?status=running" | jq
```

Restarting while a job runs marks that job `failed` — there is no resume. Check
for running jobs first.

Both services together (substitute your configured unit):

```bash
systemctl status sglang deberta-tuner --no-pager
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

## Troubleshooting

### Service will not start

```bash
journalctl -u deberta-tuner -n 50 --no-pager
```

- `FileNotFoundError: 'ninja'` or `'nvcc'` → `PATH` in the unit lost its venv or
  CUDA entry.
- `ModuleNotFoundError` → `/opt/deberta-tuner/app` is stale; re-run the
  installer.
- Port in use → something else holds 8100 (`ss -tlnp | grep 8100`).

### Jobs fail with CUDA out of memory

Confirm the co-tenant service actually stopped:

```bash
grep -E "systemctl|GPU free" /var/lib/deberta-tuner/jobs/<job_id>/train.log
```

Expect a `systemctl stop ...: ok` line followed by a `GPU free:` figure large
enough to train. If free memory is still small, arbitration failed — check the sudoers grant and that
`NoNewPrivileges` is not set:

```bash
sudo -u deberta sudo -n systemctl stop sglang.service   # should succeed silently
systemctl show deberta-tuner -p NoNewPrivileges
```

If arbitration works and it still OOMs, the job is genuinely too large: lower
`batch_size`, then `max_length`, or use a smaller base model.

### The co-tenant service did not come back

The restart is in a `finally`, so this should be rare.

```bash
systemctl status sglang
grep systemctl /var/lib/deberta-tuner/jobs/<job_id>/train.log
sudo systemctl start sglang
```

If it hit the start-rate limiter (`Start request repeated too quickly`):

```bash
sudo systemctl reset-failed sglang && sudo systemctl start sglang
```

### `Found dtype Float but expected Half`

Already fixed, documented in case it resurfaces. transformers ≥5 loads
checkpoints in their **native dtype**, and several DeBERTa checkpoints are fp16.
The regression head does `logits.to(labels.dtype)`, so backward pushes a Float
gradient into a Half tensor.

`app/training.py` forces fp32 after loading. If you refactor model construction,
keep the `.to(dtype=torch.float32)`. It was also silently degrading token
classification — seqeval F1 went 0.0 → 0.67 once fixed.

### Jobs stuck in `queued`

The worker thread died, or a job ahead is still running.

```bash
curl -sS "$TUNER_URL/v1/jobs?status=running" | jq
```

Nothing running and nothing progressing means a dead worker; restart the
service. Queued jobs survive, since the queue is in SQLite.

### Every job says "service restarted while this job was running"

Expected after a restart during training, and applied at startup by
`db.init()`. If it happens repeatedly the unit is crash-looping — check
`journalctl` and `Restart=always` behaviour.

## Disk usage

Nothing is cleaned up automatically. Each job keeps its uploads, the model
directory *and* the tarball, so a `deberta-v3-large` job costs roughly 3.5 GB.

```bash
du -sh /var/lib/deberta-tuner/jobs/* | sort -h | tail
df -h /
```

Reclaim by deleting jobs through the API, which removes the row and the files:

```bash
curl -sS -X DELETE $TUNER_URL/v1/jobs/<job_id>
```

Bulk-delete everything terminal older than a week:

```bash
curl -sS "$TUNER_URL/v1/jobs?limit=1000" \
  | jq -r --arg cut "$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%S)" \
      '.[] | select(.status=="succeeded" or .status=="failed" or .status=="cancelled")
           | select(.created_at < $cut) | .id' \
  | xargs -r -I{} curl -sS -X DELETE $TUNER_URL/v1/jobs/{}
```

Deleting the directory by hand leaves an orphan row; prefer the API.

The base-model cache in `/var/lib/deberta-tuner/hf-cache` grows once per distinct
base model (~1.7 GB for all seven) and is safe to leave.

## Security posture

The service binds `0.0.0.0` with no authentication, so anyone who can reach the
port can submit jobs, read logs, download models and delete jobs. Deploy it on a
trusted network, or put authentication in front of it.

Bounded by: the `base_model` allow-list, the upload cap, and the two-command
sudoers grant. Not bounded: job submission rate, disk consumption, or who can
take the co-tenant GPU service down.

To close it, see
[configuration.md](configuration.md#adding-authentication), or bind `127.0.0.1`
and front it with something authenticating.

## Backups

Worth preserving: `/var/lib/deberta-tuner/jobs.db` and any `model.tar.gz` you
care about. Uploads and logs are reproducible; the HF cache is re-downloadable.

## Upgrading

```bash
# edit app/... in your checkout, then
TUNER_FORCE_CPU=1 python -m tests.smoke
sudo ./deploy/install-system.sh
```

The installer re-syncs code, runs `uv sync --locked --no-dev` and restarts.
`--locked` asserts the lockfile is current for `pyproject.toml` and aborts if it
is not, so a forgotten `uv lock` fails the deploy instead of shipping stale
dependencies. Check for running jobs first — a restart fails them.

> Use `--locked`, not `--frozen`. `--frozen` installs from the lockfile without
> checking it matches `pyproject.toml`, so a stale lock deploys silently and
> exits 0.

To change a dependency: edit `pyproject.toml`, run `uv lock`, re-run the test
suites, commit `pyproject.toml` **and** `uv.lock`, then deploy.

Consider pinning the NVIDIA driver packages (`apt-mark hold` or equivalent). An
unattended driver upgrade desynchronises the running kernel module from
userspace, which breaks CUDA for every process until the host reboots.
