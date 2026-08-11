"""Runtime configuration, all overridable by environment variable."""
from __future__ import annotations

import os
from pathlib import Path

# Where jobs, uploads, logs and artifacts live. Defaults to a local dir for
# development; the systemd unit points this at /var/lib/deberta-tuner.
STATE_DIR = Path(os.environ.get("TUNER_STATE_DIR", "./state")).resolve()
JOBS_DIR = STATE_DIR / "jobs"
DB_PATH = STATE_DIR / "jobs.db"

HOST = os.environ.get("TUNER_HOST", "0.0.0.0")
PORT = int(os.environ.get("TUNER_PORT", "8100"))

MAX_UPLOAD_MB = int(os.environ.get("TUNER_MAX_UPLOAD_MB", "512"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# qgpu1 has a single 4090 shared with the SGLang server. Training stops that
# unit for the duration of a job and restarts it afterwards. Requires the
# sudoers grant installed by deploy/install-system.sh.
SGLANG_UNIT = os.environ.get("TUNER_SGLANG_UNIT", "sglang.service")
SGLANG_CONTROL = os.environ.get("TUNER_SGLANG_CONTROL", "1") == "1"

# Force CPU regardless of availability (used by the test suite).
FORCE_CPU = os.environ.get("TUNER_FORCE_CPU", "0") == "1"

# Only these bases may be requested. An open field would let a caller pull
# arbitrary code off the hub via trust_remote_code-style model configs.
ALLOWED_BASE_MODELS = set(
    filter(
        None,
        os.environ.get(
            "TUNER_ALLOWED_BASE_MODELS",
            ",".join(
                [
                    "microsoft/deberta-v3-xsmall",
                    "microsoft/deberta-v3-small",
                    "microsoft/deberta-v3-base",
                    "microsoft/deberta-v3-large",
                    "microsoft/mdeberta-v3-base",
                    "microsoft/deberta-base",
                    "microsoft/deberta-large",
                ]
            ),
        ).split(","),
    )
)


def ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
