#!/usr/bin/env bash
#
# Install the DeBERTa tuning service system-wide on qgpu1.
#
#   sudo ~/Q/dev/deberta-tuner/deploy/install-system.sh
#
# Idempotent: safe to re-run after editing the app source (it re-syncs the code
# and restarts the unit).
#
set -euo pipefail

SRC_DIR="$(dirname "$(readlink -f "$0")")"
PROJ_DIR="$(dirname "$SRC_DIR")"
PREFIX=/opt/deberta-tuner
STATE=/var/lib/deberta-tuner
SVC_USER=deberta
PORT=8100

[ "$(id -u)" -eq 0 ] || { echo "must run as root (use sudo)" >&2; exit 1; }

say() { printf '\n=== %s ===\n' "$*"; }

say "1/7 creating the ${SVC_USER} system user"
if ! id "$SVC_USER" >/dev/null 2>&1; then
    useradd --system --create-home --home-dir "$STATE" \
            --shell /usr/sbin/nologin --comment "DeBERTa tuning service" "$SVC_USER"
else
    echo "  already exists"
fi
usermod -aG video,render "$SVC_USER"
install -d -o "$SVC_USER" -g "$SVC_USER" -m 0755 "$PREFIX" "$STATE"

say "2/7 locating uv"
UV=""
for cand in /usr/local/bin/uv "$SRC_DIR/uv" "$(command -v uv 2>/dev/null || true)"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then UV="$cand"; break; fi
done
if [ -z "$UV" ]; then
    cat >&2 <<'MSG'
  uv not found. Install it system-wide, for example:

      curl -LsSf https://astral.sh/uv/install.sh \
        | sudo env UV_INSTALL_DIR=/usr/local/bin sh

  or place a uv binary at deploy/uv next to this script.

  Note a user-local uv under ~/.local is often NOT usable here: root cannot
  traverse a 0700 home directory, and NFS root_squash makes that worse.
MSG
    exit 1
fi
# Put it somewhere the unprivileged service user can also execute it.
if [ "$UV" != /usr/local/bin/uv ]; then
    install -m 0755 "$UV" /usr/local/bin/uv
fi
/usr/local/bin/uv --version

say "3/7 syncing application code to ${PREFIX}"
rm -rf "$PREFIX/app"
cp -r "$PROJ_DIR/app" "$PREFIX/app"
# pyproject.toml + uv.lock are the dependency source of truth; both must ship.
cp "$PROJ_DIR/pyproject.toml" "$PROJ_DIR/uv.lock" "$PROJ_DIR/README.md" "$PREFIX/"
find "$PREFIX/app" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
chown -R "$SVC_USER:$SVC_USER" \
    "$PREFIX/app" "$PREFIX/pyproject.toml" "$PREFIX/uv.lock" "$PREFIX/README.md"

say "4/7 syncing the venv at ${PREFIX}/.venv from uv.lock"
# --locked: assert the lockfile is current for this pyproject.toml and abort if
#   not. NOTE: --frozen would NOT do this -- it installs from the lockfile
#   without checking it is in step, so a forgotten `uv lock` would deploy stale
#   dependencies and still exit 0.
# --no-dev: skip the dev group (httpx, needed only by the test suite).
# -H: without it sudo keeps root's HOME and uv writes its cache to /root/.cache
#   as the service user, which fails.
sudo -u "$SVC_USER" -H /usr/local/bin/uv sync \
    --locked --no-dev --python 3.12 --project "$PREFIX"

say "5/7 installing the sudoers grant for sglang control"
# Validate before installing: a malformed sudoers file can lock out sudo.
install -m 0440 "$SRC_DIR/deberta-tuner.sudoers" /etc/sudoers.d/deberta-tuner
if ! visudo -cf /etc/sudoers.d/deberta-tuner; then
    rm -f /etc/sudoers.d/deberta-tuner
    echo "sudoers file rejected; removed it and aborting" >&2
    exit 1
fi

say "6/7 installing the unit"
install -m 0644 "$SRC_DIR/deberta-tuner.service" /etc/systemd/system/deberta-tuner.service
systemctl daemon-reload
systemctl enable deberta-tuner.service
systemctl reset-failed deberta-tuner.service 2>/dev/null || true
systemctl restart deberta-tuner.service

say "7/7 waiting for the API to answer"
ready=0
deadline=$((SECONDS + 180))
while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/healthz" 2>/dev/null; then
        ready=1
        break
    fi
    if ! systemctl is-active --quiet deberta-tuner.service; then
        echo "  service died during startup; last 40 log lines:" >&2
        journalctl -u deberta-tuner.service -n 40 --no-pager >&2
        exit 1
    fi
    sleep 3
done
[ "$ready" -eq 1 ] || {
    echo "  timed out waiting for /healthz" >&2
    journalctl -u deberta-tuner.service -n 40 --no-pager >&2
    exit 1
}

echo "  API is answering"
curl -fsS "http://127.0.0.1:${PORT}/healthz"; echo

cat <<EOF

Done. Enabled at boot, listening on 0.0.0.0:${PORT}.

  systemctl status deberta-tuner
  journalctl -u deberta-tuner -f
  curl http://10.10.2.126:${PORT}/healthz
  curl http://10.10.2.126:${PORT}/docs      # interactive OpenAPI UI

Submitting a job stops sglang.service for the duration and restarts it after.
Reminder: bound to 0.0.0.0 with no auth, same as sglang. ufw is off.
EOF
