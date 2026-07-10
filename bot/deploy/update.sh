#!/usr/bin/env bash
# VPS-side deploy script, invoked over SSH by GitHub Actions (or manually):
#   bot/deploy/update.sh [ref]
# Pulls code anonymously over https, syncs the venv, restarts the user unit.
# GUARD: never touches .env or data/ (they exist only on the VPS).
set -euo pipefail

REF="${1:-main}"
APP_DIR="/opt/traderbot"

cd "$APP_DIR"

# hard guard: refuse to run if .env/data are tracked by git (they must not be)
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
    echo "FATAL: .env is tracked by git — aborting" >&2
    exit 1
fi

git fetch origin "$REF"
git reset --hard FETCH_HEAD

if [ ! -d venv ]; then
    python3.11 -m venv venv
fi
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r bot/requirements.txt

cp -f bot/deploy/bot.service "$HOME/.config/systemd/user/bot.service"
systemctl --user daemon-reload
systemctl --user restart bot
sleep 5
systemctl --user is-active bot
echo "deployed $(git rev-parse --short HEAD) at $(date -u +%FT%TZ)"
# the bot self-reports "started, config sha256=..." to Telegram — CI never sees the token
