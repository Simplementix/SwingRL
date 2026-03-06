#!/usr/bin/env bash
# ci-homelab.sh — Full CI quality gate for the SwingRL homelab server.
#
# Stages:
#   [1/5] Git pull     — fast-forward only to catch diverged branches early
#   [2/5] Docker build — cached by default; pass --no-cache for clean build
#   [3/5] Run tests    — pytest inside container (MPS test skipped on Linux)
#   [4/5] Lint + types — ruff check + ruff format --check + mypy inside container
#   [5/5] Cleanup      — docker compose down + prune dangling images
#
# Usage:
#   bash ~/swingrl/scripts/ci-homelab.sh              # cached build (fast)
#   bash ~/swingrl/scripts/ci-homelab.sh --no-cache   # clean build (for lockfile changes)
#
# From M1 Mac:
#   ssh homelab "cd ~/swingrl && bash scripts/ci-homelab.sh"
#   ssh homelab "cd ~/swingrl && bash scripts/ci-homelab.sh --no-cache"
#
# Environment variables:
#   REPO_DIR — path to repo on homelab (default: $HOME/swingrl)
#
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/swingrl}"
NO_CACHE="${1:-}"

cd "$REPO_DIR"

echo "=== [1/5] Git pull ==="
git pull --ff-only

echo "=== [2/5] Docker build ==="
if [[ "$NO_CACHE" == "--no-cache" ]]; then
    docker compose build --no-cache
else
    docker compose build
fi

echo "=== [3/5] Run tests ==="
docker compose run --rm --entrypoint "" swingrl uv run pytest tests/ -v

echo "=== [4/5] Lint + type check ==="
docker compose run --rm --entrypoint "" swingrl uv run sh -c \
    'ruff check . && ruff format --check . && mypy src/'

echo "=== [5/5] Cleanup ==="
docker compose down
docker image prune -f

echo "=== CI PASSED ==="
