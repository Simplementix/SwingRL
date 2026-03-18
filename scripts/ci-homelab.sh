#!/usr/bin/env bash
# ci-homelab.sh — Full CI quality gate for the SwingRL homelab server.
#
# Stages:
#   [1/6] Git pull              — fast-forward only to catch diverged branches early
#   [2/6] Docker build          — cached by default; pass --no-cache for clean build
#   [3/6] Run tests             — pytest inside container (MPS test skipped on Linux)
#   [4/6] Lint + types          — ruff check + ruff format --check + mypy inside container
#   [4a/6] Memory service lint  — build swingrl-memory and run ruff + mypy inside it
#   [5/6] Cleanup               — docker compose down + prune dangling images
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

echo "=== [1/6] Git pull ==="
git pull --ff-only

echo "=== [2/6] Docker build ==="
if [[ "$NO_CACHE" == "--no-cache" ]]; then
    docker compose build --no-cache
    docker compose -f docker-compose.prod.yml build --no-cache swingrl-memory
else
    docker compose build
    docker compose -f docker-compose.prod.yml build swingrl-memory
fi

echo "=== [3/6] Run tests ==="
docker compose run --rm --entrypoint "" swingrl uv run pytest tests/ -v

echo "=== [4/6] Lint + type check ==="
docker compose run --rm --entrypoint "" swingrl uv run sh -c \
    'ruff check . && ruff format --check . && mypy src/'

echo "=== [4a/6] Memory service lint ==="
# Lint services/memory/ inside the swingrl-memory container (no Ollama needed for linting)
docker compose -f docker-compose.prod.yml run --rm --no-deps --entrypoint "" swingrl-memory sh -c \
    'pip install ruff==0.15.5 mypy==1.19.1 -q && ruff check /app && ruff format --check /app && mypy /app --ignore-missing-imports'

echo "=== [5/6] Cleanup ==="
docker compose down
docker compose -f docker-compose.prod.yml down
docker image prune -f

echo "=== CI PASSED ==="
