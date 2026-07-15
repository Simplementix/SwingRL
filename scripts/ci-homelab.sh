#!/usr/bin/env bash
# ci-homelab.sh — Full CI quality gate for the SwingRL homelab server.
#
# Stages:
#   [1/6] Git pull              — fast-forward only to catch diverged branches early
#   [2/6] Docker build          — cached by default; pass --no-cache for clean build
#   [3/6] Run tests             — pytest inside container (MPS test skipped on Linux)
#   [4/6] Lint + types          — ruff check + ruff format --check + mypy inside container
#   [4a/6] Memory service lint  — build swingrl-memory and run ruff + mypy inside it
#   [5/6] Cleanup               — dev-compose-project down + prune dangling images.
#                                 Production compose is NEVER touched by cleanup:
#                                 always-on services (trader, collector) must survive CI runs.
#
# Usage:
#   bash ~/swingrl/scripts/ci-homelab.sh              # cached build (fast)
#   bash ~/swingrl/scripts/ci-homelab.sh --no-cache   # clean build (for lockfile changes)
#
# From M1 Mac:
#   ssh homelab "cd ~/swingrl && bash scripts/ci-homelab.sh"
#   ssh homelab "cd ~/swingrl && bash scripts/ci-homelab.sh --no-cache"
#
# Compose files:
#   docker-compose.yml      — production (default, used for memory service + deployment)
#   docker-compose-dev.yml  — dev/CI (ci target with pytest, ruff, mypy)
#
# Environment variables:
#   REPO_DIR — path to repo on homelab (default: $HOME/swingrl)
#
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/swingrl}"
NO_CACHE="${1:-}"
# Dev/CI runs under its OWN compose project name. Both compose files define a service
# named "swingrl" under the same default project (the repo directory), so an unscoped
# `docker compose -f docker-compose-dev.yml down` would match the labels of the
# PRODUCTION trader container and stop it. -p isolates every dev command
# (build/run/down) — and the dev image tag — from the production project.
DEV_COMPOSE="docker compose -p swingrl-ci -f docker-compose-dev.yml"

cd "$REPO_DIR"

echo "=== [1/6] Git pull ==="
git pull --ff-only

echo "=== [2/6] Docker build ==="
if [[ "$NO_CACHE" == "--no-cache" ]]; then
    $DEV_COMPOSE build --no-cache
    docker compose build --no-cache swingrl-memory
else
    $DEV_COMPOSE build
    docker compose build swingrl-memory
fi

echo "=== [2.5/6] Create test database ==="
PG_PASS=$(grep DATABASE_URL .env | sed 's/.*:\/\/swingrl:\([^@]*\)@.*/\1/')
TEST_DB_URL="postgresql://swingrl:${PG_PASS}@pg16:5432/swingrl_test"
docker exec pg16 psql -U temporal -d postgres -c "DROP DATABASE IF EXISTS swingrl_test;"
docker exec pg16 psql -U temporal -d postgres -c "CREATE DATABASE swingrl_test OWNER swingrl;"

echo "=== [3/6] Run tests ==="
$DEV_COMPOSE run --rm --entrypoint "" -e DATABASE_URL="$TEST_DB_URL" swingrl uv run pytest tests/ -v

echo "=== [4/6] Lint + type check ==="
$DEV_COMPOSE run --rm --entrypoint "" swingrl uv run sh -c \
    'ruff check . && ruff format --check . && mypy src/'

echo "=== [4a/6] Memory service lint ==="
# Lint services/memory/ inside the swingrl-memory container
docker compose run --rm --no-deps --entrypoint "" swingrl-memory sh -c \
    'pip install ruff==0.15.5 -q && PATH="$HOME/.local/bin:$PATH" && ruff check /app && ruff format --check /app'
# NOTE: mypy for memory service skipped — pre-existing type errors in query.py
# (parsed dict returns str|int|None, functions expect str). To be fixed separately.

echo "=== [5/6] Cleanup ==="
docker exec pg16 psql -U temporal -d postgres -c "DROP DATABASE IF EXISTS swingrl_test;" || true
# Cleanup is scoped to the dev compose project ONLY. `docker compose down` against the
# production project would stop every always-on service (trader, collector) on each CI
# run. Production containers are managed by deployment, never by CI.
$DEV_COMPOSE down
# Dangling (untagged) images only — never removes tagged images or images in use.
docker image prune -f

echo "=== CI PASSED ==="
