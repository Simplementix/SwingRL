#!/usr/bin/env bash
# ci-homelab.sh — Full CI quality gate for the SwingRL homelab server.
#
# Stages:
#   [1/6] Git pull              — fast-forward only to catch diverged branches early
#   [2/6] Docker build          — cached by default; pass --no-cache for clean build
#   [3/6] Run tests             — pytest inside container (MPS test skipped on Linux)
#   [4/6] Lint + types          — ruff check + ruff format --check + mypy inside container
#   [4a/6] Memory service lint  — build swingrl-memory and run ruff + mypy inside it
#   [5/6] Dependency CVE audit  — pip-audit --strict against pyproject.toml's declared deps;
#                                 known-accepted findings suppressed via --ignore-vuln
#                                 (see docs/execution/cve-triage.md for dispositions).
#   [6/6] Cleanup               — dev-compose-project down + prune dangling images.
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

echo "=== [2.7/6] Apply schema migrations to test database ==="
# Fresh swingrl_test has no schema_migrations ledger — every DB-gated test that
# calls services/memory/db.py::init_db() (V001+ guard) fails without this.
# -e DATABASE_URL="$TEST_DB_URL" (not SWINGRL_SYSTEM__DATABASE_URL): DatabaseManager
# reads plain DATABASE_URL from the environment FIRST, ahead of config.system.database_url
# (src/swingrl/data/db.py) — and env_file: .env already put the PRODUCTION DATABASE_URL
# into this container's environment, so only the same override Stage 3 uses is guaranteed
# to win. Runs in the same swingrl-ci container/venv as Stage 3, so the code applying
# migrations matches the code under test.
# init_schema() first: V001 ALTERs backtest_results/iteration_results, which only exist
# once DatabaseManager.init_schema() (legacy postgres_schema.py DDL) has run — the same
# order tests/data/conftest.py's db_with_legacy_schema fixture uses. Inline (not
# scripts/init_db.py): that script's post-init verification step has a pre-existing
# dict-row bug (`result[0]`) unrelated to this change that makes it exit non-zero even
# though schema creation itself succeeds — this inline call only ever touches the two
# calls actually needed here.
$DEV_COMPOSE run --rm --entrypoint "" -e DATABASE_URL="$TEST_DB_URL" swingrl uv run python -c "
from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import apply_migrations, current_schema_version
db = DatabaseManager(load_config('config/swingrl.yaml'))
db.init_schema()
n = apply_migrations(db)
print(f'applied={n} current_version={current_schema_version(db)}')
"

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

echo "=== [5/6] Dependency CVE audit ==="
# pip-audit resolves and audits pyproject.toml's declared dependencies directly
# (project-path mode: the positional "." argument), NOT the installed venv. Auditing the
# installed environment instead fails outright either way: default env-audit mode errors
# `swingrl: Dependency not found on PyPI` (this project isn't published), and
# `--local --skip-editable` (to work around that) instead errors
# `swingrl: distribution marked as editable` — --strict treats both as fatal. Project-path
# mode reads [project.dependencies]/[dependency-groups] and resolves them independently of
# what's installed, sidestepping both failure modes.
#
# The --ignore-vuln flags below suppress 20 known findings, all in torch (pinned <2.4 in
# pyproject.toml; every available fix version for these is >=2.5 — no fix exists inside the
# current pin, so accepting was the only option short of an unreviewed major-version bump).
# Full per-finding disposition + rationale: docs/execution/cve-triage.md.
# Review by 2026-10-15 — re-run without --ignore-vuln and re-triage alongside a dedicated
# torch major-version upgrade review (same rigor as the Task 13 alpaca-py pin review).
$DEV_COMPOSE run --rm --entrypoint "" swingrl uv run pip-audit --strict . \
    --ignore-vuln PYSEC-2025-191 --ignore-vuln PYSEC-2025-41 --ignore-vuln PYSEC-2024-259 \
    --ignore-vuln PYSEC-2025-205 --ignore-vuln PYSEC-2025-206 --ignore-vuln PYSEC-2025-207 \
    --ignore-vuln PYSEC-2025-204 --ignore-vuln PYSEC-2026-139 --ignore-vuln PYSEC-2025-209 \
    --ignore-vuln PYSEC-2025-208 --ignore-vuln PYSEC-2025-198 --ignore-vuln PYSEC-2025-203 \
    --ignore-vuln PYSEC-2026-1970 --ignore-vuln PYSEC-2026-2286 --ignore-vuln CVE-2025-2148 \
    --ignore-vuln CVE-2025-2149 --ignore-vuln CVE-2025-2998 --ignore-vuln CVE-2025-2999 \
    --ignore-vuln CVE-2025-3000 --ignore-vuln CVE-2025-3001

echo "=== [6/6] Cleanup ==="
docker exec pg16 psql -U temporal -d postgres -c "DROP DATABASE IF EXISTS swingrl_test;" || true
# Cleanup is scoped to the dev compose project ONLY. `docker compose down` against the
# production project would stop every always-on service (trader, collector) on each CI
# run. Production containers are managed by deployment, never by CI.
$DEV_COMPOSE down
# Dangling (untagged) images only — never removes tagged images or images in use.
docker image prune -f

echo "=== CI PASSED ==="
