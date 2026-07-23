#!/usr/bin/env bash
# Build/refresh the pre-migrated test-DB template (swingrl_test_template) and
# reap stale per-worker clones. Run on homelab from the repo root whenever a
# migration ships (EXPECTED_SCHEMA_VERSION changes) or the template goes stale.
# Requires: docker access to pg16, a .env with the swingrl DATABASE_URL password.
set -euo pipefail

PG_CONTAINER="${PG_CONTAINER:-pg16}"
TEMPLATE_DB="swingrl_test_template"
ENV_FILE="${ENV_FILE:-.env}"

PG_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$PG_CONTAINER")
PG_PASS=$(grep DATABASE_URL "$ENV_FILE" | sed 's/.*:\/\/swingrl:\([^@]*\)@.*/\1/')

echo "=== [1/4] Grant CREATEDB to swingrl (idempotent; workers clone their own DBs) ==="
docker exec "$PG_CONTAINER" psql -U temporal -d postgres -c "ALTER ROLE swingrl CREATEDB;"

echo "=== [2/4] Recreate ${TEMPLATE_DB} ==="
docker exec "$PG_CONTAINER" psql -U temporal -d postgres \
    -c "DROP DATABASE IF EXISTS ${TEMPLATE_DB} WITH (FORCE);"
docker exec "$PG_CONTAINER" psql -U temporal -d postgres \
    -c "CREATE DATABASE ${TEMPLATE_DB} OWNER swingrl;"

echo "=== [3/4] Apply schema + migrations to ${TEMPLATE_DB} ==="
DATABASE_URL="postgresql://swingrl:${PG_PASS}@${PG_IP}:5432/${TEMPLATE_DB}" uv run python -c "
from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import apply_migrations, current_schema_version
db = DatabaseManager(load_config('config/swingrl.yaml'))
db.init_schema()
n = apply_migrations(db)
print(f'applied={n} current_version={current_schema_version(db)}')
"

echo "=== [4/4] Reap stale worker/session clones ==="
docker exec "$PG_CONTAINER" psql -U temporal -d postgres -tAc \
    "SELECT datname FROM pg_database WHERE datname ~ '^swingrl_([a-z0-9]+_)*(gw|main)[0-9]+_test$'" |
while read -r db; do
    [ -n "$db" ] && docker exec "$PG_CONTAINER" psql -U temporal -d postgres \
        -c "DROP DATABASE IF EXISTS \"$db\" WITH (FORCE);"
done

echo "=== TEMPLATE READY: ${TEMPLATE_DB} on ${PG_CONTAINER} (${PG_IP}) ==="
