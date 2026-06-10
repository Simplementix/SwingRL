#!/usr/bin/env bash
# Run baseline + 5 memory-enhanced training iterations.
#
# Usage: bash scripts/run_iterations.sh [extra args passed to train_pipeline.py]
#
# Environment:
#   REPO_DIR   Override repository root (default: directory containing this script's parent)
#   ITERATIONS Override iteration count (default: 5)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ITERATIONS="${ITERATIONS:-5}"

cd "$REPO_DIR"

# Use venv python if available (Docker), otherwise fall back to uv run (local dev)
if [ -x /app/.venv/bin/python ]; then
    exec /app/.venv/bin/python scripts/train_pipeline.py --env all --iterations "$ITERATIONS" --force "$@"
else
    exec uv run python scripts/train_pipeline.py --env all --iterations "$ITERATIONS" --force "$@"
fi
