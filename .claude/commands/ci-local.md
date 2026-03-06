---
description: Run full CI quality gate natively on Mac — mirrors ci-homelab.sh stages 2-5 (4 native stages) without Docker build. Usage: /project:ci-local
---

Run the same quality checks as ci-homelab.sh stages 2-5 but natively for fast pre-push validation.
Skips Stage 1 (Docker build) — run actual ci-homelab.sh via ssh for full pre-merge validation.

```bash
set -e

echo "=== [1/4] Tests ==="
uv run pytest tests/ -v

echo "=== [2/4] Lint ==="
uv run ruff check .
uv run ruff format --check .

echo "=== [3/4] Type check ==="
uv run mypy src/

echo "=== [4/4] Security ==="
uv run bandit -r src/ -c pyproject.toml -q

echo ""
echo "=== CI LOCAL PASSED — all 4 stages green ==="
echo "For full validation (with Docker build): ssh homelab 'bash ~/swingrl/scripts/ci-homelab.sh'"
```

Stages match ci-homelab.sh stages 2-5. Stage 1 (docker compose build) is skipped for speed.
