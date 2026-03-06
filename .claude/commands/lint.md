---
description: Run ruff lint + ruff format check + bandit security scan. Usage: /project:lint [path]
---

Run all linting checks natively using uv.

```bash
uv run ruff check ${ARGUMENTS:-.}
uv run ruff format --check ${ARGUMENTS:-.}
uv run bandit -r src/ -c pyproject.toml -q
```

Common patterns:
- Full repo: `/project:lint`
- Single file: `/project:lint src/swingrl/config/schema.py`
- Auto-fix ruff issues: `uv run ruff check --fix .`
