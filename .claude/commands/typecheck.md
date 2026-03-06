---
description: Run mypy type checking on src/. Usage: /project:typecheck [path]
---

Run mypy type checking natively using uv.

```bash
uv run mypy ${ARGUMENTS:-src/}
```

Common patterns:
- Full src: `/project:typecheck`
- Single module: `/project:typecheck src/swingrl/config/schema.py`
- Strict mode: `uv run mypy src/ --strict`

Note: mypy is configured in pyproject.toml with disallow_untyped_defs=true.
Overrides ignore missing stubs for finrl, hmmlearn, alpaca, stockstats, pydantic_settings.
