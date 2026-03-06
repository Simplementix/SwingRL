---
description: Run pytest suite natively on Mac (fast, no Docker). Usage: /project:test [path] [-k filter] [-x]
---

Run SwingRL tests natively using uv. Faster than Docker for development iteration.

```bash
uv run pytest $ARGUMENTS -v
```

Common patterns:
- Full suite: `/project:test`
- Smoke tests only: `/project:test tests/test_smoke.py`
- Config tests: `/project:test tests/test_config.py`
- Single test by name: `/project:test -k test_valid_config_loads`
- Stop on first failure: `/project:test -x`
- With Docker instead: `docker compose run --rm swingrl pytest $ARGUMENTS -v`
